#' @title Train the SETRED model
#' @description Builds and trains a model to predict the label of instances,
#' according to SETRED algorithm.
#' @param x A object that can be coerced as matrix.
#' This object have various interpretations depending on the value set in \code{dist} argument.
#' See \code{dist} argument.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param bclassif Base classifier specification. Default is \code{bClassifOneNN()}. For defining new base clasifiers
#' see \code{\link{bClassif}}.
#' @param dist Distance information. Valid options are:
#' \itemize{
#'    \item \code{"matrix"}: this string indicates that \code{x} is a distance matrix.
#'    \item \emph{string}: the name of a distance method available in \code{proxy} package.
#'    In this case \code{x} is interpreted as a matrix of instances.
#'    \item \emph{function}: a function defined by the user that computes the distance between two vectors.
#'    This function is called passing the vectors in the firsts two arguments. If the function have others arguments,
#'    those arguments must be have default values.
#'    In this case \code{x} is interpreted as a matrix of instances.
#' }
#' @param theta Rejection threshold to test the critical region. Default is 0.1.
#' @param min.amount Minimum number of unlabeled instances for stopping training. When the amount
#' of remaining instances reaches this number the self-labeling process is stopped. Default is
#' 0.3 * <number of unlabeled instances>.
#' @param max.iter Maximum number of iterations in the self-labeling process. Default is 50.
#' @return The trained model stored in a list with the following named values:
#' \describe{
#'   \item{base.m}{The model trained.}
#'   \item{bclassif}{The value of \code{bclassif} argument.}
#'   \item{dist}{The value of \code{dist} argument.
#'   This value is optionally returned when the argument \code{dist} is different from "matrix".}
#'   \item{included.insts}{The indexes of the training instances used to train \code{base.m} model.
#'   Those indexes are relative to \code{x} argument.}
#'   \item{tr.insts}{The training instances corresponding to the indexes returned in \code{included.insts}.
#'    This value is optionally returned when the \code{dist} argument is different from "matrix".}
#'   \item{labels.map}{An internal map used for the labels.}
#' }
#' @references
#' Ming Li and ZhiHua Zhou.\cr
#' \emph{Setred: Self-training with editing.}\cr
#' In Advances in Knowledge Discovery and Data Mining, volume 3518 of Lecture Notes in
#' Computer Science, pages 611–621. Springer Berlin Heidelberg, 2005.
#' ISBN 978-3-540-26076-9. doi: 10.1007/11430919 71.
#' @examples
#' # This example is part of SETRED demo.
#' # Use demo(SETRED) to see all the examples.
#'
#' ## Load Wine data set
#' data(wine)
#'
#' x <- wine[, -14] # instances without classes
#' y <- wine[, 14] # the classes
#' x <- scale(x) # scale the attributes
#'
#' ## Prepare data
#' set.seed(20)
#' # Use 50% of instances for training
#' tra.idx <- sample(x = length(y), size = ceiling(length(y) * 0.5))
#' xtrain <- x[tra.idx,] # training instances
#' ytrain <- y[tra.idx]  # classes of training instances
#' # Use 70% of train instances as unlabeled set
#' tra.na.idx <- sample(x = length(tra.idx), size = ceiling(length(tra.idx) * 0.7))
#' ytrain[tra.na.idx] <- NA # remove class information of unlabeled instances
#'
#' # Use the other 50% of instances for inductive testing
#' tst.idx <- setdiff(1:length(y), tra.idx)
#' xitest <- x[tst.idx,] # testing instances
#' yitest <- y[tst.idx] # classes of testing instances
#'
#' ## Example: Using the Euclidean distance in proxy package.
#' m <- setred(xtrain, ytrain, dist = "Euclidean")
#' pred <- predict(m, xitest)
#' caret::confusionMatrix(table(pred, yitest))
#'
#' @export
setred <- function(
  x, y,
  bclassif = bClassifOneNN(),  dist = "matrix",
  theta = 0.1,
  min.amount = ceiling(length(which(is.na(y))) * 0.3),
  max.iter = 50
) {
  # Check common parameters
  dists.l <- NULL # initialized for checkCommonParams
  dist.is.matrix <- NULL # initialized for checkCommonParams
  checkCommonParams(environment())
  # Check specifics parameters
  if(!(is.numeric(min.amount) && length(min.amount) == 1)) stop("min.amount must be a numeric value")
  if(!(min.amount > 0)) warning("min.amount is not a positive number")
  if(!(is.numeric(max.iter) && length(max.iter) == 1)) stop("max.iter must be a numeric value")
  if(!(max.iter > 0)) stop("max.iter must be a positive value")
  if(!(theta >= 0 && theta <= 1)) stop("theta must be between 0 and 1")

  getDist <- if(dist.is.matrix){
    function(i, j) { x[i,j] }
  }else{
    function(i, j) { dists.l[[1]][i,j] }
  }

  # Lista donde se almacena el modelo
  m <- list()
  class(m) <- "setred"
  m$bclassif <- bclassif

  #calcular los ratio de cada clase para determinar las cantidades a seleccionar en cada iteracion
  yy <- y[!is.na(y)]
  m <- normalizeLabels(m, yy)
  indexInstTL <- cbind(which(!is.na(y)), mapLabels(m, yy))
  indexInstTU <- which(is.na(y))

  # cantidad de clases
  classes <- length(m$labels.map)
  inicialLen <- length(yy)

  #guardar las cant por cada clase
  cantClass <- sapply(1:classes, function(c) length(which(indexInstTL[,2] == c)))
  # calcular proporcion de cada clase
  proportion <- cantClass / inicialLen
  cantClass <- floor(cantClass / min(cantClass)) # divido por el valor minimo
  totalPerIter <- sum(cantClass)

  iter <- 1
  while ((length(indexInstTU) > min.amount) && (length(indexInstTU) > totalPerIter) && (iter <= max.iter)) {

    # entrenar clasificador
    tra.inds <- indexInstTL[,1]
    base.m <- runTrain(bclassif, dists.l, tra.inds, indexInstTL[,2])

    # escojo las instancias con mayores probabilidades de pertenecer a cada clase
    prob <- runPredict(bclassif$predProb, base.m, dists.l, indexInstTU, tra.inds)
    indexInstCand <- selectInst(cantClass, prob)
    # construir L'
    labeledPrima <- matrix()

    for (j in 1:classes) { # para cada clase
      cand <- indexInstCand[[j]]
      # añadir candidatos a L'
      if (j != 1)
        labeledPrima <- rbind(labeledPrima, cbind(cand,j))
      else
        labeledPrima <- cbind(cand, j)
    }

    # L = L u L'
    lenL <- nrow(indexInstTL)
    indexInstTL <- rbind(indexInstTL, cbind(indexInstTU[labeledPrima[,1]], labeledPrima[,2]))

    # U = U - L'
    indexInstTU <- indexInstTU[-labeledPrima[,1]]


    # Build a neighborhood graph G with L U L'

    len <- nrow(indexInstTL)
    ady <- vector("list", len) # lista de adyacencia de G

    # distancia entre xi y xj
    distance <- function(i, j) {
      getDist(indexInstTL[i,1], indexInstTL[j,1])
    }

    for (i in (lenL+1):len)
      for (j in 1:(i-1)) {
        con <- TRUE
        for (k in 1:len)
          if (k != i && k != j && distance(i,j) > max(distance(i,k),distance(k,j))) {
            con <- FALSE
            break
          }
        if (con) {
          ady[[i]] <- c(ady[[i]],j)
          ady[[j]] <- c(ady[[j]],i)
        }
      }

    # 		vecinos <- mean(sapply(ady[(lenL+1):len], length))


    # Compute the bad examples and remove them
    remove <- c() # indices de los elementos a eliminar
    for (i in (lenL+1):len) { # solo en L'

      propi <- proportion[indexInstTL[i,2]]

      # calcular observacion Oi de Ji
      Oi <- 0
      nv <- W <- k <- 0
      for (j in ady[[i]]) {
        k <- k + 1
        W[k] <- 1 / (1 + distance(i,j))
        if (indexInstTL[i,2] != indexInstTL[j,2]) {
          Oi <- Oi + W[k]
          nv <- nv + 1
        }
      }

      if (normalCriterion(theta, Oi, length(ady[[i]]), propi, W))
        remove <- c(remove, i)
    }

    # eliminar los que estan remove
    if (length(remove) > 0)
      indexInstTL <- indexInstTL[-remove,]

    iter <- iter + 1
  }

  # dejar el clasificador entrenado con las nuevas instancias
  tra.inds <- indexInstTL[,1]
  m$included.insts <- tra.inds
  base.m <- runTrain(bclassif, dists.l, tra.inds, indexInstTL[,2])
  m$base.m <- base.m
  if(! dist.is.matrix){
    # dist and included.insts are needs for compute the distances before predict
    # Save dist in the model.
    m$dist <- dist
    # Save the labeled instances in the model
    m$tr.insts <- x[m$included.insts, ]
  }
  m
}

#' @title Model Predictions
#' @description Predicts the label of instances according to SETRED model.
#' @details For additional help see \code{\link{setred}} examples.
#' @param object SETRED model object built with \code{\link{setred}} function.
#' @param x A object that can be coerced as matrix.
#' Depending on how was the model built, \code{x} is interpreted as a matrix
#' with the distances between the unseen instances and the selected training instances,
#' or a matrix of instances.
#' @param ... Additional objects that can be coerced as matrix.
#' Depending on how was the model built, this objects are used or ignored.
#' This objects are interpreted as matrices of distances between the unseen instances
#' and the selected training instances.
#' @return Vector with labels of instances.
#' @export
#' @importFrom stats predict
predict.setred <- function(object, x, ...) {
  dists.l <- getDists(object, c(list(x), list(...)))

  pred <- runPredict(object$bclassif$predClass, object$base.m, dists.l)
  r <- restoreLabels(object, pred)
  r
}

#' @title Normal criterion
#' @details Computes the critical value using the normal distribution as the authors suggest
#' when the neighborhood is big for the instances in the RNG.
#' @return A boolean value indicating if the instance must be eliminated
#' @noRd
normalCriterion <- function(theta, Oi, vec, propi, W) {
  # calcular media y desv est de J
  mean <- (1 - propi) * sum(W)
  sd <- sqrt(propi * (1 - propi) * sum(W^2))

  # calcular el p-value para Oi
  vc <- stats::qnorm(theta/2, mean, sd)

  if (vc < 0 && Oi == 0) # caso especial en que vc < 0 producto de la aproximacion mediante la dist. Normal
    FALSE
  else
    Oi >= vc
}
