#' @title Train the Self-training model
#' @description Builds and trains a model to predict the label of instances,
#' according to Self-training algorithm.
#' @param x A object that can be coerced as matrix.
#' This object have various interpretations depending on the value set in \code{dist} argument.
#' See \code{dist} argument.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param bclassif Base classifier specification. Default is \code{bClassifOneNN()}. For defining new base classifiers
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
#' @param min.amount Minimum number of unlabeled instances to stop the training process. When the size
#' of unlabeled training instances reaches this number the self-labeling process is stopped. Default is
#' 0.3 * <number of unlabeled instances>.
#' @param max.iter Maximum number of iterations to execute in the self-labeling process. Default is 50.
#' @return The trained model stored in a list with the following named values:
#' \describe{
#'   \item{base.m}{The model trained.}
#'   \item{bclassif}{The value of \code{bclassif} argument.}
#'   \item{dist}{The value of \code{dist} argument.
#'   This value is optionally returned when the argument \code{dist} is different from "matrix".}
#'   \item{included.insts}{The indexes of the training instances used to train \code{base.m} model.
#'   Those indexes are relative to \code{x} argument.}
#'   \item{tr.insts}{The training instances corresponding to the indexes returned in \code{included.insts}.
#'   This value is optionally returned when the \code{dist} argument is different from "matrix".}
#'   \item{labels.map}{An internal map used for the labels.}
#' }
#' @references
#' David Yarowsky.\cr
#' \emph{Unsupervised word sense disambiguation rivaling supervised methods.}\cr
#' In Proceedings of the 33rd annual meeting on Association for Computational Linguistics,
#' pages 189–196. Association for Computational Linguistics, 1995.
#' @examples
#' # This example is part of SelfTraining demo.
#' # Use demo(SelfTraining) to see all the examples.
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
#' m <- selfTraining(xtrain, ytrain, dist = "Euclidean")
#' pred <- predict(m, xitest)
#' caret::confusionMatrix(table(pred, yitest))
#'
#' @export
selfTraining <- function(
  x, y,
  bclassif = bClassifOneNN(), dist = "matrix",
  min.amount = ceiling(length(which(is.na(y))) * 0.3),
  max.iter = 50
){
  # check common parameters
  dists.l <- NULL # initialized for checkCommonParams
  dist.is.matrix <- NULL # initialized for checkCommonParams
  checkCommonParams(environment())
  # check specifics parameters
  if(!(is.numeric(min.amount) && length(min.amount) == 1)) stop("min.amount must be a numeric value")
  if(!(min.amount > 0)) warning("min.amount is not a positive number")
  if(!(is.numeric(max.iter) && length(max.iter) == 1)) stop("max.iter must be a numeric value")
  if(!(max.iter > 0)) stop("max.iter must be a positive value")

  # Lista donde se almacena el modelo
  m <- list()
  class(m) <- "selfTraining"
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
  cantClass <- round(cantClass / min(cantClass)) # divido por el valor minimo
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

    for(j in 1:classes) { # para cada clase
      if (length(labeledPrima) != 1)  # si labeledPrima no es vacia
        labeledPrima <- rbind(labeledPrima, cbind(indexInstCand[[j]],j))
      else
        labeledPrima <- cbind(indexInstCand[[j]],j)
    }# end for

    # aumento L con las nuevas instancias
    indexInstTL <- rbind(indexInstTL,
                         cbind(indexInstTU[labeledPrima[,1]], labeledPrima[,2]))

    # elimino esas instancias de U
    indexInstTU <- indexInstTU[-labeledPrima[,1]]

    iter <- iter + 1
  }

  # dejar el clasificador entrenado con las nuevas instancias
  tra.inds <- indexInstTL[,1]
  m$included.insts <- tra.inds
  if(! dist.is.matrix){
    # dist and included.insts are needs for compute the distances before predict
    # Save dist in the model.
    m$dist <- dist
    # Save the labeled instances in the model
    m$tr.insts <- x[m$included.insts, ]
  }
  base.m <- runTrain(bclassif, dists.l, tra.inds, indexInstTL[,2])
  m$base.m <- base.m

  m
}

#' @title Model Predictions
#' @description Predicts the label of instances according to SelfTraining model.
#' @details For additional help see \code{\link{selfTraining}} examples.
#' @param object SelfTraining model built with \code{\link{selfTraining}} function.
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
predict.selfTraining <- function(object, x, ...) {
  dists.l <- getDists(object, c(list(x), list(...)))

  pred <- runPredict(object$bclassif$predClass, object$base.m, dists.l)
  r <- restoreLabels(object, pred)
  r
}
