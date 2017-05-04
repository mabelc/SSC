#' @title Train the SNNRCE model
#' @description Builds and trains a model to predict the label of instances,
#' according to SNNRCE algorithm.
#' @param x A object that can be coerced as matrix.
#' This object have various interpretations depending on the value set in \code{dist} argument.
#' See \code{dist} argument.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
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
#' @param alpha Rejection threshold to test the critical region. Default is 0.1.
#' @return The trained model stored in a list with the following named values:
#' \describe{
#'   \item{base.m}{The model trained.}
#'   \item{bclassif}{The classifier specification returned by \code{bClassifOneNN} function.}
#'   \item{dist}{The value of \code{dist} argument.
#'   This value is optionally returned when the argument \code{dist} is different from "matrix".}
#'   \item{included.insts}{The indexes of the training instances used to train \code{base.m} model.
#'   Those indexes are relative to \code{x} argument.}
#'   \item{tr.insts}{The training instances corresponding to the indexes returned in \code{included.insts}.
#'    This value is optionally returned when the \code{dist} argument is different from "matrix".}
#'   \item{labels.map}{An internal map used for the labels.}
#' }
#' @references
#' Yu Wang, Xiaoyan Xu, Haifeng Zhao, and Zhongsheng Hua.\cr
#' \emph{Semisupervised learning based on nearest neighbor rule and cut edges.}\cr
#' Knowledge-Based Systems, 23(6):547–554, 2010. ISSN 0950-7051. doi: http://dx.doi.org/10.1016/j.knosys.2010.03.012.
#' @examples
#' # This example is part of SNNRCE demo.
#' # Use demo(SNNRCE) to see all the examples.
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
#' m <- snnrce(xtrain, ytrain, dist = "Euclidean")
#' pred <- predict(m, xitest)
#' caret::confusionMatrix(table(pred, yitest))
#'
#' @export
snnrce <- function(
  x, y, dist = "matrix",
  alpha = 0.1
) {
  # check specifics parameters
  if(!(alpha >= 0 && alpha <= 1)) stop("alpha must be between 0 and 1")
  if( length(dist) != 1 ) stop("Expected only one value on 'dist'.")
  # check common parameters
  dists.l <- NULL # initialized for checkCommonParams
  dist.is.matrix <- NULL # initialized for checkCommonParams
  checkCommonParams(environment())
  if(dist.is.matrix){ rm(x) } # dists.l contain x

  # distancia entre instancias i y j
  distInst <- function(i, j) { dists.l[[1]][i,j] }

  # Lista donde se almacena el modelo
  m <- list()
  class(m) <- "snnrce"
  bclassif <- bClassifOneNN()
  m$bclassif <- bclassif

  #calcular los ratio de cada clase para determinar las cantidades a seleccionar en cada iteracion
  yy <- y[!is.na(y)]
  m <- normalizeLabels(m, yy)
  yy <- mapLabels(m, yy)
  indexInstTL <- cbind(which(!is.na(y)), yy)
  indexInstTU <- which(is.na(y))

  # STEPS 1-2
  classes <- length(m$labels.map)
  labeledLen <- length(yy)

  # guardar las cant por cada clase
  cantClass <- sapply(1:classes, function(c) length(which(y == c)))

  # calcular proporcion de cada clase
  proportion <- cantClass / labeledLen


  # STEP 3

  # etiquetar los que tienen Rj = 0
  rem <- NULL # elementos de U etiquetados inicialmente
  for (i in 1:length(indexInstTU)) {
    w <- indexInstTU[i]
    clase <- -1
    good <- TRUE # todos los vecinos tienen la misma etiqueta

    # construir RNG
    for (j in 1:labeledLen) {
      a <- indexInstTL[j,1]
      edge <- TRUE
      for (b in indexInstTL[1:labeledLen,1])
        if (a != b && distInst(w,a) > max(distInst(w,b),distInst(b,a))) {
          edge <- FALSE
          break
        }
      if (edge) {
        if (clase == -1)
          clase <- indexInstTL[j,2]
        else if (clase != indexInstTL[j,2]) {
          good <- FALSE
          break
        }
      }
    }

    if (good) {
      # etiquetar y eliminar de unlabeled
      indexInstTL <- rbind(indexInstTL, c(w,clase))
      rem <- c(rem,i)
    }
  }

  indexInstTU <- indexInstTU[-rem]
  nmax <- proportion * length(indexInstTU)
  inicialLen <- nrow(indexInstTL)

  # STEP 5 autolabel

  iter <- 0
  cantClass <- rep(0,classes)
  while (all(cantClass < nmax)) {
    iter <- iter + 1

    tra.inds <- indexInstTL[,1]
    base.m <- runTrain(bclassif, dists.l, tra.inds, indexInstTL[,2])
    cl <- runPredict(bclassif$predProb, base.m, dists.l, indexInstTU, tra.inds)

    indexCand <- selectInst(rep(1,classes), cl)
    rem <- c()
    for (c in 1:classes) {
      rem <- c(rem, indexCand[[c]])
      indexInstTL <- rbind(indexInstTL, cbind(indexInstTU[indexCand[[c]]], c))
      cantClass <- cantClass + length(indexCand[[c]])
    }
    indexInstTU <- indexInstTU[-rem]
  }

  len <- nrow(indexInstTL)
  if (inicialLen < len) { #adicione nuevas instancias que voy a editar
    # STEP 6 Construir RNG para L

    # distancia entre Xi y Xj
    distIdx <- function(i, j) {
      dists.l[[1]][indexInstTL[i,1], indexInstTL[j,1]]
    }

    ady <- vector("list", len) # lista de adyacencia de G
    for (i in 2:len)
      for (j in 1:(i-1)) {
        con <- TRUE
        for (k in 1:len)
          if (k != i && k != j && distIdx(i,j) > max(distIdx(i,k),distIdx(k,j))) {
            con <- FALSE
            break
          }
        if (con) {
          ady[[i]] <- c(ady[[i]],j)
          ady[[j]] <- c(ady[[j]],i)
        }
      }

    # STEP 7 Relabel

    # calcular Ii y Ji
    I <- rep(0,len) # = 0 len veces
    J <- rep(0,len)
    for (i in 1:len)
      for (j in ady[[i]]) {
        Wij <- 1 / (1 + distIdx(i,j))
        I[i] <- I[i] + Wij
        if (indexInstTL[i,2] != indexInstTL[j,2])
          J[i] <- J[i] + Wij
      }

    # calcular media y ds de R
    R <- J / I; rm(J,I)
    media <- mean(R)
    ds <- stats::sd(R)
    u <- stats::qnorm(1-alpha/2)
    RCritico <- media + u * ds

    relabel <- which(R[(labeledLen+1):len] > RCritico)
    for (i in relabel + labeledLen) {
      w <- -1
      if (classes > 2) {
        wc <- rep(0, classes)
        for (j in ady[[i]]) {
          Wij <- 1 / (1 + distIdx(i,j))
          wc[indexInstTL[j,2]] <- wc[indexInstTL[j,2]] + Wij
          #sc[indexInstTL[j,2]] <- sc[indexInstTL[j,2]] + 1
        }
        wc[indexInstTL[i,2]] <- 0
        w <- which.max(wc)
      }
      else # si son 2 clases es solo invertir la etiqueta
        w <- ifelse(indexInstTL[i,2] == 1, 2, 1)

      if (w != -1)
        indexInstTL[i,2] <- w
    }
    rm(ady) # eliminar objeto para liberar memoria
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
#' @description Predicts the label of instances according to SNNRCE model.
#' @details For additional help see \code{\link{snnrce}} examples.
#' @param object SNNRCE model object built with \code{\link{snnrce}} function.
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
predict.snnrce <- function(object, x, ...) {
  dists.l <- getDists(object, c(list(x), list(...)))

  pred <- runPredict(object$bclassif$predClass, object$base.m, dists.l)
  r <- restoreLabels(object, pred)
  r
}
