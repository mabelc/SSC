#' @title Train the Tri-training model
#' @description Builds and trains a model to predict the label of instances,
#' according to Tri-training algorithm.
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
#' @return The trained model stored in a list with the following named values:
#' \describe{
#'   \item{models}{A list with the models (classifiers) trained.}
#'   \item{bclassif}{The value of \code{bclassif} argument.}
#'   \item{dist}{The value of \code{dist} argument.
#'   This value is optionally returned when the argument \code{dist} is different from "matrix".}
#'   \item{included.insts}{The indexes of the training instances used.
#'   This set is the union of the indexes used to train each model in \code{models}.
#'   These indexes are relative to \code{x} argument.}
#'   \item{indexes}{ A list of vectors. Each vector indicates the indexes of the instances
#'   used to train the corresponding model in \code{models}. For example, the instances used to train
#'   \code{models[1]} are \code{x[included.insts[indexes[1]], ]}.}
#'   \item{tr.insts}{The training instances corresponding to the indexes returned in \code{included.insts}.
#'    This value is optionally returned when the \code{dist} argument is different from "matrix".}
#'   \item{labels.map}{An internal map used for the labels.}
#' }
#' @references
#' ZhiHua Zhou and Ming Li.\cr
#' \emph{Tri-training: exploiting unlabeled data using three classifiers.}\cr
#' IEEE Transactions on Knowledge and Data Engineering, 17(11):1529–1541, Nov 2005. ISSN 1041-4347. doi: 10.1109/TKDE.2005. 186.
#' @examples
#' # This example is part of TriTraining demo.
#' # Use demo(TriTraining) to see all the examples.
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
#' m <- triTraining(xtrain, ytrain, dist = "Euclidean")
#' pred <- predict(m, xitest)
#' caret::confusionMatrix(table(pred, yitest))
#'
#' @export
triTraining <- function(
  x, y,
  bclassif = bClassifOneNN(), dist = "matrix"
) {
  # Check common parameters and initialize common variables
  dists.l <- NULL # initialized for checkCommonParams
  dist.is.matrix <- NULL # initialized for checkCommonParams
  checkCommonParams(environment())

  # Lista donde se almacena el modelo
  m <- list()
  class(m) <- "triTraining"
  m$bclassif <- bclassif
  models <- list()

  #calcular los ratio de cada clase para determinar las cantidades a seleccionar en cada iteracion
  yy <- y[!is.na(y)]
  m <- normalizeLabels(m, yy)
  indexInstTL <- cbind(which(!is.na(y)), mapLabels(m, yy))
  indexInstTU <- which(is.na(y))

  #inicializar base classifiers
  S <- resample(indexInstTL = indexInstTL, N = 3)

  for (i in 1:3) {
    models[[i]] <- runTrain(bclassif, dists.l, inds = S[[i]][,1], y = S[[i]][,2])
  }

  ePrima <- rep(x=0.5, times=3)
  lPrima <- rep(x=0, times=3)

  updateClassifier <- rep(x=TRUE, times=3)
  L <- vector(mode="list",length=3)
  iter <- 0

  classify <- function(model, test.inds, train.inds){
    runPredict(bclassif$predClass, model, dists.l, test.inds, train.inds)
  }

  while (any(updateClassifier)){ # at least one classifier was modified

    iter <- iter + 1
    updateClassifier[1:3] <- FALSE
    e <- c()

    for (i in 1:3){ # train every classifier
      # init L for i
      L[[i]] <- matrix(nrow = 0, ncol = 2)

      # get the two values in 1:3 different to i
      j <- i %% 3 + 1
      k <- (i+1) %% 3 + 1

      # measure error
      cj <- classify(models[[j]], test.inds = indexInstTL[,1], train.inds = S[[j]][,1])
      ck <- classify(models[[k]], test.inds = indexInstTL[,1], train.inds = S[[k]][,1])
      e[i] <- measureError(cj, ck, indexInstTL[,2])

      if(e[i] < ePrima[i]){

        cj <- classify(models[[j]], test.inds = indexInstTU, train.inds = S[[j]][,1])
        ck <- classify(models[[k]], test.inds = indexInstTU, train.inds = S[[k]][,1])
        agree <- (which(cj == ck))

        for (u in agree){ # for every unlabeled where the classifiers agree
           L[[i]] <- rbind(L[[i]], c(indexInstTU[u], cj[u]))
        }

        if(lPrima[i] == 0){ # is the first time
          lPrima[i] <- floor(e[i]/(ePrima[i]-e[i]) + 1)
        }

        if (lPrima[i] < nrow(L[[i]])){
          if (e[i]*nrow(L[[i]]) < ePrima[i]*lPrima[i]){
            updateClassifier[i] <- TRUE
          }
          else if (lPrima[i] > e[i]/(ePrima[i]-e[i])){
            L[[i]] <- subsample(L[[i]], ceiling(ePrima[i]*lPrima[i]/e[i] - 1))
            updateClassifier[i] <- TRUE
          }
        }
      }#end if e < e'
    }#end for every classifier

    for(i in 1:3){
      if (updateClassifier[i]){
        # train the model again
        S[[i]] <- rbind(indexInstTL, L[[i]])
        models[[i]] <- runTrain(bclassif, dists.l, inds = S[[i]][,1], y = S[[i]][,2])
        # update values for i
        ePrima[i] <- e[i]
        lPrima[i] <- nrow(L[[i]])
      }
    }
  }#end while

  # determine labeled instances
  m$included.insts <- union(S[[3]][,1], union(S[[1]][,1], S[[2]][,1]))
  m$indexes <- list()
  # map indexes respect to m$included.insts
  for(i in 1:3){
    m$indexes[[i]] <- vapply(S[[i]][,1], FUN = function(e){ which(e == m$included.insts)}, FUN.VALUE = 1)
  }
  m$models <- models
  if(! dist.is.matrix){
    # dist and included.insts are needs for compute the distances before predict
    # Save dist in the model.
    m$dist <- dist
    # Save the labeled instances in the model
    m$tr.insts <- x[m$included.insts, ]
  }
  m
}

#' @title Measure the error of two base classifiers
#' @param cj predicted classes using classifier j
#' @param ck predicted classes using classifier k
#' @param y expected classes
#' @return The error of the two classifiers.
#' @noRd
measureError <- function(cj, ck, y){
  agree <- (which(cj == ck))
  agreeCorrect <- which (cj[agree] == y[agree])
  error <- (length(agree) - length(agreeCorrect))/length(agree)

  if (is.nan(error)){#si no coinciden en ningun caso el error es maximo
    error <- 1
  }
  error
}

#' @noRd
subsample <- function(L, N){
  s <- sample(x = 1:nrow(L), size = N)
  L[s,]
}

#' @title Model Predictions
#' @description Predicts the label of instances according to TriTraining model.
#' @details For additional help see \code{\link{triTraining}} examples.
#' @param object TriTraining model object built with \code{\link{triTraining}} function.
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
predict.triTraining <- function(object, x, ...) {
  dists.l <- getDists(object, c(list(x), list(...)))

  preds <- matrix(nrow = 3, ncol = nrow(x))
  for(i in 1:3){
    preds[i,] <- runPredict(
      object$bclassif$predClass, # predict function
      object$models[[i]], # model i
      dists.l, # matrixes of distances
      1:nrow(x), # test indexes
      object$indexes[[i]] # train indexes
    )
  }
  # get the mode of the predictions for every instance
  pred <- c()
  for(i in 1:ncol(preds)){
    pred[i] <- statisticalMode(preds[,i])
  }

  r <- restoreLabels(object, pred)
  r
}
