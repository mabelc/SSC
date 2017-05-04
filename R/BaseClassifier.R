#' @title Base Classifier Specification
#' @description This function defines a supervised classifier specification
#' with the format expected by the semi-supervised classifiers in this package.
#' @param train a function to train a supervised classifier.
#' This function should return the model trained.
#' For usage information see Details.
#' @param predClass a function to predict the class of
#' instances, using the model trained with \code{train}.
#' This function should return a vector with the predicted classes.
#' For usage information see Details.
#' @param predProb a function to predict the probabilities per class
#' of instances, using the model trained with \code{train}.
#' This function should return a matrix with the predicted
#' probabilities. The matrix, should have:
#' the number of rows equal to the size of prediction set and
#' the number of columns equal to the number of possible classes.
#' For usage information see Details.
#' @param ... additional arguments passed to \code{train}.
#' @return The classifier specification stored in a list of class \code{bClassif}.
#' @details
#'
#' The \code{train} function is used as follows:
#'
#' \code{model  <-  train(m,  y,  ...)}
#'
#' The training data is provided in the two first arguments. Here, \code{m}
#' is a distance matrix between the training instances and \code{y} is a vector
#' with the classes of those instances. Additional arguments of the classifier can
#' be passed to \code{train} through the additional arguments (\code{...})
#' specified in \code{bClassif}.
#'
#' The \code{predClass} and \code{predProb} functions are used as follows:
#'
#' \code{cls  <-  predClass(model, pm)} \cr
#' \code{cls  <-  predProb(model, pm)}
#'
#' In both cases, the prediction data is supplied using \code{pm}.
#' \code{pm} is a distance matrix with the following dimensions:
#' the number of rows equal to the size of the prediction set and the number
#' of columns equal to the size of the training set. \code{predClass} returns
#' a vector with the predicted classes and \code{predProb} returns a matrix
#' that contains the class probabilities. In this matrix, there is a column
#' for each class and a row for each instance.
#' @export
bClassif <- function(
  train,
  predClass,
  predProb,
  ...
){
  train.args <- list(...)
  if(length(unique(names(train.args))) != length(train.args)){
    stop("Found arguments in '...' with the same name.")
  }
  r <- list(train=train, predClass=predClass, predProb=predProb, train.args = train.args)
  class(r) <- "bClassif"
  return(r)
}

#' @title Function caller
#' @description \code{runTrain} call the train function
#' @param base.arg Base algorithm as that created using \code{\link{bClassif}} function.
#' @param dists.l List of distance matrices.
#' @param inds Vector of indexes that indicates the rows and columns of matrices on list 'dists.l'
#' to use.
#' @param y Vector with the labels corresponding to 'inds' instances.
#' @param ... Args to run base algorithm train function
#' @noRd
runTrain <- function(bclassif, dists.l, inds, y) {
  dists.l <- lapply(dists.l, function(m) m[inds, inds])
  do.call(bclassif$train, c(dists.l, list(y), bclassif$train.args))
}

#' @title Function caller
#' @description \code{runPredict} call a predict function
#' @param predFun the predict function to run
#' @param model The model to provide to \code{predFun}
#' @param dists.l List of distance matrices.
#' @param inds.row Indexes vector that indicates the rows of distances matrices to use
#' @param inds.col Indexes vector that indicates the columns of distances matrices to use
#' @noRd
runPredict <- function(predFun, model, dists.l, inds.row = TRUE, inds.col = TRUE) {
  dists.l <- lapply(dists.l, function(m) m[inds.row, inds.col])
  do.call(predFun, c(list(model), dists.l))
}
