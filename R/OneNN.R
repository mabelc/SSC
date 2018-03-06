
#' @title 1-NN supervised classifier builder
#' @description Build a model using the given data to be able
#' to predict the label or the probabilities of other instances,
#' according to 1-NN algorithm.
#' @param x This argument is not used, the reason why he gets is to fulfill an agreement
#' @param y a vector with the labels of training instances
#' @return A model with the data needed to use 1-NN
#' @export
oneNN <- function(x=NULL, y) {
  if(! is.factor(y) ){
     stop("y must be a factor")
  }

  m <- list()
  class(m) <- "OneNN"
  m$y <- y
  m
}

#' @title Predict the class of one or more instances
#' @param m a model of class OneNN built with \code{\link{oneNN}}
#' @param dists a matrix of distances between the instances to classify (by rows) and
#' the instances used to train the model (by column)
#' @return a vector of length equal to the rows number of matrix dists, containing the predicted labels
#' @noRd
predClass.OneNN <- function(m, dists) {
  if(!is.matrix(dists)){
    stop("The 'dists' argument is not a matrix.")
  }
  if(ncol(dists) != length(m$y)){
    stop("The columns number of 'dists' are different from the number of examples used to train 'm'.")
  }
  indexes <- apply(X = dists, MARGIN = 1, FUN = which.min)
  m$y[indexes]
}

#' @title Predict the probabilities of one or more instances
#' @param m a model of class OneNN built with \code{\link{oneNN}}
#' @param dists a matrix of distances between the instances to classify (by rows) and
#' the instances used to train the model (by column)
#' @param distance.weighting the way in that the distances are used to compute the probabilities.
#' Valid options are:
#' \itemize{
#'    \item \code{"none"}: distances are not used to compute probabilities.
#'    \item \code{"reciprocal"}: the reciprocal for the distance 
#'    (\emph{1/distance}) with the 1-nearest-neighbour is used 
#'    to compute probabilities.
#'    \item \code{"reciprocalexp"}: the reciprocal for the exponential 
#'    function of the distance ((\emph{1/exp(distance)})) with
#'    the 1-nearest-neighbour is used to compute probabilities.
#' }
#' @param initial.value the probabilities for each class are initiaized with this value. 
#' Default is 0.
#' @return a matrix of probabilities, with a row per instance and a column per class.
#' @details The returned matrix has \code{nrow(dists)} rows and a column for every
#' class, where each cell represents the probability that the instance belongs to the
#' class, according to 1NN.
#' @noRd
predProb.OneNN <- function(m, dists, distance.weighting = "reciprocalexp", initial.value = 0) {
  if(!is.matrix(dists)){
    stop("The'dists' argument is not a matrix.")
  }
  if(ncol(dists) != length(m$y)){
    stop("The columns number of 'dists' are different from the number of examples used to train 'm'.")
  }
  if(!(is.numeric(initial.value)) || (initial.value < 0) || (initial.value > 1)){
    stop("Parameter initial.value must be a numeric value in the range 0 to 1.")
  }
  
  probabilities <- matrix(data = initial.value, nrow = nrow(dists), ncol = length(levels(m$y)))
  minDistances <- matrix(nrow = nrow(dists), ncol = length(levels(m$y)))

  for (q in 1:nrow(dists)) { # for each query instance
    # find the most close instance to xi  for each class in dists
    for (j in 1:length(m$y)) {
      cj <- unclass(m$y[j]) # clase
      d <- dists[q, j] # distance between xq and xj
      if (is.na(minDistances[q,cj]) || d < minDistances[q,cj])
        minDistances[q,cj] <- d
    }
  }# matrix contains the minimun distances for each class

  if (distance.weighting == "none"){ #adds 1 vote in the pos of 1NN
    probabilities <- probabilities + t(apply(X = minDistances, MARGIN = 1,FUN = function(f){
      r <- vector(mode = "numeric",length = length(f))
      indice <- which.min(f)
      r[indice] <- 1
      r
    }))
  }
  else if (distance.weighting == "reciprocal"){#adds 1/distance vote in the pos of 1NN
    probabilities <- probabilities + t(apply(X = minDistances, MARGIN = 1,FUN = function(f){
      r <- vector(mode = "numeric",length = length(f))
      indice <- which.min(f)
      r[indice] <- 1/(f[indice]+0.000001) #to avoid div by zero
      r
    }))
  }
  else if (distance.weighting == "reciprocalexp"){
    maxE <- max(minDistances) # get global max value
    probabilities <- probabilities + minDistances/maxE #normalizing the matrix
    probabilities <- exp(-probabilities) # computing 1/exp(dist)
  }
  else stop("The'distance.weighting' argument has not a valid value.")

  for (q in 1:nrow(dists)){ # for each query instance
    total <- sum(probabilities[q,])
    probabilities[q,] <- probabilities[q,]/total #normalize the probabilities
  }

  colnames(probabilities) <- levels(m$y)
  probabilities
}

#' @title Model Predictions
#' @description This function predicts the class label of instances or its probability of
#' pertaining to each class based on the distance matrix.
#' @param object A model of class OneNN built with \code{\link{oneNN}}
#' @param dists A matrix of distances between the instances to classify (by rows) and
#' the instances used to train the model (by column)
#' @param type A string that can take two values: \code{"class"} for computing the class of
#' the instances or \code{"prob"} for computing the probabilities of belonging to each class.
#' @param ... Currently not used.
#' @return If \code{type} is equal to \code{"class"} a vector of length equal to the rows number
#' of matrix \code{dists}, containing the predicted labels. If \code{type} is equal
#' to \code{"prob"} it returns a matrix which has \code{nrow(dists)} rows and a column for every
#' class, where each cell represents the probability that the instance belongs to the class,
#' according to 1NN.
#' @export
#' @importFrom stats predict
predict.OneNN <- function(object, dists, type="prob", ...){
  if(type == "class"){
    predClass.OneNN(object, dists)
  }else if(type == "prob"){
    predProb.OneNN(object, dists, ...)
  }else{
    stop("'type' invalid.")
  }
}


