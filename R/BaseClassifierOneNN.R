
#' @title 1-NN supervised classifier builder
#' @description Build a model using the given data to be capable
#' of predict the label or the probabilities of other instances,
#' according to 1-NN algorithm.
#' @param x This argument is not used, the reason why he gets is to fulfill an agreement
#' @param y a vector with the labels of training instances
#' @return A model wish the data needed to use 1-NN
#' @export
oneNN <- function(x=NULL, y) {
  if(! is.vector(y) ){
    stop("y must be a vector")
  }

  m <- list()
  class(m) <- "OneNN"
  m$classes <- unique(y)
  m$y.map <- vapply(y, FUN = function(e){ which(e == m$classes)}, FUN.VALUE = 1)
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
#' @return a matrix of probabilities, with a row per instance and a column per class.
#' @details The returned matrix has \code{nrow(dists)} rows and a column for every
#' class, where each cell represents the probability that the instance belongs to the
#' class, according to 1NN.
#' @noRd
predProb.OneNN <- function(m, dists) {
  if(!is.matrix(dists)){
    stop("The'dists' argument is not a matrix.")
  }
  if(ncol(dists) != length(m$y)){
    stop("The columns number of 'dists' are different from the number of examples used to train 'm'.")
  }

  probabilities <- matrix(nrow = nrow(dists), ncol = length(m$classes))

  for (q in 1:nrow(dists)) { #para cada instancia no etiquetada
    # buscar el mas cercano de cada clase a xi en L
    for (j in 1:length(m$y)) {
      cj <- m$y.map[j] # clase
      d <- dists[q, j] # distancia entre xq y xj
      if (is.na(probabilities[q,cj]) || d < probabilities[q,cj])
        probabilities[q,cj] <- d
    }
  }#la matrix contiene las distancias minimas a cada clase

  maxE <- max(probabilities) #obtengo el mayor valor
  probabilities <- probabilities/maxE #ahora he normalizado la matrix

  for (q in 1:nrow(dists)){ #para cada instancia no etiquetada
    probabilities[q,] <- exp(-probabilities[q,])
    sumatoria <- sum(probabilities[q,])
    probabilities[q,] <- probabilities[q,]/sumatoria #normalizo las probabilidades
  }

  probabilities
}

#' @title Model Predictions
#' @description This function predicts the class label of instances or its probability of
#' pertaining to each class based on the distance matrix.
#' @param object A model of class OneNN built with \code{\link{oneNN}}
#' @param dists A matrix of distances between the instances to classify (by rows) and
#' the instances used to train the model (by column)
#' @param type It is a string that can take two values: \code{"class"} for computing the class of
#' the instances or \code{"prob"} for computing the probabilities of belonging to each class.
#' @param ... Currently not used.
#' @return If \code{type} is equal to \code{"class"} a vector of length equal to the rows number
#' of matrix \code{dists}, containing the predicted labels. If \code{type} is equal
#' to \code{"prob"} it returns a matrix which has \code{nrow(dists)} rows and a column for every
#' class, where each cell represents the probability that the instance belongs to the class,
#' according to 1NN.
#' @export
#' @importFrom stats predict
predict.OneNN <- function(object, dists, type="class", ...){
  if(type == "class"){
    predClass.OneNN(object, dists)
  }else if(type == "prob"){
    predProb.OneNN(object, dists)
  }else{
    stop("'type' invalid.")
  }
}

#' @title 1-NN classifier specification builder
#' @description Defines a 1-NN classifier specification for
#' use in conjunction with the semi-supervised classifiers in this package.
#' @return A classifier specification built using
#' \code{\link{bClassif}} function.
#' @export
bClassifOneNN <- function(){
  bClassif(train = oneNN, predClass = predClass.OneNN, predProb = predProb.OneNN)
}

