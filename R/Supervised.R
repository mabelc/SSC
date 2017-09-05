
#' @title Train a model
#' @param x matrix of training instances
#' @param y factor of classes
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier
#' @param learner.pars a list with parameters that are to be passed to the \code{learner}
#' function.
#' @return the trained model 
#' @noRd
trainModel <- function(x, y, learner, learner.pars){
  # Train a model
  lpars <- c(list(x, y), learner.pars)
  # TODO: Call learner function using a try cast function
  model <- do.call(learner, lpars)
  
  return(model)
}

#' @title Predict classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param classes vector of possible classes
#' @return a factor with the predicted classes
#' @noRd
predClass <- function(model, x, pred, pred.pars, classes){
  # Predict probabilities
  prob <- predProb(model, x, pred, pred.pars, classes)
  # Obtain classes from probabilities
  map <- apply(prob, MARGIN = 1, FUN = which.max)
  # Convert classes indexes in a factor of classes
  r <- factor(classes[map], classes)
  
  return(r)
}

#' @title Predict classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param classes vector of possible classes
#' @return a vector with indexes that correspont to \code{classes}. 
#' This is a map of the predicted classes.
#' @noRd
predClassIdx <- function(model, x, pred, pred.pars, classes){
  # Predict probabilities
  prob <- predProb(model, x, pred, pred.pars, classes)
  # Obtain classes from probabilities
  map <- apply(prob, MARGIN = 1, FUN = which.max)
  
  return(map)
}

#' @title Predict probabilities per classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param classes vector of possible classes
#' @return a matrix of predicted probabilities with the column names equals 
#' to \code{classes}
#' @noRd
predProb <- function(model, x, pred, pred.pars, classes) {
  # Predict probabilities
  ppars <- c(list(model, x), pred.pars)
  # TODO: Call pred function using a try cast function
  prob <- do.call(pred, ppars)
  
  # Check probabilities matrix
  if(!is.matrix(prob) ||
     nrow(x) != nrow(prob) ||
     length(classes) != length(intersect(classes, colnames(prob)))){
    # TODO: Explain the error cause in the next error message
    stop("Incorrect value returned by pred function.")
  }
  r <- prob[, classes]
  
  return(r)
}
