
#' @title Train a Self-training model
#' @description Trains a model for classification,
#' according to Self-training algorithm.
#' @param x A matrix or a dataframe with the training instances.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier
#' @param learner.pars A list with parameters that are to be passed to the \code{learner}
#' function.
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes,
#' using a base classifier trained with function \code{learner}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param max.iter Maximum number of iterations to execute the self-labeling process. 
#' Default is 50.
#' @param perc.full A number between 0 and 1. If the percentage 
#' of new labeled examples reaches this value the self-training process is stopped.
#' Default is 0.7.
#' @param thr.conf A number between 0 and 1 that indicates the theshold confidence.
#' At each iteration, only the new label examples with a confidence greater than 
#' this value (\code{thr.conf}) are added to training set.
#' @return The trained model.
#' @export
selfTraining <- function(
  x, y,
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  max.iter = 50,
  perc.full = 0.7,
  thr.conf = 0.5
){
  ### Check parameters ###
  # Check x
  if(!is.matrix(x) && !is.data.frame(x)){
    stop("Parameter x is neither a matrix or a data frame.")
  }
  # Check y 
  if(!is.factor(y)){
    stop("Parameter y is not a factor. Use as.factor(y) to convert y to a factor.")
  }
  # Check relation between x and y
  if(nrow(x) != length(y)){
    stop("The rows number of x must be equal to the length of y.")
  }
  # Check max.iter
  if(max.iter < 1){
    stop("Parameter max.iter is less than 1. Expected a value greater than and equal to 1.")
  }
  # Check perc.full
  if(perc.full < 0 || perc.full > 1){
    stop("Parameter perc.full is not in the range 0 to 1.")
  }
  # Check perc.full
  if(thr.conf < 0 || thr.conf > 1){
    stop("Parameter thr.conf is not in the range 0 to 1.")
  }
  
  ### Init variables ###
  # Identify the classes
  classes <- levels(y)
  nclasses <- length(classes)
  
  # Init variable to store the labels
  ynew <- y

  # Obtain the indexes of labeled and unlabeled instances
  labeled <- which(!is.na(y))
  unlabeled <- which(is.na(y))
  ## Check the labeled and unlabeled sets
  if(length(labeled) == 0){   # labeled is empty
    stop("The labeled set is empty. All the values in y parameter are NA.")
  }
  if(length(unlabeled) == 0){ # unlabeled is empty
    stop("The unlabeled set is empty. None value in y parameter is NA.")
  }
  
  ### Self Training algorithm ###
  
  # Count the examples per class
  cls.summary <- summary(y[labeled])
  # Determine the total of instances to include per iteration 
  cantClass <- round(cls.summary / min(cls.summary)) # divido por el valor minimo
  totalPerIter <- sum(cantClass)
  # Compute count full
  count.full <- length(labeled) + round(length(unlabeled) * perc.full)
  
  iter <- 1
  while ((length(labeled) < count.full) && (length(unlabeled) >= totalPerIter) && (iter <= max.iter)) {
    
    # Train classifier
    model <- trainModel(x[labeled, ], ynew[labeled], learner, learner.pars)
    
    # Predict probabilities per classes of unlabeled examples
    prob <- predProb(model, x[unlabeled, ], pred, pred.pars, classes)
    
    # Select the instances with better class probability
    pre.selection <- selectInstances(cantClass, prob[, classes])  
    
    # Select the instances with probability grather than the theshold confidence
    indexes <- which(pre.selection$prob.cls > thr.conf)
    if(length(indexes) == 0){ 
      next
    }
    selection <- pre.selection[indexes,]
    
    # Add selected instances to L
    labeled.prime <- unlabeled[selection$unlabeled.idx]
    sel.classes <- classes[selection$class.idx]
    ynew[labeled.prime] <- sel.classes
    labeled <- c(labeled, labeled.prime)
    
    # Delete selected instances from U
    unlabeled <- unlabeled[-selection$unlabeled.idx]
    
    iter <- iter + 1
  }  
  
  ### Result ###
  
  # Train final model
  model <- trainModel(x[labeled, ], ynew[labeled], learner, learner.pars)
  
  # Save result
  result <- list(
    model = model,
    classes = classes,
    pred = pred,
    pred.pars = pred.pars
  )
  class(result) <- "selfTraining"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.selfTraining <- function(object, x, ...) {
  
  r <- predClass(object$model, x, object$pred, object$pred.pars, object$classes)
  return(r)
}
