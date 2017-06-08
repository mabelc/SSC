
#' @title Train a Self-training model
#' @description Trains a model for classification,
#' according to Self-training algorithm.
#' @param x A matrix or a dataframe with the training instances.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier
#' @param learner.pars A list with parameters that are to be passed to the \code{learner}
#' function at each self-training iteration.
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
#' @param thr.conf A number between 0 and 1 that indicates the confidence theshold.
#' At each iteration, only the new label examples with a confidence greater than 
#' this value (\code{thr.conf}) are added to training set.
#' @export
selfTraining <- function(
  x, y,
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  max.iter = 50,
  perc.full = 0.7,
  thr.conf = 0.5,
  train.model = TRUE
){
  # Check x
  if(!is.matrix(x) && !is.data.frame(x)){
    stop("Parameter x is neither a matrix or a data frame.")
  }
  # Check y 
  if(!is.factor(y)){
    stop("Parameter y is not a factor. Use as.factor(y) to convert y to a factor.")
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
  
  # Determine the indexes of labeled instances
  labeled <- which(!is.na(y))
  y.labeled.count <- length(labeled)
  # Determine the indexes of unlabeled examples
  unlabeled <- which(is.na(y))
  y.unlabeled.count <- length(unlabeled)
  ## Check the labeled and unlabeled sets
  if(length(labeled) == 0){   # labeled is empty
    stop("The labeled set is empty. All the values in y parameter are NA.")
  }
  if(length(unlabeled) == 0){ # unlabeled is empty
    stop("The unlabeled set is empty. None value in y parameter is NA.")
  }
  # Identify the classes
  classes <- levels(y)
  nclasses <- length(classes)
  
  # Count the examples per class
  cls.summary <- summary(y[labeled])
  # Determine the total of instances to include per iteration 
  cantClass <- round(cls.summary / min(cls.summary)) # divido por el valor minimo
  totalPerIter <- sum(cantClass)
  
  # Compute count full
  count.full <- length(labeled) + floor(length(unlabeled) * perc.full)
  # round(3.5) = 4, round(4.5) = 4
  # floor(3.5) = 3, round(4.5) = 4
  
  # Init variables
  y.new <- y
  y.new.prob <- matrix(nrow = 0, ncol = 1 + nclasses)
  colnames(y.new.prob) <- c("y.index", classes)
  
  iter <- 1
  while ((length(labeled) < count.full) && (length(unlabeled) >= totalPerIter) && (iter <= max.iter)) {
    
    # Train classifier
    lpars <- c(list(x[labeled, ], y.new[labeled]), learner.pars)
    model <- do.call(learner, lpars)
    
    # Predict probabilities per classes of unlabeled examples
    ppars <- c(list(model, x[unlabeled, ]), pred.pars)
    prob <- do.call(pred, ppars)
    
    # Select the instances with better class probability 
    pre.selection <- selectInstances(cantClass, prob)
    
    # 
    indexes <- which(pre.selection$prob.cls > thr.conf)
    if(length(indexes) == 0){
      next
    }
    selection <- pre.selection[indexes,]
    
    # Add selected instances to L
    labeled.prime <- unlabeled[selection$unlabeled.idx]
    y.new[labeled.prime] <- classes[selection$class.idx]
    labeled <- c(labeled, labeled.prime)
    
    # Delete selected instances from U
    unlabeled <- unlabeled[-selection$unlabeled.idx]
    
    # Save probabilities of selected instances
    y.new.prob <- rbind(
      y.new.prob, cbind(labeled.prime, prob[selection$unlabeled.idx,])
    )
    
    iter <- iter + 1
  }  
  
  # Build model
  if(train.model){
    lpars <- c(list(x[labeled, ], y.new[labeled]), learner.pars)
    model <- do.call(learner, lpars) 
  }else{
    model <- NULL
  }
  
  # Save result
  info <- list(
    y.labeled.count = y.labeled.count,
    y.unlabeled.count = y.unlabeled.count,
    perc.full = perc.full,
    count.full = count.full,
    max.iter = max.iter,
    iter.count = iter - 1,
    thr.conf = thr.conf,
    train.model = train.model
  )
  blearner <- list(
    learner = learner,
    learner.pars = learner.pars,
    pred = pred,
    pred.pars = pred.pars
  )
  result <- list(
    model = model,
    y = data.frame(y = y, y.new = y.new),
    y.new.prob = as.data.frame(y.new.prob),
    info = info,
    blearner = blearner
  )
  class(result) <- "selfTraining"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.selfTraining <- function(object, x, ...) {
  if(is.null(object$model)){
    stop("The model not exists.")
  }
  predict(object$model, x, ...)
}
