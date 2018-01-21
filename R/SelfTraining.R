
#' @title Self-training generic method
#' @description Self-training is a simple and effective semi-supervised
#' learning classification method. The self-training classifier is initially
#' trained with a reduced set of labeled examples. Then it is iteratively retrained
#' with its own most confident predictions over the unlabeled examples. 
#' Self-training follows a wrapper methodology using one base supervised 
#' classifier to establish the possible class of unlabeled instances. 
#' @param y A vector with the labels of training instances. In this vector the 
#' unlabeled instances are specified with the value \code{NA}.
#' @param gen.learner A function for training a supervised base classifier.
#' This function needs two parameters, indexes and cls, where indexes indicates
#' the instances to use and cls specifies the classes of those instances.
#' @param gen.pred A function for predicting the probabilities per classes.
#' This function must be two parameters, model and indexes, where the model
#' is a classifier trained with \code{gen.learner} function and
#' indexes indicates the instances to predict.
#' @param max.iter Maximum number of iterations to execute the self-labeling process. 
#' Default is 50.
#' @param perc.full A number between 0 and 1. If the percentage 
#' of new labeled examples reaches this value the self-training process is stopped.
#' Default is 0.7.
#' @param thr.conf A number between 0 and 1 that indicates the theshold confidence.
#' At each iteration, only the new label examples with a confidence greater than 
#' this value (\code{thr.conf}) are added to training set.
#' @details 
#' SelfTrainingG can be helpful in those cases where the method selected as 
#' base classifier needs a \code{learner} and \code{pred} functions with other
#' specifications. For more information about the general self-training method,
#' please see \code{\link{selfTraining}} function. Essentially, \code{selfTraining}
#' function is a wrapper of \code{selfTrainingG} function.
#' @return A list object of class "selfTrainingG" containing:
#' \describe{
#'   \item{model}{The final base classifier trained using the enlarged labeled set.}
#'   \item{instances.index}{The indexes of the training instances used to 
#'   train the \code{model}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   Those indexes are relative to the \code{y} argument.}
#' }
#' @example demo/SelfTrainingG.R
#' @export
selfTrainingG <- function(
  y, gen.learner, gen.pred,
  max.iter = 50,
  perc.full = 0.7,
  thr.conf = 0.5
){
  ### Check parameters ###
  # Check y 
  if(!is.factor(y) ){
    if(!is.vector(y)){
      stop("Parameter y is neither a vector nor a factor.")  
    }else{
      y = as.factor(y)
    }
  }
  # Check max.iter
  if(max.iter < 1){
    stop("Parameter max.iter is less than 1. Expected a value greater than and equal to 1.")
  }
  # Check perc.full
  if(perc.full < 0 || perc.full > 1){
    stop("Parameter perc.full is not in the range 0 to 1.")
  }
  # Check thr.conf
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
  cantClass <- round(cls.summary / min(cls.summary))
  totalPerIter <- sum(cantClass)
  # Compute count full
  count.full <- length(labeled) + round(length(unlabeled) * perc.full)
  
  iter <- 1
  while ((length(labeled) < count.full) && (length(unlabeled) >= totalPerIter) && (iter <= max.iter)) {
    
    # Train classifier
    #model <- trainModel(x[labeled, ], ynew[labeled], learner, learner.pars)
    model <- gen.learner(labeled, ynew[labeled])
    
    # Predict probabilities per classes of unlabeled examples
    #prob <- predProb(model, x[unlabeled, ], pred, pred.pars, classes)
    prob <- checkProb(prob = gen.pred(model, unlabeled), ninstances = length(unlabeled), classes)
    
    # Select the instances with better class probability
    pre.selection <- selectInstances(cantClass, prob)  
    
    # Select the instances with probability grather than the theshold confidence
    indexes <- which(pre.selection$prob.cls > thr.conf)
    if(length(indexes) == 0){ 
      iter <- iter + 1
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
  #model <- trainModel(x[labeled, ], ynew[labeled], learner, learner.pars)
  model <- gen.learner(labeled, ynew[labeled])
  
  result <- list(
    model = model,
    instances.index = labeled
  )
  class(result) <- "selfTrainingG"
  
  return(result)
}

#' @title Self-training method
#' @description Self-training is a simple and effective semi-supervised
#' learning classification method. The self-training classifier is initially
#' trained with a reduced set of labeled examples. Then it is iteratively retrained
#' with its own most confident predictions over the unlabeled examples. 
#' Self-training follows a wrapper methodology using one base supervised 
#' classifier to establish the possible class of unlabeled instances. 
#' @param x A object that can be coerced as matrix. This object has two possible 
#' interpretations according to the value set in the \code{x.dist} argument: 
#' a matrix distance between the training examples or a matrix with the 
#' training instances where each row represents a single instance.
#' @param y A vector with the labels of the training instances. In this vector 
#' the unlabeled instances are specified with the value \code{NA}.
#' @param x.dist A boolean value that indicates if \code{x} is or not a distance matrix.
#' Default is \code{FALSE}. 
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier, using a set of instances
#' (or optionally a distance matrix) and it's corresponding classes.
#' @param learner.pars A list with additional parameters for the
#' \code{learner} function if necessary.
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes,
#' using the base classifier trained with the \code{learner} function.
#' @param pred.pars A list with additional parameters for the
#' \code{pred} function if necessary.
#' @param max.iter maximum number of iterations to execute the self-labeling process. 
#' Default is 50.
#' @param perc.full A number between 0 and 1. If the percentage 
#' of new labeled examples reaches this value the self-training process is stopped.
#' Default is 0.7.
#' @param thr.conf A number between 0 and 1 that indicates the theshold confidence.
#' At each iteration, only the new labeled examples with a confidence greater than 
#' this value (\code{thr.conf}) are added to the training set.
#' @details 
#' For predicting the most accurate instances per iteration, \code{selfTraining}
#' uses the predictions obtained with the learner specified. To train a model 
#' using the \code{learner} function, it is required a set of instances 
#' (or a distance matrix between the instances if \code{x.dist} parameter is \code{TRUE})
#' in conjunction with the corresponding classes. 
#' Additionals parameters are provided to the \code{learner} function via the 
#' \code{learner.pars} argument. The model obtained is a supervised classifier
#' ready to predict new instances through the \code{pred} function. 
#' Using a similar idea, the additional parameters to the \code{pred} function
#' are provided using \code{pred.pars} argument. The \code{pred} function returns 
#' the probabilities per classes for each new instance. The value of the 
#' \code{thr.conf} argument controls the confidence of instances selected 
#' to enlarge the labeled set for the next iteration.
#' 
#' The stopping criterion is defined through the fulfillment of one of the following
#' criteria: the algorithm reaches the number of iterations defined in the \code{max.iter}
#' parameter or the portion of unlabeled set, defined in the \code{perc.full} parameter,
#' is moved to the labeled set. In some cases, the process stopps and not instances 
#' are added to the original labeled set. In this case, the user must to assign a more 
#' flexible value to the \code{thr.conf} parameter.
#' 
#' @return A list object of class "selfTraining" containing:
#' \describe{
#'   \item{model}{The final base classifier trained using the enlarged labeled set.}
#'   \item{instances.index}{The indexes of the training instances used to 
#'   train the \code{model}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   Those indexes are relative to \code{x} argument.}
#'   \item{classes}{The levels of \code{y} factor.}
#'   \item{pred}{The function provided in the \code{pred} argument.}
#'   \item{pred.pars}{The list provided in the \code{pred.pars} argument.}
#' }
#' @references
#' David Yarowsky.\cr
#' \emph{Unsupervised word sense disambiguation rivaling supervised methods.}\cr
#' In Proceedings of the 33rd annual meeting on Association for Computational Linguistics,
#' pages 189-196. Association for Computational Linguistics, 1995.
#' @example demo/SelfTraining.R
#' @export
selfTraining <- function(
  x, y, x.dist = FALSE,
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  max.iter = 50,
  perc.full = 0.7,
  thr.conf = 0.5
){
  ### Check parameters ###
  # Check x.dist
  if(!is.logical(x.dist)){
    stop("Parameter x.dist is not logical.")
  }
  
  if(x.dist){
    # Distance matrix case
    # Check matrix distance in x
    if(class(x) == "dist"){
      x <- proxy::as.matrix(x)
    }
    if(!is.matrix(x)){
      stop("Parameter x is neither a matrix or a dist object.")
    } else if(nrow(x) != ncol(x)){
      stop("The distance matrix x is not a square matrix.")
    } else if(nrow(x) != length(y)){
      stop(sprintf(paste("The dimensions of the matrix x is %i x %i", 
                         "and it's expected %i x %i according to the size of y."), 
                   nrow(x), ncol(x), length(y), length(y)))
    }
    
    gen.learner1 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, training.ints], cls, learner, learner.pars)
      r <- list(m = m, training.ints = training.ints)
      return(r)
    }
    gen.pred1 <- function(r, testing.ints){
      prob <- predProb(r$m, x[testing.ints, r$training.ints], pred, pred.pars)
      return(prob)
    }
    
    result <- selfTrainingG(y, gen.learner1, gen.pred1, max.iter, perc.full, thr.conf)
    result$model <- result$model$m
  }else{
    # Instance matrix case
    # Check x
    if(!is.matrix(x) && !is.data.frame(x)){
      stop("Parameter x is neither a matrix or a data frame.")
    }
    # Check relation between x and y
    if(nrow(x) != length(y)){
      stop("The rows number of x must be equal to the length of y.")
    }
    
    gen.learner2 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, ], cls, learner, learner.pars)
      return(m)
    }
    gen.pred2 <- function(m, testing.ints){
      prob <- predProb(m, x[testing.ints, ], pred, pred.pars)
      return(prob)
    }
    
    result <- selfTrainingG(y, gen.learner2, gen.pred2, max.iter, perc.full, thr.conf)
  }
 
  ### Result ###
  result$classes = levels(y)
  result$pred = pred
  result$pred.pars = pred.pars
  class(result) <- "selfTraining"
  
  return(result)
}

#' @title Predictions of the Self-training method
#' @description Predicts the label of instances according to the \code{selfTraining} model.
#' @details For additional help see \code{\link{selfTraining}} examples.
#' @param object Self-training model built with the \code{\link{selfTraining}} function.
#' @param x A object that can be coerced as matrix.
#' Depending on how was the model built, \code{x} is interpreted as a matrix 
#' with the distances between the unseen instances and the selected training instances, 
#' or a matrix of instances.
#' @param ... This parameter is included for compatibility reasons.
#' @return Vector with the labels assigned.
#' @export
#' @importFrom stats predict
predict.selfTraining <- function(object, x, ...) {
  if(class(x) == "dist"){
    x <- proxy::as.matrix(x)
  }
  
  result <- getClass(
    checkProb(
      predProb(object$model, x, object$pred, object$pred.pars), 
      ninstances = nrow(x), 
      object$classes
    )
  )
  
  return(result)
}
