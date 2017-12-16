
#' @title Self-training base method
#' @description Self-training is a simple and effective semi-supervised
#' learning classification method. The self-training classifier is initially
#' trained with a reduced set of labeled examples. Then it is iteratively retrained
#' with its own most confident predictions over the unlabeled examples. 
#' Self-training follows a wrapper methodology using one base supervised 
#' classifier to establish the possible class of unlabeled instances. 
#' @param y A vector with the labels of training instances. In this vector the 
#' unlabeled instances are specified with the value \code{NA}.
#' @param learnerB A function for training a supervised base classifier.
#' This function needs two parameters, indexes and cls, where indexes indicates
#' the instances to use and cls specifies the classes of those instances.
#' @param predB A function for predicting the probabilities per classes.
#' This function must be two parameters, model and indexes, where the model
#' is a classifier trained with \code{learnerB} function and
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
#' SelfTrainingBase can be helpful in those cases where the method selected as 
#' base classifier needs a \code{learner} and \code{pred} functions with other
#' specifications. For more information about the general self-training method,
#' please see \code{\link{selfTraining}} function. Essentially, \code{selfTraining}
#' function is a wrapper of \code{selfTrainingBase} function.
#' @return A list object of class "selfTrainingBase" containing:
#' \describe{
#'   \item{model}{The final base classifier trained using the enlarged labeled set.}
#'   \item{included.insts}{The indexes of the training instances used to 
#'   train the \code{model}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   Those indexes are relative to \code{y} argument.}
#' }
#' @examples 
#' library(ssc)
#' 
#' ## Load Wine data set
#' data(wine)
#' 
#' cls <- which(colnames(wine) == "Wine")
#' x <- wine[, -cls] # instances without classes
#' y <- wine[, cls] # the classes
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
#' ## Example: Training from a set of instances with 1-NN (knn3) as base classifier.
#' learnerB <- function(indexes, cls) 
#'   caret::knn3(x = xtrain[indexes, ], y = cls, k = 1)
#' predB <- function(model, indexes)  
#'   predict(model, xtrain[indexes, ]) 
#' 
#' md1 <- selfTrainingBase(y = ytrain, learnerB, predB)
#' md1$model
#' 
#' cls1 <- predict(md1$model, xitest, type = "class")
#' caret::confusionMatrix(table(cls1, yitest))
#' 
#' ## Example: Training from a distance matrix with 1-NN (oneNN) as base classifier.
#' dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE))
#' learnerB <- function(indexes, cls) {
#'   m <- ssc::oneNN(y = cls)
#'   attr(m, "tra.idxs") <- indexes
#'   m
#' }
#' 
#' predB <- function(model, indexes)  {
#'   tra.idxs <- attr(model, "tra.idxs")
#'   d <- dtrain[indexes, tra.idxs]
#'   prob <- predict(model, d, type = "prob",  initial.value = 0) 
#'   prob
#' }
#' 
#' md2 <- selfTrainingBase(y = ytrain, learnerB, predB)
#' ditest <- proxy::dist(x = xitest, y = xtrain[md2$included.insts,],
#'                       method = "euclidean", by_rows = TRUE)
#' cls2 <- predict(md2$model, ditest)
#' caret::confusionMatrix(table(cls2, yitest))
#' 
#' @export
selfTrainingBase <- function(
  y, learnerB, predB, 
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
    model <- learnerB(labeled, ynew[labeled])
    
    # Predict probabilities per classes of unlabeled examples
    #prob <- predProb(model, x[unlabeled, ], pred, pred.pars, classes)
    prob <- predB(model, unlabeled)
    prob <- getProb(prob, ninstances = length(unlabeled), classes)
    
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
  model <- learnerB(labeled, ynew[labeled])
  
  result <- list(
    model = model,
    included.insts = labeled
  )
  class(result) <- "selfTrainingBase"
  
  return(result)
}

#' @title Self-training method
#' @description Self-training is a simple and effective semi-supervised
#' learning classification method. The self-training classifier is initially
#' trained with a reduced set of labeled examples. Then it is iteratively retrained
#' with its own most confident predictions over the unlabeled examples. 
#' Self-training follows a wrapper methodology using one base supervised 
#' classifier to establish the possible class of unlabeled instances. 
#' @param x A matrix or a dataframe with the training instances. 
#' Each row represents a single instance.
#' @param y A vector with the labels of training instances. In this vector 
#' the unlabeled instances are specified with the value \code{NA}.
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier.
#' @param learner.pars A list with additional parameters for the
#' \code{learner} function if necessary.
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes,
#' using the base classifier trained with the \code{learner} function.
#' @param pred.pars A list with additional parameters for the
#' \code{pred} function if necessary.
#' @param x.dist A boolean value that indicates if \code{x} is a distance matrix.
#' Default is \code{FALSE}. 
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
#' criteria: the algorithm reaches the number of iterations defined in \code{max.iter}
#' parameter or the portion of unlabeled set, defined in \code{perc.full} parameter,
#' is moved to the labeled set. In some cases, the process stopps and not instances 
#' are added to the original labeled set. In this case, the user must to assign a more 
#' flexible value to the \code{thr.conf} parameter.
#' 
#' @return A list object of class "selfTraining" containing:
#' \describe{
#'   \item{model}{The final base classifier trained using the enlarged labeled set.}
#'   \item{included.insts}{The indexes of the training instances used to 
#'   train the \code{model}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   Those indexes are relative to \code{x} argument.}
#'   \item{classes}{The levels of \code{y} factor.}
#'   \item{pred}{The function provided in \code{pred} argument.}
#'   \item{pred.pars}{The list provided in \code{pred.pars} argument.}
#' }
#' @references
#' David Yarowsky.\cr
#' \emph{Unsupervised word sense disambiguation rivaling supervised methods.}\cr
#' In Proceedings of the 33rd annual meeting on Association for Computational Linguistics,
#' pages 189–196. Association for Computational Linguistics, 1995.
#' @examples 
#' 
#' ## Load Wine data set
#' data(wine)
#' 
#' cls <- which(colnames(wine) == "Wine")
#' x <- wine[, -cls] # instances without classes
#' y <- wine[, cls] # the classes
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
#' ## Example: Training from a set of instances with 1-NN as base classifier.
#' m <- selfTraining(x = xtrain, y = ytrain, 
#'                   learner = caret::knn3, 
#'                   learner.pars = list(k = 1),
#'                   pred = "predict")
#' pred <- predict(m, xitest)
#' caret::confusionMatrix(table(pred, yitest))
#' 
#' ## Example: Training from a distance matrix with 1-NN as base classifier.
#' dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE))
#' m2 <- selfTraining(x = dtrain, y = ytrain, x.dist = TRUE,
#'                   learner = ssc::oneNN, 
#'                   pred = "predict",
#'                   pred.pars = list(type = "prob", initial.value = 0))
#' ditest <- proxy::dist(x = xitest, y = xtrain[m2$included.insts,],
#'                       method = "euclidean", by_rows = TRUE)
#' pred2 <- predict(m2, ditest)
#' caret::confusionMatrix(table(pred2, yitest))
#' 
#' @export
selfTraining <- function(
  x, y, 
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  x.dist = FALSE,
  max.iter = 50,
  perc.full = 0.7,
  thr.conf = 0.5
){
  ### Check parameters ###
  # Check x
  if(!is.matrix(x) && !is.data.frame(x)){
    stop("Parameter x is neither a matrix or a data frame.")
  }
  # Check relation between x and y
  if(nrow(x) != length(y)){
    stop("The rows number of x must be equal to the length of y.")
  }
  
  if(x.dist){
    # Distance matrix case
    if(nrow(x) != ncol(x)){
      stop("The distance matrix x is not a square matrix.")
    } 
    
    learnerB1 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, training.ints], cls, learner, learner.pars)
      r <- list(m = m, training.ints = training.ints)
      return(r)
    }
    predB1 <- function(r, testing.ints){
      prob <- predProb(r$m, x[testing.ints, r$training.ints], pred, pred.pars)
      return(prob)
    }
    
    result <- selfTrainingBase(y, learnerB1, predB1, max.iter, perc.full, thr.conf)
    result$model <- result$model$m
  }else{
    # Instance matrix case
    learnerB2 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, ], cls, learner, learner.pars)
      return(m)
    }
    predB2 <- function(m, testing.ints){
      prob <- predProb(m, x[testing.ints, ], pred, pred.pars)
      return(prob)
    }
    
    result <- selfTrainingBase(y, learnerB2, predB2, max.iter, perc.full, thr.conf)
  }
 
  ### Result ###
  result$classes = levels(y)
  result$pred = pred
  result$pred.pars = pred.pars
  class(result) <- "selfTraining"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.selfTraining <- function(object, x, ...) {
  
  prob <- predProb(object$model, x, object$pred, object$pred.pars)
  result <- getClass(prob, ninstances = nrow(x), object$classes)
  
  return(result)
}
