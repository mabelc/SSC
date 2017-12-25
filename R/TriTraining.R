#' @title Tri-training base method
#' @description Tri-training is a semi-supervised learning algorithm with a co-training 
#' style. This algorithm trains three classifiers with the same learning scheme from a 
#' reduced set of labeled examples. For each iteration, an unlabeled example is labeled 
#' for a classifier if the other two classifiers agree on the labeling proposed.
#' @param y A vector with the labels of training instances. In this vector the 
#' unlabeled instances are specified with the value \code{NA}.
#' @param learnerB A function for training three supervised base classifiers.
#' This function needs two parameters, indexes and cls, where indexes indicates
#' the instances to use and cls specifies the classes of those instances.
#' @param predB A function for predicting the probabilities per classes.
#' This function must be two parameters, model and indexes, where the model
#' is a classifier trained with \code{learnerB} function and
#' indexes indicates the instances to predict.
#' @details 
#' TriTrainingBase can be helpful in those cases where the method selected as 
#' base classifier needs a \code{learner} and \code{pred} functions with other
#' specifications. For more information about the general triTraining method,
#' please see \code{\link{triTraining}} function. Essentially, \code{triTraining}
#' function is a wrapper of \code{triTrainingBase} function.
#' @return A list object of class "triTrainingBase" containing:
#' \describe{
#'   \item{model}{The final base classifier trained using the enlarged labeled set.}
#'   \item{included.insts}{The indexes of the training instances used to 
#'   train the \code{model}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   Those indexes are relative to \code{y} argument.}
#' }
#' @examples
#' @export
triTrainingBase <- function(
  y, learnerB, predB
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
  
  ### Init variables ###  
  # Identify the classes
  classes <- levels(y)
  nclasses <- length(classes)
  
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
  
  ylabeled <- y[labeled]
  ylabeled.map <- unclass(ylabeled)
  
  ### Tri-training algorithm ###
  
  # Init base classifiers
  Sind <- resample(y[labeled], N = 3)
  
  models <- vector(mode = "list", length = 3)
  final.indexes <- vector(mode = "list", length = 3)
  for(i in 1:3){
    # Train classifier
    indexes <- labeled[Sind[[i]]] # vector of indexes
    # trainModel(x[indexes, ], y[indexes], learner, learner.pars)
    models[[i]] <- learnerB(indexes, y[indexes])
    final.indexes[[i]] <- indexes
  }
  
  ePrima <- rep(x = 0.5, times = 3)
  lPrima <- rep(x = 0, times = 3)
  
  updateClassifier <- rep(x = TRUE, times = 3)
  Lind <- vector(mode = "list", length = 3)
  Lcls <- vector(mode = "list", length = 3)
  
  
  iter <- 0
  while (any(updateClassifier)){ # At least one classifier was modified
    
    iter <- iter + 1
    updateClassifier[1:3] <- FALSE
    e <- c()
    
    for (i in 1:3){ # train every classifier
      # init L for i
      Lind[[i]] <- numeric()
      Lcls[[i]] <- numeric()
      
      # get the two values in 1:3 different to i
      j <- i %% 3 + 1
      k <- (i+1) %% 3 + 1
      
      # measure error
      prob <- predB(models[[j]], labeled)
      cj <- getClassIdx(prob, ninstances = length(labeled), classes)
      
      prob <- predB(models[[k]], labeled)
      ck <- getClassIdx(prob, ninstances = length(labeled), classes)
      e[i] <- measureError(cj, ck, ylabeled.map)
      
      if(e[i] < ePrima[i]){
        
        prob <- predB(models[[j]], unlabeled)  
        cj <- getClassIdx(prob, ninstances = length(unlabeled), classes)
        
        prob <- predB(models[[j]], unlabeled)  
        ck <- getClassIdx(prob, ninstances = length(unlabeled), classes)
        agree <- (which(cj == ck))
        
        Lind[[i]] <- unlabeled[agree]
        Lcls[[i]] <- cj[agree]
        
        if(lPrima[i] == 0){ # is the first time
          lPrima[i] <- floor(e[i] / (ePrima[i] - e[i]) + 1)
        }
        
        len <- length(agree)
        if (lPrima[i] < len){
          if (e[i] * len < ePrima[i] * lPrima[i]){
            updateClassifier[i] <- TRUE
          } else if (lPrima[i] > e[i] / (ePrima[i] - e[i])){
            indexes <- sample(
              x = 1:len, 
              size = ceiling(ePrima[i] * lPrima[i] / e[i] - 1)
            )
            Lind[[i]] <- Lind[[i]][indexes]  
            Lcls[[i]] <- Lcls[[i]][indexes]  
            
            updateClassifier[i] <- TRUE
          }
        }
      }#end if e < e'
    }#end for every classifier
    
    for(i in 1:3){
      if (updateClassifier[i]){
        # Train classifier
        indexes <- c(labeled, Lind[[i]])
        models[[i]] <- learnerB(
          indexes, 
          factor(classes[c(ylabeled.map, Lcls[[i]])], classes)
        )
        final.indexes[[i]] <- indexes
        
        # update values for i
        ePrima[i] <- e[i]
        lPrima[i] <- length(Lind[[i]])
      }
    }
  }#end while
  
  ### Result ###
  
  # determine labeled instances
  included.insts <- union(final.indexes[[1]], 
                          union(final.indexes[[2]], 
                                final.indexes[[3]]))
  # map indexes respect to m$included.insts
  indexes <- vector(mode = "list", length = 3)
  for(i in 1:3){
    indexes[[i]] <- vapply(final.indexes[[i]], FUN.VALUE = 1,
                           FUN = function(e){ which(e == included.insts)})
  }
  
  # Save result
  result <- list(
    models = models,
    indexes = indexes,
    included.insts = included.insts 
  )
  class(result) <- "triTrainingBase"
  
  return(result)
}

#' @title Tri-training method
#' @description Tri-training is a semi-supervised learning algorithm with a co-training 
#' style. This algorithm trains three classifiers with the same learning scheme from a 
#' reduced set of labeled examples. For each iteration, an unlabeled example is labeled 
#' for a classifier if the other two classifiers agree on the labeling proposed.
#' @param x A object that can be coerced as matrix. This object has two possible 
#' interpretations according to the value set in \code{x.dist} argument: 
#' a matrix distance between the training examples or a matrix with the 
#' training instances where each row represents a single instance.
#' @param y A vector with the labels of the training instances. In this vector 
#' the unlabeled instances are specified with the value \code{NA}.
#' @param learner either a function or a string naming the function for 
#' training the supervised base classifiers.
#' @param learner.pars A list with additional parameters for the
#' \code{learner} function if necessary.
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes,
#' using the base classifiers trained with the \code{learner} function.
#' @param pred.pars A list with additional parameters for the
#' \code{pred} function if necessary.
#' @param x.dist A boolean value that indicates if \code{x} is or not a distance matrix.
#' Default is \code{FALSE}. 
#' @details 
#' SETRED initiates the self-labeling process by training a model from the original 
#' labeled set. In each iteration, the \code{learner} function detects unlabeled 
#' examples on wich it makes most confident prediction and labels those examples 
#' according to the \code{pred} function. The identification of mislabeled examples is 
#' performed using a neighborhood graph created from distance matrix \code{D}. 
#' Most examples possess the same label in a neighborhood. So if an example locates 
#' in a neighborhood with too many neighbors from different classes, this example should 
#' be considered problematic. The value of the \code{theta} argument controls the confidence 
#' of the candidates selected to enlarge the labeled set. The lower this value is, the more 
#' restrictive it is the selection of the examples that are considered good.
#' For more information about the self-labeled process and the remainders parameters, please 
#' see \code{\link{selfTraining}}.
#'  
#' @return A list object of class "triTraining" containing:
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
#' ZhiHua Zhou and Ming Li.\cr
#' \emph{Tri-training: exploiting unlabeled data using three classifiers.}\cr
#' IEEE Transactions on Knowledge and Data Engineering, 17(11):1529–1541, Nov 2005. ISSN 1041-4347. doi: 10.1109/TKDE.2005. 186.
#' @examples
#' @export 
triTraining <- function(
  x, y,
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  x.dist = FALSE
) {
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
    
    learnerB1 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, training.ints], cls, learner, learner.pars)
      r <- list(m = m, training.ints = training.ints)
      return(r)
    }
    predB1 <- function(r, testing.ints){
      prob <- predProb(r$m, x[testing.ints, r$training.ints], pred, pred.pars)
      return(prob)
    }
    
    result <- triTrainingBase(y, learnerB1, predB1)
    for(i in 1:3){
      result$models[[i]] <- result$models[[i]]$m 
    }
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
    
    learnerB2 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, ], cls, learner, learner.pars)
      return(m)
    }
    predB2 <- function(m, testing.ints){
      prob <- predProb(m, x[testing.ints, ], pred, pred.pars)
      return(prob)
    }
    
    result <- triTrainingBase(y, learnerB2, predB2)
  }
  
  ### Result ###
  result$classes = levels(y)
  result$pred = pred
  result$pred.pars = pred.pars
  result$x.dist = x.dist
  class(result) <- "triTraining"
  
  return(result)
}

#' @title Predictions of the Tri-training method
#' @description Predicts the label of instances according to the \code{triTraining} model.
#' @details For additional help see \code{\link{triTraining}} examples.
#' @param object Tri-training model built with the \code{\link{triTraining}} function.
#' @param x A object that can be coerced as matrix.
#' Depending on how was the model built, \code{x} is interpreted as a matrix 
#' with the distances between the unseen instances and the selected training instances, 
#' or a matrix of instances.
#' @param ... This parameter is included for compatibility reasons.
#' @return Vector with the labels assigned.
#' @export
#' @importFrom stats predict
predict.triTraining <- function(object, x, ...) {
  if(class(x) == "dist"){
    x <- proxy::as.matrix(x)
  }
  
  preds <- matrix(nrow = 3, ncol = nrow(x))
  ninstances = nrow(x)
  for(i in 1:3){
    preds[i,] <- getClassIdx(
      prob <- predProb(
        object$models[[i]], 
        if(object$x.dist) x[, object$indexes[[i]]] else x,
        object$pred, 
        object$pred.pars
      ), 
     ninstances, object$classes)
  }
  # get the mode of the predictions for every instance
  pred <- apply(X = preds, MARGIN = 2, FUN = statisticalMode)
  pred <- factor(object$classes[pred], object$classes)
  
  return(pred)
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