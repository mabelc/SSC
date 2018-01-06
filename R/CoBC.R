
#' @title CoBC base method
#' @description CoBC is a semi-supervised learning algorithm with a co-training 
#' style. This algorithm trains \code{N} classifiers with the learning scheme defined in 
#' \code{learnerB} using a reduced set of labeled examples. For each iteration, an unlabeled 
#' example is labeled for a classifier if the most confident classifications assigned by the 
#' other \code{N-1} classifiers agree on the labeling proposed. The unlabeled examples 
#' candidates are selected randomly from a pool of size \code{u}.
#' @param y A vector with the labels of training instances. In this vector the 
#' unlabeled instances are specified with the value \code{NA}.
#' @param learnerB A function for training \code{N} supervised base classifiers.
#' This function needs two parameters, indexes and cls, where indexes indicates
#' the instances to use and cls specifies the classes of those instances.
#' @param predB A function for predicting the probabilities per classes.
#' This function must be two parameters, model and indexes, where the model
#' is a classifier trained with \code{learnerB} function and
#' indexes indicates the instances to predict.
#' @param N The number of classifiers used as committee members. All these classifiers 
#' are trained using the \code{learnerB} function. Default is 3.
#' @param perc.full A number between 0 and 1. If the percentage 
#' of new labeled examples reaches this value the self-labeling process is stopped.
#' Default is 0.7.
#' @param u Number of unlabeled instances in the pool. Default is 100.
#' @param max.iter Maximum number of iterations to execute in the self-labeling process. 
#' Default is 50.
#' @details 
#' coBCBase can be helpful in those cases where the method selected as 
#' base classifier needs a \code{learner} and \code{pred} functions with other
#' specifications. For more information about the general coBC method,
#' please see \code{\link{coBC}} function. Essentially, \code{coBC}
#' function is a wrapper of \code{coBCBase} function.
#' @return A list object of class "coBCBase" containing:
#' \describe{
#'   \item{models}{The final three base classifiers trained using the enlarged labeled set.}
#'   \item{included.insts}{The indexes of the total of training instances used to 
#'   train the three \code{models}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   These indexes are relative to the \code{y} argument.}
#'   \item{indexes}{List of three vectors of indexes related to the training instances 
#'   used per each classifier. These indexes are relative to \code{included.insts}.}
#'   \item{classes}{The levels of \code{y} factor.}
#' }
#' @examples
#' @export
coBCBase <- function(
  y,
  learnerB,
  predB,
  N = 3,
  perc.full = 0.7,
  u = 100, 
  max.iter = 50
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
  # Check N
  if(N <= 0){
    stop("Parameter N must be a positive number.") 
  }
  # Check max.iter
  if(max.iter < 1){
    stop("Parameter max.iter is less than 1. Expected a value greater than and equal to 1.")
  }
  # Check perc.full
  if(perc.full < 0 || perc.full > 1){
    stop("Parameter perc.full is not in the range 0 to 1.")
  }
  
  ### Init variables ###
  # Identify the classes
  classes <- levels(y)
  nclasses <- length(classes)
  
  # Check u
  if(u <= nclasses){
    stop("Parameter u must be greather than the number of classes.")
  }
  
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
  
  ### Co-bagging algorithm ###
  
  # Count the examples per class
  cls.summary <- summary(y[labeled])
  # Determine the total of instances to include per iteration 
  cantClass <- round(cls.summary / min(cls.summary)) # divido por el valor minimo
  totalPerIter <- sum(cantClass)
  
  # Lists to store the training instances indexes and it's classes
  Lind <- vector(mode = "list", length = N)
  Lcls <- vector(mode = "list", length = N)
  # List to store the models
  H <- vector(mode = "list", length = N)
  
  # Train models
  s = resample(y[labeled], N = N)
  y.map <- unclass(y)
  for (i in 1:N) {
    # Train model
    indexes <- labeled[s[[i]]]
    H[[i]] <- learnerB(indexes, y[indexes])
    # Save instances info
    Lind[[i]] <- indexes
    Lcls[[i]] <- y.map[indexes]
  }
  # Save original hypothesis
  HO <- H
  
  iter <- 1
  min.amount <- round(length(unlabeled) * (1 - perc.full))
  while ((length(unlabeled) > min.amount) && (iter <= max.iter) ){
    
    end <- N
    for(i in 1:N){ # For each classifier
      if(length(unlabeled) > totalPerIter){# Can I satisfy the i classifier
        # Select randomly a pool of unlabeled instances
        pool <- sample(x = unlabeled, size = min(u, length(unlabeled)), replace = FALSE)
        
        ## Select the more competent instances
        # Obtain the committee for the classifier i
        committee <- setdiff(1:length(H), i)
        # Predict probabilities for unlabeled prime instances
        models <- H[committee]
        ninstances = length(pool)
        prob <- coBCCombine(
          h.prob = lapply(
            X = 1:length(models), 
            FUN =  function(i)
              checkProb(prob = predB(models[[i]], pool), ninstances, classes)
          ),
          ninstances,
          classes
        )
        # Select instances
        sel <- selectInstances(cantClass = cantClass, probabilities = prob)
        selected <- pool[sel$unlabeled.idx]
        
        ## Verify with the initial training set
        # Predict probabilities
        ninstances = length(selected)
        prob <- coBCCombine(
          h.prob = lapply(
            X = 1:N, 
            FUN =  function(i) 
              checkProb(prob = predB(HO[[i]], selected), ninstances, classes)
          ), 
          ninstances, 
          classes
        )
        # Compute classes
        cls.idx <- sapply(X = 1:nrow(prob), FUN = function(i) which.max(prob[i, ]) )
        # Compare 
        indCoinciden <- which(cls.idx == sel$class.idx)
        
        # Add indCoinciden set to the training examples set of classifier i
        Lind[[i]] <- c(Lind[[i]], selected[indCoinciden])
        Lcls[[i]] <- c(Lcls[[i]], cls.idx[indCoinciden]) 
        
        # Remove the unlabeled instances selected
        unlabeled <- setdiff(unlabeled, selected)
      } else {
        end <- i - 1
        break
      }
    }# End for each classifier
    
    # Train models with the new instances
    for (i in 1:end){
      # Train classifier
      ind <- Lind[[i]] # indexes of intances
      yi <- classes[Lcls[[i]]] # indexes of classes
      H[[i]] <- learnerB(ind, factor(yi, classes))
    }
    
    iter <- iter + 1
  }# End of main while
  
  ### Result ###
  
  # determine labeled instances
  included.insts <- c()
  for(i in 1:N){
    included.insts <- union(Lind[[i]], included.insts)
  }
  # map indexes respect to m$included.insts
  indexes <- vector(mode = "list", length = N)
  for(i in 1:N){
    indexes[[i]] <- vapply(Lind[[i]], FUN.VALUE = 1,
                           FUN = function(e){ which(e == included.insts)})
  }
  
  # Save result
  result <- list(
    models = H,
    indexes = indexes,
    classes = classes,
    included.insts = included.insts 
  )
  class(result) <- "coBCBase"
  
  return(result)
}

#' @title CoBC method
#' @description Co-Training by Committee (CoBC) is a semi-supervised learning algorithm 
#' with a co-training style. This algorithm trains \code{N} classifiers with the learning 
#' scheme defined in \code{learner} argument using a reduced set of labeled examples. For 
#' each iteration, an unlabeled 
#' example is labeled for a classifier if the most confident classifications assigned by the 
#' other \code{N-1} classifiers agree on the labeling proposed. The unlabeled examples 
#' candidates are selected randomly from a pool of size \code{u}.
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
#' @param N The number of classifiers used as committee members. All these classifiers 
#' are trained using the \code{learnerB} function. Default is 3.
#' @param perc.full A number between 0 and 1. If the percentage 
#' of new labeled examples reaches this value the self-labeling process is stopped.
#' Default is 0.7.
#' @param u Number of unlabeled instances in the pool. Default is 100.
#' @param max.iter Maximum number of iterations to execute in the self-labeling process. 
#' Default is 50.
#' @details
#' This method trains an ensemble of diverse classifiers. To promote the initial diversity 
#' the classifiers are trained from the reduced set of labeled examples by Bagging.
#' The stopping criterion is defined through the fulfillment of one of the following
#' criteria: the algorithm reaches the number of iterations defined in \code{max.iter}
#' parameter or the portion of unlabeled set, defined in \code{perc.full} parameter,
#' is moved to the enlarged labeled set of the classifiers.
#' @return A list object of class "coBC" containing:
#' \describe{
#'   \item{models}{The final three base classifiers trained using the enlarged labeled set.}
#'   \item{included.insts}{The indexes of the total of training instances used to 
#'   train the three \code{models}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   These indexes are relative to the \code{y} argument.}
#'   \item{indexes}{List of three vectors of indexes related to the training instances 
#'   used per each classifier. These indexes are relative to \code{included.insts}.}
#'   \item{classes}{The levels of \code{y} factor.}
#'   \item{pred}{The function provided in the \code{pred} argument.}
#'   \item{pred.pars}{The list provided in the \code{pred.pars} argument.}
#'   \item{x.dist}{The value provided in the \code{x.dist} argument.}
#' }
#' @examples
#' @export
coBC <- function(
  x, y,
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  x.dist = FALSE,
  N = 3,
  perc.full = 0.7,
  u = 100, 
  max.iter = 50
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
    
    result <- coBCBase(y, learnerB1, predB1, N, perc.full, u, max.iter)
    for(i in 1:N){
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
    
    result <- coBCBase(y, learnerB2, predB2, N, perc.full, u, max.iter)
  }
  
  ### Result ###
  result$pred = pred
  result$pred.pars = pred.pars
  result$x.dist = x.dist
  class(result) <- "coBC"
  
  return(result)
}

#' @title Predictions of the coBC method
#' @description Predicts the label of instances according to the \code{coBC} model.
#' @details For additional help see \code{\link{coBC}} examples.
#' @param object coBC model built with the \code{\link{coBC}} function.
#' @param x A object that can be coerced as matrix.
#' Depending on how was the model built, \code{x} is interpreted as a matrix 
#' with the distances between the unseen instances and the selected training instances, 
#' or a matrix of instances.
#' @param ... This parameter is included for compatibility reasons.
#' @return Vector with the labels assigned.
#' @export
#' @importFrom stats predict
predict.coBC <- function(object, x, ...){
  ninstances = nrow(x)
  # Predict probabilities per instances using each model
  if(object$x.dist){
    h.prob <- mapply(
      FUN = function(model, indexes){
        checkProb(
          predProb(model, x[, indexes], object$pred, object$pred.pars), 
          ninstances, 
          object$classes
        )
      },
      object$models,
      object$indexes,
      SIMPLIFY = FALSE
    )
  }else{
    h.prob <- mapply(
      FUN = function(model){
        checkProb(
          predProb(model, x, object$pred, object$pred.pars), 
          ninstances, 
          object$classes
        )
      },
      object$models,
      SIMPLIFY = FALSE
    )
  }
  
  pred <- getClass(
    # Combine probability matrices
    coBCCombine(h.prob, ninstances, object$classes)
  )
  
  return(pred)
}

#' @title Combining the hypothesis
#' @description This function combines the probabilities predicted by the committee of 
#' classifiers.
#' @param h.prob A list of probability matrices.
#' @param ninstances The number of rows of each matrix in \code{h.prob}.
#' @param classes The classes in the same order that appear 
#' in the columns of each matrix in \code{h.prob}.
#' @return A probability matrix
#' @export
coBCCombine <- function(h.prob, ninstances, classes){
  
  nclasses <- length(classes)
  
  H.pro <- matrix(nrow = ninstances, ncol = nclasses)
  for(u in 1:ninstances){
    H.pro[u, ] <- vapply(
      X = 1:nclasses, 
      FUN = function(c) {
        H.xu.wc(h.prob, u, c, nclasses) 
      },
      FUN.VALUE = 0
    )
  }
  
  colnames(H.pro) <- classes
  
  return(H.pro)
}

#' @title Compute the probability assigned by the committee H 
#' that xu belongs to class c
#' @param h.prob a list containing the probability matrix 
#' of each base classifier
#' @param u The unlabeled instance
#' @param c The class
#' @param nclasses The number of classes
#' @return The probability
#' @noRd
H.xu.wc <- function(h.prob, u, c, nclasses){
  N <- length(h.prob)
  num <- sum(sapply(X = 1:N, FUN = function(i) h.prob[[i]][u, c] ))
  den <- 0
  for(j in 1:nclasses){
    den <- den + sum(sapply(X = 1:N, FUN = function(i) h.prob[[i]][u, j] ))
  }
  return(num / den)
}
