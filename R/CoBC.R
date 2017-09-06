
#' @title Train the Co-bagging model
#' @description Trains a model for classification,
#' according to Co-bagging algorithm.
#' @param x A matrix or a dataframe with the training instances.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier
#' @param learner.pars A list with parameters that are to be passed to the \code{learner}
#' function at each Co-bagging iteration.
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes,
#' using a base classifier trained with function \code{learner}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param N The number of classifiers used as committee members. Default is 3.
#' @param perc.full A number between 0 and 1. If the percentage 
#' of new labeled examples reaches this value the Co-bagging process is stopped.
#' Default is 0.7.
#' @param u Number of unlabeled instances in the pool. Default is 100.
#' @param max.iter Maximum number of iterations to execute the self-labeling process. 
#' Default is 50.
#' @export
coBC <- function(
  x, y,
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  N = 3,
  perc.full = 0.7,
  u = 100, 
  max.iter = 50
) {
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
    H[[i]] <- trainModel(x[indexes, ], y[indexes], learner, learner.pars)
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
        prob <- H.prob(models = H[committee],
                       x = x[pool, ], 
                       pred, pred.pars, classes)
        # Select instances
        # labeledPrima[[i]] -> sel
        sel <- selectInstances(cantClass = cantClass, probabilities = prob)
        selected <- pool[sel$unlabeled.idx]
        
        ## Verify with the initial training set
        # Predict probabilities
        prob <- H.prob(models = HO, 
                       x = x[selected, ], 
                       pred, pred.pars, classes)
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
      H[[i]] <- trainModel(x[ind, ], factor(yi, classes), learner, learner.pars)
    }

    iter <- iter + 1
  }#end del while principal
  
  ### Result ###
  
  # Save result
  result <- list(
    models = H,
    classes = classes,
    pred = pred,
    pred.pars = pred.pars
  )
  class(result) <- "coBC"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.coBC <- function(object, x, ...){
  
  prob <- H.prob(object$models, x,
                 object$pred, object$pred.pars, object$classes)
  
  cls.idx <- sapply(X = 1:nrow(prob), FUN = function(i) which.max(prob[i, ]) )
  
  pred <- factor(object$classes[cls.idx], object$classes)
  
  return(pred)
}

#' TODO: Write help
#' @noRd
H.prob <- function(models, x, pred, pred.pars, classes){
  
  nunlabeled <- nrow(x)
  nclasses <- length(classes)
  
  lapply(X = 1:length(models), 
         FUN =  function(i) {
           predProb(models[[i]], x, pred, pred.pars, classes)
         }
  ) -> h.prob
  
  H.pro <- matrix(nrow = nunlabeled, ncol = nclasses)
  for(u in 1:nunlabeled){
    sapply(X = 1:nclasses, 
           FUN = function(c) {
             H.xu.wc(h.prob, u, c, nclasses) 
           }
    ) -> H.pro[u, ]
  }
  
  return(H.pro)
}

#' @title Compute the probability assigned by the committee H that xu belongs to class c
#' @param h.prob is the list containing the probability matrix  of each base classifier
#' @param u the unlabeled instance
#' @param c the class
#' @param classes The number of classes
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
