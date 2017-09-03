
#' @title Train a Tri-training model
#' @description Trains a model for classification,
#' according to Tri-training algorithm.
#' @param x A matrix or a dataframe with the training instances.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier
#' @param learner.pars A list with parameters that are to be passed to the \code{learner}
#' function.
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes,
#' using a base classifier trained with the \code{learner} function.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @return The trained model.
#' @export 
triTraining <- function(
  x, y,
  learner, learner.pars = list(),
  pred, pred.pars = list()
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
  
  classify <- function(model, indexes){
    ppars <- c(list(model, x[indexes, ]), pred.pars)
    prob <- do.call(pred, ppars)
    
    # Check probabilities matrix
    if(!is.matrix(prob) ||
       nrow(prob) != length(indexes) ||
       length(intersect(classes, colnames(prob))) != nclasses){
      # TODO: Explain the error cause in the next error message
      stop("Incorrect value returned by pred function.")
    }
    prob <- prob[, classes]
    
    map <- apply(prob, MARGIN = 1, FUN = which.max)
    return(map)
  }
  
  # Init base classifiers
  Sind <- resample(y[labeled], N = 3)
  
  models <- vector(mode = "list", length = 3)
  for(i in 1:3){
    # Train classifier
    indexes <- labeled[Sind[[i]]] # vector of indexes
    lpars <- c(list(x[indexes, ], y[indexes]), learner.pars)
    # TODO: Call learner function using a try cast function
    models[[i]] <- do.call(learner, lpars)
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
      cj <- classify(models[[j]], labeled)
      ck <- classify(models[[k]], labeled)
      e[i] <- measureError(cj, ck, ylabeled.map)
      
      if(e[i] < ePrima[i]){
        
        cj <- classify(models[[j]], unlabeled)
        ck <- classify(models[[k]], unlabeled)
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
            sample(
              x = 1:len, 
              size = ceiling(ePrima[i] * lPrima[i] / e[i] - 1)
            ) -> indexes
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
        ind <- c(labeled, Lind[[i]]) # vector of indexes
        yi <- factor(classes[c(ylabeled.map, Lcls[[i]])], classes)
        lpars <- c(list(x[ind, ], yi), learner.pars)
        # TODO: Call learner function using a try cast function
        models[[i]] <- do.call(learner, lpars)
        
        # update values for i
        ePrima[i] <- e[i]
        lPrima[i] <- length(Lind[[i]])
      }
    }
  }#end while
  
  
  ### Result ###
  
  # Save result
  result <- list(
    models = models,
    classes = classes,
    pred = pred,
    pred.pars = pred.pars
  )
  class(result) <- "triTraining"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.triTraining <- function(object, x, ...) {
  
  preds <- matrix(nrow = 3, ncol = nrow(x))
  for(i in 1:3){
    ppars <- c(list(object$models[[i]], x), object$pred.pars)
    prob <- do.call(object$pred, ppars)
    
    indexes <- apply(prob[, object$classes], MARGIN = 1, FUN = which.max)
    preds[i,] <- indexes
  }
  # get the mode of the predictions for every instance
  pred <- apply(X = preds, MARGIN = 2, FUN = statisticalMode)
  pred <- factor(object$classes[pred], levels = object$classes)
  
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