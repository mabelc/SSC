
#' @title Tri-training generic method
#' @description Tri-training is a semi-supervised learning algorithm with a co-training 
#' style. This algorithm trains three classifiers with the same learning scheme from a 
#' reduced set of labeled examples. For each iteration, an unlabeled example is labeled 
#' for a classifier if the other two classifiers agree on the labeling proposed.
#' @param y A vector with the labels of training instances. In this vector the 
#' unlabeled instances are specified with the value \code{NA}.
#' @param gen.learner A function for training three supervised base classifiers.
#' This function needs two parameters, indexes and cls, where indexes indicates
#' the instances to use and cls specifies the classes of those instances.
#' @param gen.pred A function for predicting the probabilities per classes.
#' This function must be two parameters, model and indexes, where the model
#' is a classifier trained with \code{gen.learner} function and
#' indexes indicates the instances to predict.
#' @details 
#' TriTrainingG can be helpful in those cases where the method selected as 
#' base classifier needs a \code{learner} and \code{pred} functions with other
#' specifications. For more information about the general triTraining method,
#' please see the \code{\link{triTraining}} function. Essentially, the \code{triTraining}
#' function is a wrapper of the \code{triTrainingG} function.
#' @return A list object of class "triTrainingG" containing:
#' \describe{
#'   \item{model}{The final three base classifiers trained using the enlarged labeled set.}
#'   \item{model.index}{List of three vectors of indexes related to the training instances 
#'   used per each classifier. These indexes are relative to the \code{y} argument.}
#'   \item{instances.index}{The indexes of all training instances used to
#'   train the three models. These indexes include the initial labeled instances
#'   and the newly labeled instances. These indexes are relative to the \code{y} argument.}
#'   \item{model.index.map}{List of three vectors with the same information in \code{model.index}
#'   but the indexes are relative to \code{instances.index} vector.}
#' }
#' @example demo/TriTrainingG.R
#' @export
triTrainingG <- function(
  y, gen.learner, gen.pred
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
  model.index <- vector(mode = "list", length = 3)
  for(i in 1:3){
    # Train classifier
    indexes <- labeled[Sind[[i]]] # vector of indexes
    models[[i]] <- gen.learner(indexes, y[indexes])
    model.index[[i]] <- indexes
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
      cj <- getClassIdx(
        checkProb(
          prob = gen.pred(models[[j]], labeled),
          ninstances = length(labeled), 
          classes
        )
      )
      ck <- getClassIdx(
        checkProb(
          prob = gen.pred(models[[k]], labeled),
          ninstances = length(labeled), 
          classes
        )
      )
      e[i] <- measureError(cj, ck, ylabeled.map)
      
      if(e[i] < ePrima[i]){
        cj <- getClassIdx(
          checkProb(
            prob = gen.pred(models[[j]], unlabeled),
            ninstances = length(unlabeled), 
            classes
          )
        )
        ck <- getClassIdx(
          checkProb(
            prob = gen.pred(models[[k]], unlabeled),
            ninstances = length(unlabeled), 
            classes
          )
        )
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
        models[[i]] <- gen.learner(
          indexes, 
          factor(classes[c(ylabeled.map, Lcls[[i]])], classes)
        )
        model.index[[i]] <- indexes
        
        # update values for i
        ePrima[i] <- e[i]
        lPrima[i] <- length(Lind[[i]])
      }
    }
  }#end while
  
  ### Result ###
  
  # determine labeled instances
  instances.index <- unique(unlist(model.index))
  # map indexes respect to m$included.insts
  model.index.map <- lapply(
    X = model.index,
    FUN = function(indexes){
      r <- unclass(factor(indexes, levels = instances.index))
      attr(r, "levels") <- NULL
      return(r)
    }
  )
  
  # Save result
  result <- list(
    model = models,
    model.index = model.index,
    instances.index = instances.index,
    model.index.map = model.index.map
  )
  class(result) <- "triTrainingG"
  
  return(result)
}

#' @title Tri-training method
#' @description Tri-training is a semi-supervised learning algorithm with a co-training 
#' style. This algorithm trains three classifiers with the same learning scheme from a 
#' reduced set of labeled examples. For each iteration, an unlabeled example is labeled 
#' for a classifier if the other two classifiers agree on the labeling proposed.
#' @param x A object that can be coerced as matrix. This object has two possible 
#' interpretations according to the value set in the \code{x.inst} argument:
#' a matrix with the training instances where each row represents a single instance
#' or a precomputed (distance or kernel) matrix between the training examples.
#' @param y A vector with the labels of the training instances. In this vector 
#' the unlabeled instances are specified with the value \code{NA}.
#' @param x.inst A boolean value that indicates if \code{x} is or not an instance matrix.
#' Default is \code{TRUE}.
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier, using a set of instances
#' (or optionally a distance matrix) and it's corresponding classes.
#' @param learner.pars A list with additional parameters for the
#' \code{learner} function if necessary.
#' Default is \code{NULL}.
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes,
#' using the base classifiers trained with the \code{learner} function.
#' Default is \code{"predict"}.
#' @param pred.pars A list with additional parameters for the
#' \code{pred} function if necessary.
#' Default is \code{NULL}.
#' @details 
#' Tri-training initiates the self-labeling process by training three models from the 
#' original labeled set, using the \code{learner} function specified. 
#' In each iteration, the algorithm detects unlabeled examples on which two classifiers 
#' agree with the classification and includes these instances in the enlarged set of the 
#' third classifier under certain conditions. The generation of the final hypothesis is 
#' produced via the majority voting. The iteration process ends when no changes occur in 
#' any model during a complete iteration.
#'  
#' @return A list object of class "triTraining" containing:
#' \describe{
#'   \item{model}{The final three base classifiers trained using the enlarged labeled set.}
#'   \item{model.index}{List of three vectors of indexes related to the training instances 
#'   used per each classifier. These indexes are relative to the \code{y} argument.}
#'   \item{instances.index}{The indexes of all training instances used to
#'   train the three models. These indexes include the initial labeled instances
#'   and the newly labeled instances. These indexes are relative to the \code{y} argument.}
#'   \item{model.index.map}{List of three vectors with the same information in \code{model.index}
#'   but the indexes are relative to \code{instances.index} vector.}
#'   \item{classes}{The levels of \code{y} factor.}
#'   \item{pred}{The function provided in the \code{pred} argument.}
#'   \item{pred.pars}{The list provided in the \code{pred.pars} argument.}
#'   \item{x.inst}{The value provided in the \code{x.inst} argument.}
#' }
#' @references
#' ZhiHua Zhou and Ming Li.\cr
#' \emph{Tri-training: exploiting unlabeled data using three classifiers.}\cr
#' IEEE Transactions on Knowledge and Data Engineering, 17(11):1529-1541, Nov 2005. ISSN 1041-4347. doi: 10.1109/TKDE.2005. 186.
#' @example demo/TriTraining.R
#' @export
triTraining <- function(
  x, y, x.inst = TRUE,
  learner, learner.pars = NULL,
  pred = "predict", pred.pars = NULL
) {
  ### Check parameters ###
  rownames(x) <- NULL
  checkTrainingData(environment())
  learner.pars <- as.list2(learner.pars)
  pred.pars <- as.list2(pred.pars)
  
  if(x.inst){
    # Instance matrix case
    gen.learner2 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, ], cls, learner, learner.pars)
      return(m)
    }
    gen.pred2 <- function(m, testing.ints){
      prob <- predProb(m, x[testing.ints, ], pred, pred.pars)
      return(prob)
    }
    
    result <- triTrainingG(y, gen.learner2, gen.pred2)
  }else{
    # Distance matrix case
    gen.learner1 <- function(training.ints, cls){
      m <- trainModel(x[training.ints, training.ints], cls, learner, learner.pars)
      r <- list(m = m, training.ints = training.ints)
      return(r)
    }
    gen.pred1 <- function(r, testing.ints){
      prob <- predProb(r$m, x[testing.ints, r$training.ints], pred, pred.pars)
      return(prob)
    }
    
    result <- triTrainingG(y, gen.learner1, gen.pred1)
    result$model <- lapply(X = result$model, FUN = function(e) e$m)
  }
  
  ### Result ###
  result$classes = levels(y)
  result$pred = pred
  result$pred.pars = pred.pars
  result$x.inst = x.inst
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
  x <- as.matrix2(x)
  
  # Classify the instances using each classifier
  # The result is a matrix of indexes that indicates the classes
  # The matrix have one column per classifier and one row per instance
  ninstances = nrow(x)
  if(object$x.inst){
    preds <- mapply(
      FUN = function(model){
        getClassIdx(
          checkProb(
            predProb(model, x, object$pred, object$pred.pars), 
            ninstances, 
            object$classes
          )
        ) 
      },
      object$model
    )
  }else{
    preds <- mapply(
      FUN = function(model, indexes){
        getClassIdx(
          checkProb(
            predProb(model, x[, indexes], object$pred, object$pred.pars), 
            ninstances, 
            object$classes
          )
        ) 
      },
      object$model,
      object$model.index.map
    )
  }
  preds <- as.matrix2(preds)
  # Get the mode of preds for every instance (by rows)
  pred <- apply(X = preds, MARGIN = 1, FUN = getmode)
  pred <- factor(object$classes[pred], object$classes)
  
  return(pred)
}

#' @title Combining the hypothesis
#' @description This function combines the predictions obtained 
#' by the set of classifiers.
#' @param pred A list with the predictions of each classifiers
#' @return A vector of classes
#' @export
triTrainingCombine <- function(pred){
  # Check the number of instances
  ninstances <- unique(vapply(X = pred, FUN = length, FUN.VALUE = numeric(1)))
  if(length(ninstances) != 1){
    stop("The length of objects in the 'pred' parameter are not all equals.") 
  }
  as.factor(
    mapply(
      FUN = function(a, b, c){
        getmode(c(a, b, c))
      },
      pred[[1]], pred[[2]], pred[[3]]
    )
  )
}

#' @title Statistical mode
#' @noRd
getmode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
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
