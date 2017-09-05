
#' @title Train a SETRED model
#' @description Trains a model for classification,
#' according to SETRED algorithm.
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
#' @param dist The name of a distance method available in \code{proxy} package or
#' a function defined by the user that computes the distance between two instances.
#' @param max.iter Maximum number of iterations to execute the self-labeling process. 
#' Default is 50.
#' @param perc.full A number between 0 and 1. If the percentage 
#' of new labeled examples reaches this value the self-training process is stopped.
#' Default is 0.7.
#' @param thr.conf A number between 0 and 1 that indicates the confidence theshold.
#' At each iteration, only the new label examples with a confidence greater than 
#' this value (\code{thr.conf}) are added to training set.
#' @return The trained model.
#' @export
setred <- function(
  x, y,
  learner, learner.pars = list(),
  pred, pred.pars = list(),
  dist = "Euclidean",
  theta = 0.1,
  max.iter = 50,
  perc.full = 0.7,
  thr.conf = 0.5
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
  # Check dist
  ## This check is made by proxy::dist function in Init variables section
  # Check theta
  if(!(theta >= 0 && theta <= 1)) {
    stop("theta must be between 0 and 1")
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
  # Compute distance matrix D
  D <- as.matrix(
    proxy::dist(x = x, method = dist, diag = TRUE, upper = TRUE, by_rows = TRUE)
  )
  
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
  
  ### SETRED algorithm ###

  # Count the examples per class
  cls.summary <- summary(y[labeled])
  # Ratio between count per class and the initial number of labeled instances
  proportion <- cls.summary / length(labeled)
  # Determine the total of instances to include per iteration
  cantClass <- round(cls.summary / min(cls.summary))
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
    # TODO: Is always possible select the number of instances requested per class?
    pre.selection <- selectInstances(cantClass, prob)
    # Select the instances with probability grather than the theshold confidence
    indexes <- which(pre.selection$prob.cls > thr.conf)
    if(length(indexes) == 0){ 
      next
    }
    selection <- pre.selection[indexes,]
    
    # Save count of labeled set before it's modification
    nlabeled.old <- length(labeled)
    
    # Add selected instances to L
    labeled.prime <- unlabeled[selection$unlabeled.idx]
    sel.classes <- classes[selection$class.idx]
    ynew[labeled.prime] <- sel.classes
    labeled <- c(labeled, labeled.prime)
    
    # Delete selected instances from U
    unlabeled <- unlabeled[-selection$unlabeled.idx]
    
    # Save count of labeled set after it's modification
    nlabeled.new <- length(labeled)

    # Build a neighborhood graph G with L U L'
    ady <- vector("list", nlabeled.new) # Adjacency list of G
    
    for (i in (nlabeled.old + 1):nlabeled.new){
      for (j in 1:(i - 1)) {
        con <- TRUE
        for (k in 1:nlabeled.new)
          if (k != i && k != j && D[labeled[i], labeled[j]] > 
              max(D[labeled[i], labeled[k]], D[labeled[k], labeled[j]])) {
            con <- FALSE
            break
          }
        if (con) {
          ady[[i]] <- c(ady[[i]],j)
          ady[[j]] <- c(ady[[j]],i)
        }
      }
    }

    # Compute the bad examples and noise instances
    noise.insts <- c() # instances to delete from labeled set
    for (i in (nlabeled.old + 1):nlabeled.new) { # only on L'
      propi <- proportion[unclass(ynew[labeled[i]])]
  
      # calcular observacion Oi de Ji
      Oi <- 0
      nv <- W <- k <- 0
      for (j in ady[[i]]) {
        k <- k + 1
        W[k] <- 1 / (1 + D[labeled[i], labeled[j]])
        if (ynew[labeled[i]] != ynew[labeled[j]]) {
          Oi <- Oi + W[k]
          nv <- nv + 1
        }
      }

      if (normalCriterion(theta, Oi, length(ady[[i]]), propi, W)){
        noise.insts <- c(noise.insts, i) 
      }
    }

    # Delete from labeled the noise instances
    if (length(noise.insts) > 0){
      ynew[labeled[noise.insts]] <- NA
      labeled <- labeled[-noise.insts]
    }

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
  class(result) <- "setred"

  result
}

#' @export
#' @importFrom stats predict
predict.setred <- function(object, x, ...) {
  
  r <- predClass(object$model, x, object$pred, object$pred.pars, object$classes)
  return(r)
}


#' @title Normal criterion
#' @details Computes the critical value using the normal distribution as the authors suggest
#' when the neighborhood is big for the instances in the RNG.
#' @return A boolean value indicating if the instance must be eliminated
#' @noRd
normalCriterion <- function(theta, Oi, vec, propi, W) {
  # calcular media y desv est de J
  mean <- (1 - propi) * sum(W)
  sd <- sqrt(propi * (1 - propi) * sum(W^2))

  # calcular el p-value para Oi
  vc <- stats::qnorm(theta/2, mean, sd)

  if (vc < 0 && Oi == 0) # caso especial en que vc < 0 producto de la aproximacion mediante la dist. Normal
    FALSE
  else
    Oi >= vc
}
