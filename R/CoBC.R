
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
        h.prob <- lapply(X = 1:length(models), 
               FUN =  function(i) {
                 prob <- predB(models[[i]], pool)
                 prob <- getProb(prob, ninstances, classes)
               }
        )
        prob <- H.prob(h.prob, ninstances, nclasses)
        # Select instances
        # labeledPrima[[i]] -> sel
        sel <- selectInstances(cantClass = cantClass, probabilities = prob)
        selected <- pool[sel$unlabeled.idx]
        
        ## Verify with the initial training set
        # Predict probabilities
        ninstances = length(selected)
        h.prob <- lapply(X = 1:N, 
                         FUN =  function(i) {
                           prob <- predB(HO[[i]], selected)
                           prob <- getProb(prob, ninstances, classes)
                         }
        )
        prob <- H.prob(h.prob, ninstances, nclasses)
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
    included.insts = included.insts 
  )
  class(result) <- "coBCBase"
  
  return(result)
}

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
  result$classes = levels(y)
  result$pred = pred
  result$pred.pars = pred.pars
  result$x.dist = x.dist
  class(result) <- "coBC"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.coBC <- function(object, x, ...){
  
  h.prob <- list()
  ninstances = nrow(x)
  for(i in 1:length(object$models)){
    if(object$x.dist){
      prob <- predProb(object$models[[i]], x[, object$indexes[[i]]], object$pred, object$pred.pars)  
    } else{
      prob <- predProb(object$models[[i]], x, object$pred, object$pred.pars)  
    }
  
    h.prob[[i]] <- getProb(prob, ninstances, object$classes)
  }
  
  prob <- H.prob(h.prob, ninstances, nclasses = length(object$classes))
  
  cls.idx <- sapply(X = 1:nrow(prob), FUN = function(i) which.max(prob[i, ]) )
  
  pred <- factor(object$classes[cls.idx], object$classes)
  
  return(pred)
}

#' TODO: Write help
#' @noRd
H.prob <- function(h.prob, ninstances, nclasses){

  H.pro <- matrix(nrow = ninstances, ncol = nclasses)
  for(u in 1:ninstances){
    H.pro[u, ] <- sapply(X = 1:nclasses, 
           FUN = function(c) {
             H.xu.wc(h.prob, u, c, nclasses) 
           }
    )
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
