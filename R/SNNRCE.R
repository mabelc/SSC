
#' @title Train a SNNRCE model
#' @description Trains a model for classification,
#' according to SNNRCE algorithm.
#' @param x A matrix or a dataframe with the training instances.
#' @param y A vector with the labels of training instances. 
#' In this vector the unlabeled instances are specified with the value \code{NA}.
#' @param dist The name of a distance method available in \code{proxy} package or
#' a function defined by the user that computes the distance between two instances.
#' Default is Euclidean distance.
#' @param alpha Rejection threshold to test the critical region. Default is 0.1.
#' @return The trained model.
#' @export
snnrce <- function(
  x, y,
  dist = "Euclidean",
  alpha = 0.1
){
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
  # This check is made by proxy::dist function in Init variables section
  # Check alpha
  if(!(alpha >= 0 && alpha <= 1)) {
    stop("Parameter alpha must be between 0 and 1")
  }
  
  ### Init variables ###
  # Compute distance matrix D
  D <- as.matrix(
    proxy::dist(x = x, method = dist, diag = TRUE, upper = TRUE, by_rows = TRUE)
  )
  
  # Init variable to store the labels
  ynew <- y
  
  # Obtain the indexes of labeled and unlabeled instances
  labeled <- which(!is.na(y))
  unlabeled <- which(is.na(y))
  # Check the labeled and unlabeled sets
  if(length(labeled) == 0){   # labeled is empty
    stop("The labeled set is empty. All the values in y parameter are NA.")
  }
  if(length(unlabeled) == 0){ # unlabeled is empty
    stop("The unlabeled set is empty. None value in y parameter is NA.")
  }
  
  # Identify the classes
  classes <- levels(y)
  nclasses <- length(classes)
  
  ### SNNRCE algorithm ###
  
  # Initial number of labeled instances
  labeledLen <- length(labeled)
  
  # STEPS 1-2
  # Count the examples per class
  cls.summary <- summary(y[labeled])
  # Ratio between count per class and the initial number of labeled instances
  proportion <- cls.summary / labeledLen
  
  # STEP 3
  # Label the instances with Rj = 0
  rem <- NULL
  for (i in 1:length(unlabeled)) {
    w <- unlabeled[i]
    clase <- -1
    # w is good when it's neighbors have all the same label
    good <- TRUE
    
    # Build RNG
    for (j in 1:labeledLen) {
      a <- labeled[j]
      edge <- TRUE
      for (b in labeled)
        if (a != b && D[w, a] > max(D[w, b], D[b, a])) {
          edge <- FALSE
          break
        }
      if (edge) {
        if (clase == -1)
          clase <- y[a]
        else if (clase != y[a]) {
          good <- FALSE
          break
        }
      }
    }
    
    if (good) {
      # label w and delete it from unlabeled
      ynew[w] <- clase
      rem <- c(rem,  i)
    }
  }

  ## Update labeled and unlabeled sets
  labeled <- c(labeled, unlabeled[rem])
  unlabeled <- unlabeled[-rem]
  
  # STEP 5 Autolabel
  initialLen <- length(labeled)
  
  max.per.class <- proportion * length(unlabeled)
  nmax <- min(max.per.class)
  count <- 0
  while (count < nmax) {
    # Predict prob using 1-NN
    model <- oneNN(y = ynew[labeled])
    prob <- predict(model, D[unlabeled, labeled], type = "prob")
    
    # Select one instance per class
    selection <- selectInstances(rep(1, nclasses), prob)
    
    # Add selected instances to labeled
    labeled.prime <- unlabeled[selection$unlabeled.idx]
    sel.classes <- classes[selection$class.idx]
    ynew[labeled.prime] <- sel.classes
    labeled <- c(labeled, labeled.prime)
    
    # Delete selected instances from unlabeled
    unlabeled <- unlabeled[-selection$unlabeled.idx]
    
    count <- count + 1
  }
  
  len <- length(labeled)
  if (initialLen < len) { # new instances were added
    # STEP 6 Build RNG for L
    
    ady <- vector("list", len) # Adjacency list of G
    for (i in 2:len)
      for (j in 1:(i-1)) {
        con <- TRUE
        for (k in 1:len)
          if (k != i && k != j && D[labeled[i], labeled[j]] > 
              max(D[labeled[i], labeled[k]], D[labeled[k], labeled[j]])) {
            con <- FALSE
            break
          }
        if (con) {
          ady[[i]] <- c(ady[[i]], j)
          ady[[j]] <- c(ady[[j]], i)
        }
      }
    
    # STEP 7 Relabel
    
    # Build Ii and Ji
    I <- rep(0, len) # = 0 len times
    J <- rep(0, len)
    for (i in 1:len)
      for (j in ady[[i]]) {
        Wij <- 1 / (1 + D[labeled[i], labeled[j]])
        I[i] <- I[i] + Wij
        if (ynew[labeled[i]] != ynew[labeled[j]])
          J[i] <- J[i] + Wij
      }
    
    # Compute mean and standard desviation of R
    R <- J / I; rm(J,I)
    media <- base::mean(R)
    ds <- stats::sd(R)
    u <- stats::qnorm(1-alpha/2)
    RCritico <- media + u * ds
    
    relabel <- which(R[(labeledLen + 1):len] > RCritico)
    for (i in relabel + labeledLen) {
      w <- -1
      if (nclasses > 2) {
        wc <- rep(0, nclasses)
        for (j in ady[[i]]) {
          Wij <- 1 / (1 + D[labeled[i], labeled[j]])
          pos <- unclass(ynew[labeled[j]])
          wc[pos] <- wc[pos] + Wij
        }
        wc[unclass(ynew[labeled[i]])] <- 0
        w <- which.max(wc)
      } else { # if two classes invert the label
        w <- ifelse(unclass(ynew[labeled[i]]) == 1, 2, 1)
      }
      
      if (w != -1)
        ynew[labeled[i]] <- classes[w]
    }
    rm(ady)
  }
  
  ### Result ###
  
  # Save result
  result <- list(
    xtrain = x[labeled,],
    ytrain = ynew[labeled],
    dist = dist
  )
  class(result) <- "snnrce"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.snnrce <- function(object, xtest, ...) {
  D <- proxy::dist(x = xtest, y = object$xtrain, method = object$dist, 
                   diag = TRUE, upper = TRUE, by_rows = TRUE)
  
  model <- oneNN(y = object$ytrain)
  cls <- predict(model, D, type = "class")
  
  return(cls)
}

