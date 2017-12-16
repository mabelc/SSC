
snnrceBase <- function(
  D, y,
  alpha = 0.1
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
  # Check distance matrix
  if(class(D) == "dist"){
    D <- proxy::as.matrix(D)
  }
  if(!is.matrix(D)){
    stop("Parameter D is neither a matrix or a dist object.")
  }else if(nrow(D) != ncol(D)){
    stop("The matrix D is not square.")
  }else if(nrow(D) != length(y)){
    stop(sprintf(paste("The dimensions of the matrix D is %i x %i", 
                       "and it's expected %i x %i according to the size of y."), 
                 nrow(D), ncol(D), length(y), length(y)))
  }
  # Check alpha
  if(!(alpha >= 0 && alpha <= 1)) {
    stop("Parameter alpha must be between 0 and 1")
  }
  
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
    model = oneNN(y = ynew[labeled]),
    included.insts = labeled
  )
  
  class(result) <- "snnrceBase"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.snnrceBase <- function(object, D, ...) {
  if(class(D) == "dist"){
    D <- proxy::as.matrix(D)
  }
  
  cls <- predict(object$model, D, type = "class")
  return(cls)
}

#' @export
snnrce <- function(
  x, y,
  dist = "Euclidean",
  x.dist = FALSE,
  alpha = 0.1
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
    
    result <- snnrceBase(D = x, y, alpha)
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
    
    result <- snnrceBase(
      D = proxy::dist(x, method = dist, by_rows = TRUE,
                      diag = TRUE, upper = TRUE), 
      y, 
      alpha
    )
    
    class(result) <- "snnrce"
    result$xtrain <- x[result$included.insts, ]
    result$dist <- dist
  }
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.snnrce <- function(object, x, ...) {
  D <- proxy::dist(x, y = object$xtrain, method = object$dist, 
                   diag = TRUE, upper = TRUE, by_rows = TRUE)
  
  cls <- predict(object$model, D, type = "class")
  return(cls)
}

