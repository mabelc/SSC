

#' @export
snnrce <- function(
  x, y,
  dist,
  alpha = 0.1
){
  # Check y 
  if(!is.factor(y)){
    stop("Parameter y is not a factor. Use as.factor(y) to convert y to a factor.")
  }
  # Determine the indexes of labeled instances
  labeled <- which(!is.na(y))
  # Determine the indexes of unlabeled examples
  unlabeled <- which(is.na(y))
  ## Check the labeled and unlabeled sets
  if(length(labeled) == 0){   # labeled is empty
    stop("The labeled set is empty. All the values in y parameter are NA.")
  }
  if(length(unlabeled) == 0){ # unlabeled is empty
    stop("The unlabeled set is empty. None value in y parameter is NA.")
  }
  
  # Check x
  if(!is.matrix(x) && !is.data.frame(x)){
    stop("Parameter x is neither a matrix or a data frame.")
  }
  # Check relation between x and y
  if(nrow(x) != length(y)){
    stop("The rows number of x must be equal to the length of y.")
  }
  
  # Check alpha
  if(!(alpha >= 0 && alpha <= 1)) {
    stop("alpha must be between 0 and 1")
  }
  
  # Compute distance matrix D
  D <- as.matrix(
    proxy::dist(x = x, method = dist, diag = TRUE, upper = TRUE, by_rows = TRUE)
  )
  
  # Identify the classes
  classes <- levels(y)
  nclasses <- length(classes)
  
  # Count the examples per class
  cls.summary <- summary(y[labeled])
  # Determine the total of instances to include per iteration
  cantClass <- round(cls.summary / min(cls.summary))
  
  # Init variable to increment the labeled set
  y.new <- y
  
  # Initial number of labeled instances
  labeledLen <- length(labeled)
  # Ratio between count per class and the initial number of labeled instances
  proportion <- cantClass / labeledLen
  
  ### STEP 3 ###
  
  # Label the instances with Rj = 0
  rem <- NULL
  for (i in 1:length(unlabeled)) {
    w <- unlabeled[i]
    clase <- -1
    good <- TRUE # todos los vecinos tienen la misma etiqueta
    
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
          clase <- y[labeled[j]]
        else if (clase != y[labeled[j]]) {
          good <- FALSE
          break
        }
      }
    }
    
    if (good) {
      # etiquetar y eliminar de unlabeled
      y.new[w] <- clase
      rem <- c(rem,i)
    }
  }
  
  labeled <- c(labeled, unlabeled[rem])
  unlabeled <- unlabeled[-rem]
  
  nmax <- proportion * length(unlabeled)
  inicialLen <- length(labeled)
  
  # STEP 5 autolabel
  
  iter <- 0
  cantClass <- rep(0, nclasses)
  while (all(cantClass < nmax)) {
    iter <- iter + 1
    
    # Training 1-NN
    model <- oneNN(y = y.new[labeled])
    prob <- predict(model, D[unlabeled, labeled], type = "prob")
    
    # Select the instances with better class probability 
    selection <- selectInstances(rep(1, nclasses), prob)
    
    # Add selected instances to L
    labeled.prime <- unlabeled[selection$unlabeled.idx]
    sel.classes <- classes[selection$class.idx]
    y.new[labeled.prime] <- sel.classes
    labeled <- c(labeled, labeled.prime)
    
    # Delete selected instances from U
    unlabeled <- unlabeled[-selection$unlabeled.idx]
    
    cantClass <- cantClass + 1
  }
  
  len <- length(labeled)
  if (inicialLen < len) { # new instances were added
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
        if (y.new[labeled[i]] != y.new[labeled[j]])
          J[i] <- J[i] + Wij
      }
    
    # Compute mean and standard desviation of R
    R <- J / I; rm(J,I)
    media <- base::mean(R)
    ds <- stats::sd(R)
    u <- stats::qnorm(1-alpha/2)
    RCritico <- media + u * ds
    
    relabel <- which(R[(labeledLen+1):len] > RCritico)
    for (i in relabel + labeledLen) {
      w <- -1
      if (nclasses > 2) {
        wc <- rep(0, nclasses)
        for (j in ady[[i]]) {
          Wij <- 1 / (1 + D[labeled[i], labeled[j]])
          pos <- unclass(y.new[labeled[j]])
          wc[pos] <- wc[pos] + Wij
        }
        wc[unclass(y.new[labeled[i]])] <- 0
        w <- which.max(wc)
      } else { # if two classes invert the label
        w <- ifelse(unclass(y.new[labeled[i]]) == 1, 2, 1)
      }
      
      if (w != -1)
        y.new[labeled[i]] <- classes[w]
    }
    rm(ady)
  }
  
  result <- list(
    tr.insts = x[labeled,],
    tr.labels = y.new[labeled], 
    dist = dist
  )
  class(result) <- "snnrce"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.snnrce <- function(object, x, ...) {
  D <- proxy::dist(x = x, y = object$tr.insts, method = object$dist, 
                   diag = TRUE, upper = TRUE, by_rows = TRUE)
  
  model <- oneNN(y = object$tr.labels)
  cls <- predict(model, D, type = "class")
  
  return(cls)
}

