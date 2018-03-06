#' @title SNNRCE generic method
#' @description SNNRCE is a variant of the self-training classification 
#' method (\code{\link{selfTraining}}) with a different 
#' addition mechanism and a fixed learning scheme (1-NN). SNNRCE uses an amending scheme 
#' to avoid the introduction of noisy examples into the enlarged labeled set.
#' The mislabeled examples are identified using the local information provided 
#' by the neighborhood graph. A statistical test using cut edge weight is used to modify 
#' the labels of the missclassified examples.
#' @param D A distance matrix between all the training instances. This matrix is used to 
#' construct the neighborhood graph.
#' @param y A vector with the labels of training instances. In this vector the 
#' unlabeled instances are specified with the value \code{NA}.
#' @param alpha Rejection threshold to test the critical region. Default is 0.1.
#' @return A list object of class "snnrceG" containing:
#' \describe{
#'   \item{model}{The final base classifier trained using the enlarged labeled set.}
#'   \item{instances.index}{The indexes of the training instances used to 
#'   train the \code{model}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   Those indexes are relative to the \code{y} argument.}
#' }
#' @noRd
snnrceG <- function(
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
    prob <- predict(model, D[unlabeled, labeled], type = "prob",
                    distance.weighting = "reciprocalexp")
    
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
    instances.index = labeled
  )
  
  class(result) <- "snnrceG"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.snnrceG <- function(object, D, ...) {
  if(class(D) == "dist"){
    D <- proxy::as.matrix(D)
  }
  
  cls <- predict(object$model, D, type = "class")
  return(cls)
}

#' @title SNNRCE method
#' @description SNNRCE (Self-training Nearest Neighbor Rule using Cut Edges) is a variant 
#' of the self-training classification method (\code{\link{selfTraining}}) with a different 
#' addition mechanism and a fixed learning scheme (1-NN). SNNRCE uses an amending scheme 
#' to avoid the introduction of noisy examples into the enlarged labeled set.
#' The mislabeled examples are identified using the local information provided 
#' by the neighborhood graph. A statistical test using cut edge weight is used to modify 
#' the labels of the missclassified examples.
#' @param x A object that can be coerced as matrix. This object has two possible 
#' interpretations according to the value set in the \code{x.inst} argument:
#' a matrix with the training instances where each row represents a single instance
#' or a precomputed distance matrix between the training examples.
#' @param y A vector with the labels of the training instances. In this vector 
#' the unlabeled instances are specified with the value \code{NA}.
#' @param x.inst A boolean value that indicates if \code{x} is or not an instance matrix.
#' Default is \code{TRUE}.
#' @param dist A distance function available in the \code{proxy} package to compute 
#' the distance matrix in the case that \code{x.inst} is \code{TRUE}.
#' @param alpha Rejection threshold to test the critical region. Default is 0.1.
#' @details 
#' SNNRCE initiates the self-labeling process by training a 1-NN from the original 
#' labeled set. This method attempts to reduce the noise in examples by labeling those instances 
#' with no cut edges in the initial stages of self-labeling learning. 
#' These highly confident examples are added into the training set. 
#' The remaining examples follow the standard self-training process until a minimum number 
#' of examples will be labeled for each class. A statistical test using cut edge weight is used 
#' to modify the labels of the missclassified examples The value of the \code{alpha} argument 
#' defines the critical region where the candidates examples are tested. The higher this value 
#' is, the more relaxed it is the selection of the examples that are considered mislabeled.
#'
#' @return A list object of class "snnrce" containing:
#' \describe{
#'   \item{model}{The final base classifier trained using the enlarged labeled set.}
#'   \item{instances.index}{The indexes of the training instances used to 
#'   train the \code{model}. These indexes include the initial labeled instances
#'   and the newly labeled instances.
#'   Those indexes are relative to \code{x} argument.}
#'   \item{classes}{The levels of \code{y} factor.}
#'   \item{x.inst}{The value provided in the \code{x.inst} argument.}
#'   \item{dist}{The value provided in the \code{dist} argument when x.inst is \code{TRUE}.}
#'   \item{xtrain}{A matrix with the subset of training instances referenced by the indexes 
#'   \code{instances.index} when x.inst is \code{TRUE}.}
#' }
#' @references
#' Yu Wang, Xiaoyan Xu, Haifeng Zhao, and Zhongsheng Hua.\cr
#' \emph{Semisupervised learning based on nearest neighbor rule and cut edges.}\cr
#' Knowledge-Based Systems, 23(6):547-554, 2010. ISSN 0950-7051. doi: http://dx.doi.org/10.1016/j.knosys.2010.03.012.
#' @example demo/SNNRCE.R
#' @export
snnrce <- function(
  x, y, x.inst = TRUE,
  dist = "Euclidean",
  alpha = 0.1
){
  ### Check parameters ###
  checkTrainingData(environment())
  
  if(x.inst){
    # Instance matrix case
    result <- snnrceG(
      D = proxy::dist(x, method = dist, by_rows = TRUE,
                      diag = TRUE, upper = TRUE),
      y,
      alpha
    )
  }else{
    # Distance matrix case
    result <- snnrceG(D = x, y, alpha)
  }
  result$classes = levels(y)
  result$x.inst = x.inst
  if(x.inst){
    result$dist <- dist
    result$xtrain <- x[result$instances.index, ]
  }
  class(result) <- "snnrce"
  return(result)
}

#' @title Predictions of the SNNRCE method
#' @description Predicts the label of instances according to the \code{snnrce} model.
#' @details For additional help see \code{\link{snnrce}} examples.
#' @param object SNNRCE model built with the \code{\link{snnrce}} function.
#' @param x A object that can be coerced as matrix.
#' Depending on how was the model built, \code{x} is interpreted as a matrix 
#' with the distances between the unseen instances and the selected training instances, 
#' or a matrix of instances.
#' @param ... This parameter is included for compatibility reasons.
#' @return Vector with the labels assigned.
#' @export
#' @importFrom stats predict
predict.snnrce <- function(object, x, ...) {
  x <- as.matrix2(x)
  
  if(object$x.inst){
    D <- proxy::dist(x, y = object$xtrain, method = object$dist, 
                     diag = TRUE, upper = TRUE, by_rows = TRUE)
    cls <- predict(object$model, D, type = "class")
  }else{
    cls <- predict(object$model, x, type = "class")
  }
  
  return(cls)
}

