
checkTrainingData <- function(e){
  e$y <- as.factor(e$y)
  e$x.inst <- as.logical(e$x.inst)
  if(!is.logical(e$x.inst)){
    stop("Parameter x.inst is not logical.")
  }
  if(e$x.inst){
    # Check x
    if(!is.matrix(e$x) && !is.data.frame(e$x)){
      stop("Parameter x is neither a matrix or a data frame.")
    }
    # Check relation between x and y
    if(nrow(e$x) != length(e$y)){
      stop("The rows number of x must be equal to the length of y.")
    }
  }else{
    # Check x
    e$x <- as.matrix(e$x)
    if(!is.matrix(e$x)){
      stop("Parameter x is not a matrix.")
    }
    if(nrow(e$x) != ncol(e$x)){
      stop("The distance matrix x is not a square matrix.")
    } else if(nrow(e$x) != length(e$y)){
      stop(sprintf(paste("The dimensions of the matrix x is %i x %i", 
                         "and it's expected %i x %i according to the size of y."), 
                   nrow(e$x), ncol(e$x), length(e$y), length(e$y)))
    }
  }
}

as.matrix2 <- function(x){
  if(is.matrix(x)){
    return(x)
  }else{
    return(matrix(x, nrow = 1))
  }
}

as.list2 <- function(x, len = 0){
  if(is.null(x)){
    return(vector("list", len))
  }else{
    return(x)
  }
}

#' @title Train a model
#' @param x matrix of training instances
#' @param y factor of classes
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier, using a set of instances
#' (or optionally a distance matrix) and it's corresponding classes.
#' @param learner.pars a list with parameters that are to be passed to the \code{learner}
#' function.
#' @return the trained model 
#' @noRd
trainModel <- function(x, y, learner, learner.pars){
  # Train a model
  lpars <- c(list(x, y), learner.pars)
  # TODO: Call learner function using a try cast function
  model <- do.call(learner, lpars)
  
  return(model)
}

#' @title Predict probabilities per classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @return a matrix of predicted probabilities
#' @noRd
predProb <- function(model, x, pred, pred.pars) {
  # Predict probabilities
  ppars <- c(list(model, x), pred.pars)
  # TODO: Call pred function using a try cast function
  prob <- do.call(pred, ppars)
  
  return(prob)
}

#' @title Check a matrix of probabilities
#' @description Check the number of rows and the columns names
#' of a matrix of probabilities. If the columns are
#' unordered, they are ordered according to \code{classes}.
#' @param prob a probabilities matrix
#' @param ninstances expected number of rows in \code{prob}
#' @param classes expected columns names in \code{prob}
#' @return the matrix \code{prob} with it columns in the order given by \code{classes} 
#' @noRd
checkProb <- function(prob, ninstances, classes){
  # Check probabilities matrix
  if(!is.matrix(prob)){
    stop(
      sprintf(
        paste0(
          "Predict function incorrect output.\n",
          "'prob' is an object of class %s.\n",  
          "Expected an object of class matrix."
        ),
        class(prob)
      )
    )
  }
  if(ninstances != nrow(prob)){
    stop(
      sprintf(
        paste0(
          "Predict function incorrect output.\n",
          "The row number of 'prob' is %s.\n",
          "Expected a number equal to %i (value of 'ninstances')."
        ), 
        nrow(prob), 
        ninstances)
    )
  }
  if(length(classes) != ncol(prob)){
    stop(
      sprintf(
        paste0(
          "Predict function incorrect output.\n",
          "The column number of 'prob' is %s.\n",
          "Expected a number equal to %i (length of 'classes')."
        ), 
        ncol(prob), 
        length(classes))
    )
  }
  if(length(classes) != length(intersect(classes, colnames(prob)))){
    stop(
      paste0(
        "Predict function incorrect output.\n",
        "The columns names of 'prob' is a set not equal to 'classes' set."
      )
    )
  } else {
    # order columns by classes
    prob <- prob[, classes]
    if(!is.matrix(prob)){
      # when nrow of prob is 1
      prob <- matrix(prob, nrow = 1)
      colnames(prob) <- classes
    }
  }
  
  return(prob)
}

#' @title Get classes from a matrix of probabilities
#' @param prob a probabilities matrix
#' @return a factor with classes
#' @noRd
getClass <- function(prob){
  # Obtain classes from probabilities
  map <- apply(prob, MARGIN = 1, FUN = which.max)
  # Convert classes indexes in a factor of classes
  classes <- colnames(prob)
  r <- factor(classes[map], classes)
  
  return(r)
}

#' @title Get classes from a matrix of probabilities and 
#' return the classes indexes
#' @param prob a probabilities matrix
#' @return a vector of indexes corresponding to \code{classes}
#' @noRd
getClassIdx <- function(prob){
  # Obtain classes from probabilities
  map <- apply(prob, MARGIN = 1, FUN = which.max)
  
  return(map)
}


#' @title Select best instances by classes according to its probabilities
#' @param cantClass A vector indicating how many instances must be selected by class.
#' @param probabilities A matrix of probabilities. See \link{probabilities}.
#' @return A dataframe where the rows contains the information of the selected instances.
#' @noRd
selectInstances <- function(cantClass, probabilities){
  len <- 0
  class.idx <- numeric()
  unlabeled.idx <- numeric()
  prob.cls <- numeric()
  
  for (k in 1:sum(cantClass)) { # buscar el mejor por clase y etiquetarlo
    best <- arrayInd(which.max(probabilities), dim(probabilities))
    i <- best[1] # fila (instancia)
    c <- best[2] # columna (clase)
    if (probabilities[i,c] == -1){
      break;
    } 
    
    if (cantClass[c] > 0) {
      len <- len + 1
      class.idx[len] <- c
      unlabeled.idx[len] <- i
      prob.cls[len] <- probabilities[i, c]
      
      cantClass[c] <- cantClass[c] - 1
      probabilities[i,] <- -1 # para que no se repita la instancia
      if (cantClass[c] == 0)
        probabilities[,c] <- -1 # para que no se repita la clase
    }
    
  }
  
  r <- data.frame(class.idx = class.idx, unlabeled.idx = unlabeled.idx, prob.cls = prob.cls)
  return(r)
}

#' @title A algorithm for obtaining a resample of the original
#' labeled set guaranting the representation of each class
#' @param ylabeled a factor of instances labels
#' @param N is the number of bootstrap samples
#' @return a set of bootstrap samples
#' @examples 
#' ylabeled = factor(c('a','b','a','b','c','b','c','c'))
#' resample(ylabeled, 3)
#' @noRd
resample <- function(ylabeled, N){

  classes <- levels(ylabeled) 
  sizeB <- length(ylabeled) - 2 * length(classes)

  bootstrapList <- list()
  for (i in 1:N){
    indexes <- classRepresentationRandom2(ylabeled)
    if (sizeB > 0){ #si aun faltan por adicionar instancias
      # Select the indexes
      indexes <- c(
        indexes, 
        sample(x = 1:length(ylabeled), size = sizeB, replace = TRUE)
      )
    }
    bootstrapList[[i]] <- indexes
  }

  return(bootstrapList) 
}

#' @title A algorithm for obtaining a resample with exactly an instance of each class
#' @param ylabeled a factor of instances labels
#' @return Indexes of selected instances from \code{ylabeled}
#' @examples 
#' ylabeled = factor(c(1,2,1,2,3,2))
#' classRepresentationRandom1(ylabeled)
#' @noRd
classRepresentationRandom1 <- function(ylabeled){
  indexes <- numeric()

  classes <- levels(ylabeled)
  for (i in 1:length(classes)){
    allc <- which(ylabeled == classes[i])
    if (length(allc) > 1)
      indexes[i] <- sample(x = allc, size = 1)
    else
      indexes[i] <- allc
  }

  return(indexes)
}

#' @title A algorithm for obtaining a resample with exactly two instances of each class
#' @param ylabeled a factor of instances labels
#' @return Indexes of selected instances from \code{ylabeled}
#' @examples 
#' ylabeled = factor(c(1,2,1,2,3,2))
#' classRepresentationRandom2(ylabeled)
#' @noRd
classRepresentationRandom2 <- function(ylabeled){
  i <- 1
  indexes <- numeric()
  
  for (cls in levels(ylabeled)){
    allc <- which(ylabeled == cls)
    if (length(allc) > 1) {
      s <- sample(x = allc, size = 2)
      indexes[i] <- s[1]
      i <- i + 1
      indexes[i] <- s[2]
      i <- i + 1
    }  else {
      indexes[i] <- allc
      i <- i + 1
    }
  }

  return(indexes)
}

