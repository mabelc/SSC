
#' @title Train a model
#' @param x matrix of training instances
#' @param y factor of classes
#' @param learner either a function or a string naming the function for 
#' training a supervised base classifier
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

#' @title Predict classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param classes vector of possible classes
#' @return a factor with the predicted classes
#' @noRd
predClass <- function(model, x, pred, pred.pars, classes){
  # Predict probabilities
  prob <- predProb(model, x, pred, pred.pars, classes)
  # Obtain classes from probabilities
  map <- apply(prob, MARGIN = 1, FUN = which.max)
  # Convert classes indexes in a factor of classes
  r <- factor(classes[map], classes)
  
  return(r)
}

#' @title Predict classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param classes vector of possible classes
#' @return a vector with indexes that correspont to \code{classes}. 
#' This is a map of the predicted classes.
#' @noRd
predClassIdx <- function(model, x, pred, pred.pars, classes){
  # Predict probabilities
  prob <- predProb(model, x, pred, pred.pars, classes)
  # Obtain classes from probabilities
  map <- apply(prob, MARGIN = 1, FUN = which.max)
  
  return(map)
}

#' @title Predict probabilities per classes
#' @param model supervised classifier
#' @param x instances to predict
#' @param pred either a function or a string naming the function for
#' predicting the probabilities per classes, using a base classifier in \code{model}.
#' @param pred.pars A list with parameters that are to be passed to the \code{pred}
#' function.
#' @param classes vector of possible classes
#' @return a matrix of predicted probabilities with the column names equals 
#' to \code{classes}
#' @noRd
predProb <- function(model, x, pred, pred.pars, classes) {
  # Predict probabilities
  ppars <- c(list(model, x), pred.pars)
  # TODO: Call pred function using a try cast function
  prob <- do.call(pred, ppars)
  
  # Check probabilities matrix
  if(!is.matrix(prob) ||
     nrow(x) != nrow(prob) ||
     length(classes) != length(intersect(classes, colnames(prob)))){
    # TODO: Explain the error cause in the next error message
    stop("Incorrect value returned by pred function.")
  }
  r <- prob[, classes]
  if(nrow(prob) == 1) dim(r) <- dim(prob) 
  colnames(r) <- classes
  
  return(r)
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
      c(
        indexes, 
        sample(x = 1:length(ylabeled), size = sizeB, replace = TRUE)
      ) -> indexes
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

