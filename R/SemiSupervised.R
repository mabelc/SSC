
# store in labels.map the classes found in y
#' @noRd
# TODO: Delete this function
normalizeLabels <- function(m, y) {
	m$labels.map <- unique(y)
	m
}

# map the labels in y using labels.map
#' @noRd
# TODO: Delete this function
mapLabels <- function(m, y) {
  cls <- m$labels.map
  map <- vapply(y, FUN = function(e){ which(e == cls)}, FUN.VALUE = 1)
  map
}

# unmap the labels in y using labels.map
#' @noRd
# TODO: Delete this function
restoreLabels <- function(m, y) {
  r <- m$labels.map[as.numeric(y)]
  r
}

#' @title Select best instances by classes according to its probabilities
#' @param cantClass A vector indicating how many instance must be selected by class
#' @param probabilities A matrix of probabilities. See \link{probabilities}.
#' @return A list containing a vector by each class of the indices (in indexInstTU) of the instances with major probability per class
#' @noRd
# TODO: Delete this function
selectInst <- function(cantClass, probabilities){
  instSelected <- list() # lista de las posiciones en U de las instancias mas probables por clase

  for (k in 1:sum(cantClass)) { # buscar el mejor por clase y etiquetarlo
      best <- arrayInd(which.max(probabilities), dim(probabilities))
      i <- best[1] # fila (instancia)
      c <- best[2] # columna (clase)
      if (probabilities[i,c] == -1) break;

      if (cantClass[c] > 0) {
          if (length(instSelected) < c || is.null(instSelected[[c]]))
              instSelected[[c]] <- i
          else
              instSelected[[c]] <- c(instSelected[[c]],i)
          cantClass[c] <- cantClass[c] - 1
          probabilities[i,] <- -1 # para que no se repita la instancia
          if (cantClass[c] == 0)
              probabilities[,c] <- -1 # para que no se repita la clase
      }

  }

  instSelected
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

