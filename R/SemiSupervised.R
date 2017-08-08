
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

# TODO: Write help for this function
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

#' @title the algorithm for obtaining a resample of the original
#' labeled set guaranting the representation of each class
#' @param indexInstTL the index of the instances labeled used for training and its classes
#' @param N is the number of bootstrap samples
#' @return a set of bootstrap samples
#' @noRd
# TODO: Change the parameter indexInstTL for the parameters labeled and ynew
resample <- function(indexInstTL, N){

  y <- indexInstTL[,2]
  classes <- unique(y) #determino las diferentes clases
  sizeB <- nrow(indexInstTL) - 2*length(classes)

  bootstrapList <- list()
  for (i in 1:N){
    bootstrapList[[i]] <- indexInstTL[classRepresentationRandom2(classes, indexInstTL),]
    if (sizeB > 0){ #si aun faltan por adicionar instancias
      index <- sample(x = (1:nrow(indexInstTL)), size = sizeB, replace = TRUE) #selecciono los indices
      bootstrapList[[i]] <- rbind(bootstrapList[[i]], indexInstTL[index,])#guardo las instancias correspondientes
    }
  }

  bootstrapList
}

#' @title the algorithm for obtaining a resample with exactly an instance of each class
#' @param classes all classes in the supervised problem
#' @param indexInstTL the index of the instances labeled used for training and its classes
#' @return a set of instances
# @examples classRepresentationRandom(c(1,2,3), indexInstTL)
#' @noRd
# TODO: Change the parameter indexInstTL for the parameters labeled and ynew
classRepresentationRandom1 <- function(classes, indexInstTL){
  indexs <- c()

  for (c in 1:length(classes)){
    allc <- which(indexInstTL[,2] == classes[c])
    if (length(allc) > 1)
      indexs[c] <- sample(x = allc, size = 1)
    else
      indexs[c] <- allc
  }

  indexs
}

#' @title the algorithm for obtaining a resample with exactly two instances of each class
#' @param classes all classes in the supervised problem
#' @param indexInstTL the index of the instances labeled used for training and its classes
#' @return a set of instances
# @example classRepresentationRandom(c(1,2,3),indexInstTL)
#' @noRd
# TODO: Change the parameter indexInstTL for the parameters labeled and ynew
classRepresentationRandom2 <- function(classes, indexInstTL){
  indexs <- c()
  i <- 1
  for (c in 1:length(classes)){
    allc <- which(indexInstTL[,2] == classes[c])
    if (length(allc) > 1){
      s <- sample(x = allc, size = 2)
      indexs[i] <- s[1]
      i <- i+1
      indexs[i] <- s[2]
      i <- i+1
    }
    else{
      indexs[i] <- allc
      i <- i+1
    }
  }

  indexs
}
