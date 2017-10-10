
#' @title Train a Democratic model
#' @description Trains a model for classification,
#' according to Democratic algorithm.
#' @param x A matrix or a dataframe with the training instances.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param learners A list of learner functions from diferents learning squemes.
#' The learners are used to train supervised base classifiers.
#' @param learners.pars A list of parameter sets, 
#' where each set is used with its corresponding learner,
#' according to the order in \code{learners}.
#' @param preds A list of prediction functions to obtain the probabilities per classes.
#' Each function is used in conjuntion with the trained model provided in \code{learners}.
#' @param preds.pars A list of parameter sets for the functions in \code{preds} list. 
#' According to the position, each parameter set corresponds with a specific function.
#' @return The trained model.
#' @export
democratic <- function(
  x, y,
  learners,
  learners.pars,
  preds,
  preds.pars
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
  # Check lengths
  if(!(length(learners) == length(learners.pars) &&
       length(preds) == length(preds.pars) &&
       length(learners) == length(preds))){
    stop("The lists: learners, learners.pars, preds and preds.pars must be of the same length.")
  }
  N <- length(learners) 
  if (N <= 1) {
    stop("learners must contain at least two base classifiers.") 
  }
  
  
  ### Init variables ###
  # Identify the classes
  classes <- levels(y)
  nclasses <- length(classes)
  
  # Init variable to store the labels
  ynew <- y
  
  # Obtain the indexes of labeled and unlabeled instances
  labeled <- which(!is.na(y))
  unlabeled <- which(is.na(y))
  nunlabeled <- length(unlabeled)
  ## Check the labeled and unlabeled sets
  if(length(labeled) == 0){   # labeled is empty
    stop("The labeled set is empty. All the values in y parameter are NA.")
  }
  if(length(unlabeled) == 0){ # unlabeled is empty
    stop("The unlabeled set is empty. None value in y parameter is NA.")
  }
  
  ### Democratic algorithm ###
  
  y.map <- unclass(y)
  
  H <- vector(mode = "list", length = N)
  Lind <- vector(mode = "list", length = N)
  Lcls <- vector(mode = "list", length = N)
  e <- vector(mode = "numeric", length = N)
  for (i in 1:N) {
    H[[i]] <- trainModel(x[labeled, ], y[labeled], 
                         learners[[i]], learners.pars[[i]])
    Lind[[i]] <- labeled
    Lcls[[i]] <- y.map[labeled]
    e[i] <- 0
  }
  
  iter <- 1
  changes <- TRUE
  while (changes) { #while some classifier changes
    
    changes <- FALSE
    
    LindPrima <- vector(mode = "list", length = N)
    LclsPrima <- vector(mode = "list", length = N)
    
    # Internal classify
    predU <- matrix(nrow = N, ncol = nunlabeled)
    
    for (i in 1:N) {
      predU[i,] <- predClassIdx(H[[i]], x[unlabeled, ], 
                               preds[[i]], preds.pars[[i]], classes)
    }
    
    cls <- vote(predU, nclasses) # etiquetas votadas
    
    # End Internal classify
  
    # compute the confidence interval over the original set L
    W <- sapply(1:N, function(i){
      predL <- predClassIdx(H[[i]], x[labeled,], preds[[i]], preds.pars[[i]], classes)
      confidenceInterval(predL, y.map[labeled])$W}
    )
    
    for (i in 1:nunlabeled) { #for each unlabeled example x in U
      # is the sum of the mean confidence values of the learners in the majority
      # group greater than the sum of the mean confidence values in the minority group??
      sumW <- rep(0, nclasses)
      for (j in 1:N) #for each classifier
        sumW[predU[j,i]] <- sumW[predU[j,i]] + W[j]
      
      # Calculate the maximum confidence with different label to predicted.
      lab <- cls[[i]][which.max(sumW[cls[[i]]])] #se devuelve la etiqueta mas probable
      tmp <- sumW[lab] # la confidencia total asociada a esa etiqueta
      sumW[lab] <- -Inf #para no volverla a seleccionar
      Max <- which.max(sumW)#la segunda clase con mayor confidencia
      sumW[lab] <- tmp
      
      if (sumW[lab] > sumW[Max]) {
        # if the classifier i does not label this X unlabeled as predicted, add it to Li.
        for (j in 1:N)
          if (predU[j,i] != lab) {# wrong label
            LindPrima[[j]] <- c(LindPrima[[j]], unlabeled[i])
            LclsPrima[[j]] <- c(LclsPrima[[j]], lab)
          }
      }
    }# end for each unlabeled example x in U
    
    
    # Estimate if adding Li' to Li improves the accuracy
    # AQUI
    
    LindUnion <- vector(mode = "list", length = N)
    LclsUnion <- vector(mode = "list", length = N)
    
    for (i in 1:N) {
      repeated <- intersect(Lind[[i]],LindPrima[[i]])
      if (length(repeated) != 0){#elimino instancias que ya esten en temp$idxInstLi
        indexesToRemove <- sapply( X=repeated, FUN = function(r){
          which(LindPrima[[i]] == r)
        })
        LindPrima[[i]] <- LindPrima[[i]][-indexesToRemove]
        LclsPrima[[i]] <- LclsPrima[[i]][-indexesToRemove]
      }
      if (!is.null(LindPrima[[i]])){
        LindUnion[[i]] <- c(Lind[[i]],LindPrima[[i]])
        LclsUnion[[i]] <- c(Lcls[[i]],LclsPrima[[i]])
      } else{
        LindUnion[[i]] <- Lind[[i]]
        LclsUnion[[i]] <- Lcls[[i]]
      }
        
    }
    
    L <- sapply(1:N, function(i){
      predLi <- predClassIdx(H[[i]], x[Lind[[i]],], preds[[i]], preds.pars[[i]], classes)
      confidenceInterval(predLi, Lcls[[i]])$L
    })
    
    q <- ep <- qp <- NULL
    for (i in 1:N) { # for each classifier
      sizeLi <- length(Lind[[i]])
      sizeLLP <- length(LindUnion[[i]])
      if (sizeLLP > sizeLi) { #hay instancias nuevas eb LiPrima
        q[i] <- sizeLi * (1 - 2*(e[i]/sizeLi))^2				# est. of error rate
        ep[i] <- (1 - mean(L[-i])) * length(LindPrima[[i]])			# est. of new error rate
        qp[i] <- sizeLLP * (1 - 2*(e[i]+ep[i])/sizeLLP)^2		# if Li' added
        
        if (qp[i] > q[i]) {
          #cat("Add", sizeLLP - sizeLi,"instances to",class(BaseL[[i]]),"\n")
          Lind[[i]] <- LindUnion[[i]]
          Lcls[[i]] <- LclsUnion[[i]]
          e[i] <- e[i] + ep[i]
          changes <- TRUE
          # entrenar clasificador i
          yi <- classes[Lcls[[i]]]
          H[[i]] <- trainModel(x[Lind[[i]], ], factor(yi, classes), 
                               learners[[i]], learners.pars[[i]])
        }
      }
    } # end for each classifier
    iter <- iter + 1
  } # End while

  ### Result ###
  W <- sapply(1:N, function(i){
    predL <- predClassIdx(H[[i]], x[labeled,], preds[[i]], preds.pars[[i]], classes)
    confidenceInterval(predL, y.map[labeled])$W}
  )
  
  # Save result
  result <- list(
    models = H,
    classes = classes,
    preds = preds,
    preds.pars = preds.pars,
    W = W
  )
  class(result) <- "democratic"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.democratic <- function(object, x, ...){
  if(!is.matrix(x)){
    x <- matrix(x, nrow = 1)
  }
  
  N <- length(object$models)
  pred <- matrix(nrow = N, ncol = nrow(x))
  
  for (i in 1:N) {
    pred[i,] <- predClassIdx(object$models[[i]], x, 
                             object$preds[[i]], object$preds.pars[[i]],
                             object$classes)
  }  
  
  map <- vector(mode = "numeric", length = nrow(x))
  
  for (i in 1:nrow(x)) {#for each example x in U
    pertenece <- wz <- rep(0, length(object$classes))
    
    for (j in 1:N) {#para cada clasificador
      z = pred[j,i]
      if (object$W[j] > 0.5) {
        # Allocate this classifier to group Gz
        pertenece[z] <- pertenece[z] + 1
        wz[z] <- wz[z] + object$W[j]
      }
    }
    
    # Compute group average mean confidence
    countGj <- (pertenece+0.5)/(pertenece+1) * (wz / pertenece)
    map[i] <- which.max(countGj)
  }# end for
  
  cls <- factor(object$classes[map], object$classes)
  
  return(cls)
}

#' @title Compute the 95\% confidence interval of the classifier
#' @noRd
confidenceInterval <- function(pred, conf.cls) {
  # accuracy
  W <- length(which(pred == conf.cls)) / length(conf.cls) 
  
  # lowest point of the confidence interval
  L <- W - 1.96 * sqrt(W*(1-W) / length(conf.cls)) 
  
  list(L = L, W = W)
}


#' @title Calcula la etiqueta de cada instancia por mayoria
#' @param pred Matrix con la prediccion de cada clasificador para cada instancia
#' @param ncls cantidad de clases
#' @return A list of possibles labels for each instance
#' @noRd
vote <- function(pred, ncls){
  lab <- list() # etiquetas
  
  for (i in 1:ncol(pred)) { # para cada instancia
    perClass <- rep(0,ncls)
    for (j in 1:nrow(pred)) # para cada clasificador
      perClass[pred[j,i]] <- perClass[pred[j,i]] + 1
    
    l <- which.max(perClass)
    lab[[i]] <- which(perClass == perClass[l])
  }
  
  lab
}
