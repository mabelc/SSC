
#' @example demo/DemocraticBase.R
#' @export
democraticBase <- function(
  y,
  learnersB,
  predsB
) {
  ### Check parameters ###
  # Check y 
  if(!is.factor(y) ){
    if(!is.vector(y)){
      stop("Parameter y is neither a vector nor a factor.")  
    }else{
      y = as.factor(y)
    }
  }
  # Check lengths
  if(length(learnersB) != length(predsB)){
    stop("The length of learnersB is not equal to the length of predsB.")
  }
  nclassifiers <- length(learnersB) 
  if (nclassifiers <= 1) {
    stop("learnersB must contain at least two base classifiers.") 
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
  
  H <- vector(mode = "list", length = nclassifiers)
  Lind <- vector(mode = "list", length = nclassifiers)
  Lcls <- vector(mode = "list", length = nclassifiers)
  e <- vector(mode = "numeric", length = nclassifiers)
  for (i in 1:nclassifiers) {
    H[[i]] <- learnersB[[i]](labeled, y[labeled])
    Lind[[i]] <- labeled
    Lcls[[i]] <- y.map[labeled]
    e[i] <- 0
  }
  
  iter <- 1
  changes <- TRUE
  while (changes) { #while some classifier changes
    
    changes <- FALSE
    
    LindPrima <- vector(mode = "list", length = nclassifiers)
    LclsPrima <- vector(mode = "list", length = nclassifiers)
    
    # Internal classify
    predU <- mapply(    
      FUN = function(model, pred){
        getClassIdx(
          checkProb(
            prob = pred(model, unlabeled),
            ninstances = length(unlabeled), 
            classes
          )
        )
      },
      H, predsB
    )
    
    cls <- vote(predU) # etiquetas votadas
    
    # End Internal classify
    
    # compute the confidence interval over the original set L
    W <- mapply(
      FUN = function(model, pred){
        confidenceInterval(
          getClassIdx(
            checkProb(
              prob = pred(model, labeled), 
              ninstances = length(labeled), 
              classes
            )
          ),
          y.map[labeled]
        )$W
      },
      H, predsB
    )
    
    for (i in 1:nunlabeled) { #for each unlabeled example x in U
      # is the sum of the mean confidence values of the learners in the majority
      # group greater than the sum of the mean confidence values in the minority group??
      sumW <- rep(0, nclasses)
      for (j in 1:nclassifiers) #for each classifier
        sumW[predU[i, j]] <- sumW[predU[i, j]] + W[j]
      
      # Calculate the maximum confidence with different label to predicted.
      lab <- cls[[i]][which.max(sumW[cls[[i]]])] #se devuelve la etiqueta mas probable
      tmp <- sumW[lab] # la confidencia total asociada a esa etiqueta
      sumW[lab] <- -Inf #para no volverla a seleccionar
      Max <- which.max(sumW)#la segunda clase con mayor confidencia
      sumW[lab] <- tmp
      
      if (sumW[lab] > sumW[Max]) {
        # if the classifier i does not label this X unlabeled as predicted, add it to Li.
        for (j in 1:nclassifiers)
          if (predU[i, j] != lab) {# wrong label
            LindPrima[[j]] <- c(LindPrima[[j]], unlabeled[i])
            LclsPrima[[j]] <- c(LclsPrima[[j]], lab)
          }
      }
    }# end for each unlabeled example x in U
    
    
    # Estimate if adding Li' to Li improves the accuracy
    # AQUI
    
    LindUnion <- vector(mode = "list", length = nclassifiers)
    LclsUnion <- vector(mode = "list", length = nclassifiers)
    
    for (i in 1:nclassifiers) {
      repeated <- intersect(Lind[[i]], LindPrima[[i]])
      if (length(repeated) != 0){
        indexesToRemove <- sapply(
          X = repeated, 
          FUN = function(r) {
            which(LindPrima[[i]] == r)
          }
        )
        LindPrima[[i]] <- LindPrima[[i]][-indexesToRemove]
        LclsPrima[[i]] <- LclsPrima[[i]][-indexesToRemove]
      }
      if (!is.null(LindPrima[[i]])){
        LindUnion[[i]] <- c(Lind[[i]], LindPrima[[i]])
        LclsUnion[[i]] <- c(Lcls[[i]], LclsPrima[[i]])
      } else{
        LindUnion[[i]] <- Lind[[i]]
        LclsUnion[[i]] <- Lcls[[i]]
      }
    }
    
    L <- mapply(
      FUN = function(model, pred){
        confidenceInterval(
          getClassIdx(
            checkProb(
              prob = pred(model, Lind[[i]]), 
              ninstances = length(Lind[[i]]), 
              classes
            )
          ),
          Lcls[[i]]
        )$L
      },
      H, predsB
    )
    
    q <- ep <- qp <- NULL
    for (i in 1:nclassifiers) { # for each classifier
      sizeLi <- length(Lind[[i]])
      sizeLLP <- length(LindUnion[[i]])
      if (sizeLLP > sizeLi) { #hay instancias nuevas eb LiPrima
        q[i] <- sizeLi * (1 - 2 * ( e[i] / sizeLi)) ^ 2				# est. of error rate
        ep[i] <- (1 - mean(L[-i])) * length(LindPrima[[i]])			# est. of new error rate
        qp[i] <- sizeLLP * (1 - 2 * (e[i] + ep[i]) / sizeLLP) ^ 2		# if Li' added
        
        if (qp[i] > q[i]) {
          Lind[[i]] <- LindUnion[[i]]
          Lcls[[i]] <- LclsUnion[[i]]
          e[i] <- e[i] + ep[i]
          changes <- TRUE
          # entrenar clasificador i
          yi <- classes[Lcls[[i]]]
          H[[i]] <- learnersB[[i]](Lind[[i]], factor(yi, classes))
        }
      }
    } # end for each classifier
    iter <- iter + 1
  } # End while
  
  ### Result ###
  
  # determine labeled instances
  instances.index <- unique(unlist(Lind))
  # map indexes respect to m$included.insts
  model.index <- lapply(
    X = Lind,
    FUN = function(indexes)
      unclass(factor(indexes, levels = instances.index))
  )
  
  # compute W
  W <- mapply(
    FUN = function(model, pred){
      confidenceInterval(
        getClassIdx(
          checkProb(
            prob = pred(model, labeled), 
            ninstances = length(labeled), 
            classes
          )
        ),
        y.map[labeled]
      )$W
    },
    H, predsB
  )
  
  # Save result
  result <- list(
    model = H,
    model.index = model.index,
    instances.index = instances.index,
    W = W,
    classes = classes
  )
  class(result) <- "democraticBase"
  
  return(result)
}

#' @example demo/Democratic.R
#' @export
democratic <- function(
  x, y,
  learners, learners.pars,
  preds, preds.pars,
  x.dist = FALSE
) {
  ### Check parameters ###
  # Check x.dist
  if(!is.logical(x.dist)){
    stop("Parameter x.dist is not logical.")
  }
  # Check learners
  if (length(learners)  <= 1) {
    stop("Parameter learners must contain at least two base classifiers.") 
  }
  if(!(length(learners) == length(learners.pars) &&
       length(preds) == length(preds.pars) &&
       length(learners) == length(preds))){
    stop("The lists: learners, learners.pars, preds and preds.pars must be of the same length.")
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
    
    # Build learner base functions
    d_learners_base <- mapply(
      FUN = function(learner, learner.pars){
        learner_base <- function(training.ints, cls){
          m <- trainModel(x[training.ints, training.ints], cls, learner, learner.pars)
          r <- list(m = m, training.ints = training.ints)
          return(r)
        }
        return(learner_base)
      }, 
      learners,
      learners.pars,
      SIMPLIFY = FALSE
    )
    # Build pred base functions 
    d_preds_base <- mapply(
      FUN = function(pred, pred.pars){
        pred_base <- function(r, testing.ints){
          prob <- predProb(r$m, x[testing.ints, r$training.ints], pred, pred.pars)
          return(prob)
        }    
        return(pred_base)
      }, 
      preds,
      preds.pars,
      SIMPLIFY = FALSE
    )
    # Call base method
    result <- democraticBase(y, d_learners_base, d_preds_base)
    # Extract model from list created in learnerB
    result$model <- lapply(X = result$model, FUN = function(e) e$m)
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
    
    # Build learner base functions
    m_learners_base <- mapply(
      FUN = function(learner, learner.pars){
        learner_base <- function(training.ints, cls){
          m <- trainModel(x[training.ints, ], cls, learner, learner.pars)
          return(m)
        }
        return(learner_base)
      }, 
      learners,
      learners.pars,
      SIMPLIFY = FALSE
    )
    # Build pred base functions 
    m_preds_base <- mapply(
      FUN = function(pred, pred.pars){
        pred_base <-  function(m, testing.ints){
          prob <- predProb(m, x[testing.ints, ], pred, pred.pars)
          return(prob)
        }
        return(pred_base)
      }, 
      preds,
      preds.pars,
      SIMPLIFY = FALSE
    )
    # Call base method
    result <- democraticBase(y, m_learners_base, m_preds_base)
  }
  
  ### Result ###
  result$classes = levels(y)
  result$preds = preds
  result$preds.pars = preds.pars
  result$x.dist = x.dist
  class(result) <- "democratic"
  
  return(result)
}

#' @export
#' @importFrom stats predict
predict.democratic <- function(object, x, ...){
  if(class(x) == "integer"){
    x <- matrix(x, nrow = 1)
  } else if(class(x) == "dist"){
    x <- proxy::as.matrix(x)
  }
  
  # Select classifiers for prediction
  lower.limit <- 0.5
  selected <- object$W > lower.limit # TODO: create a parameter for 0.5 lower limit
  W.selected <- object$W[selected]
  if(length(W.selected) == 0){
    stop(
      sprintf(
        "%s %s %f",
        "Any classifier selected according model's W values.", 
        "The classifiers are selected when it's W value is greater than",
        lower.limit
      )
    )
  }
  
  # Classify the instances using each classifier
  # The result is a matrix of indexes that indicates the classes
  # The matrix have one column per selected classifier 
  # and one row per instance
  ninstances = nrow(x)
  if(object$x.dist){
    pred <- mapply(
      FUN = function(model, indexes, pred, pred.pars){
        getClassIdx(
          checkProb(
            predProb(model, x[, indexes], pred, pred.pars), 
            ninstances, 
            object$classes
          )
        ) 
      },
      object$model[selected], 
      object$model.index[selected], 
      object$preds[selected], 
      object$preds.pars[selected]
    )
  }else{
    pred <- mapply(
      FUN = function(model, pred, pred.pars){
        getClassIdx(
          checkProb(
            predProb(model, x, pred, pred.pars), 
            ninstances, 
            object$classes 
          )
        ) 
      },
      object$model[selected],
      object$preds[selected],
      object$preds.pars[selected]
    )
  }
  
  # Combining predictions
  map <- vector(mode = "numeric", length = ninstances)
  
  for (i in 1:nrow(pred)) {#for each example x in U
    pertenece <- wz <- rep(0, length(object$classes))

    for (j in 1:ncol(pred)) {#para cada clasificador
      z = pred[i, j]
      # Allocate this classifier to group Gz
      pertenece[z] <- pertenece[z] + 1
      wz[z] <- wz[z] + W.selected[j]
    }

    # Compute group average mean confidence
    countGj <- (pertenece + 0.5) / (pertenece + 1) * (wz / pertenece)
    map[i] <- which.max(countGj)
  }# end for

  cls <- factor(object$classes[map], object$classes)

  return(cls)
}

#' @export
democraticCombining <- function(pred, W, classes){
  
  map <- vector(mode = "numeric", length = nrow(x))
  
  ninstances <- length(pred[[1]])
  nclassifiers <- length(W)
  for (i in 1:ninstances) {#for each example x in U
    pertenece <- wz <- rep(0, length(classes))
    
    for (j in 1:nclassifiers) {#para cada clasificador
      z <- which(pred[[j]][i] == classes)
      if (W[j] > 0.5) {
        # Allocate this classifier to group Gz
        pertenece[z] <- pertenece[z] + 1
        wz[z] <- wz[z] + W[j]
      }
    }
    
    # Compute group average mean confidence
    countGj <- (pertenece+0.5)/(pertenece+1) * (wz / pertenece)
    map[i] <- which.max(countGj)
  }# end for
  
  factor(classes[map], classes)
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
#' @param pred Matrix con la prediccion de cada 
#' clasificador para cada instancia
#' @return A list of possibles labels for each instance
#' @noRd
vote <- function(pred){
  FUN = function(p){
    as.numeric(names(which.max(summary(factor(p)))))
  }
  lab <- apply(X = pred, MARGIN = 1, FUN)
  return(lab)
}
