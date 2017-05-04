
#' @title Train the Democratic model
#' @description Builds and trains a model to predict the label of instances,
#' according to Democratic algorithm.
#' @param x A object that can be coerced as matrix.
#' This object have various interpretations depending on the value set in \code{dist} argument.
#' See \code{dist} argument.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param bclassifs A list with the classifiers specification.
#' For defining new base classifiers see \code{\link{bClassif}}.
#' @param dist Distance information. Valid options are:
#' \itemize{
#'    \item \code{"matrix"}: this string indicates that \code{x} is a distance matrix or a list of distance matrices.
#'    \item \emph{string}: the name of a distance method available in \code{proxy} package.
#'    In this case \code{x} is interpreted as a matrix of instances.
#'    \item \emph{function}: a function defined by the user that computes the distance between two vectors.
#'    This function is called passing the vectors in the firsts two arguments. If the function have others arguments,
#'    those arguments must be have default values.
#'    In this case \code{x} is interpreted as a matrix of instances.
#' }
#' @param dist.use Boolean matrix that indicates which base classifier uses each distance in case that more
#' than one distance is supplied. Each row represents a classifier and each column a distance.
#' @return The trained model stored in a list with the following named values:
#' \describe{
#'   \item{models}{A list that have a model trained for each classifier specification
#'   in \code{bclassifs} argument.}
#'   \item{bclassif}{The value of \code{bclassif} argument.}
#'   \item{dist}{The value of \code{dist} argument.
#'   This value is optionally returned when the argument \code{dist} is different from "matrix".}
#'   \item{dist.use}{The value of \code{dist.use} argument.}
#'   \item{W}{A vector with the confidence-weighted vote assigned to each classifier.}
#'   \item{included.insts}{The indexes of the training instances used.
#'   This set is the union of the indexes used to train each model in \code{models}.
#'   These indexes are relative to \code{x} argument.}
#'   \item{indexes}{ A list of vectors. Each vector indicates the indexes of the instances
#'   used to train the corresponding model in \code{models}. For example, the instances used to train
#'   \code{models[1]} are \code{x[included.insts[indexes[1]], ]}.}
#'   \item{tr.insts}{The training instances corresponding to the indexes returned in \code{included.insts}.
#'   This value is optionally returned when the \code{dist} argument is different from "matrix".}
#'   \item{ncls}{The number of classes in the training set.}
#'   \item{labels.map}{An internal map used for the labels.}
#' }
#' @references
#' Yan Zhou and Sally Goldman.\cr
#' \emph{Democratic co-learning.}\cr
#' In IEEE 16th International Conference on Tools with Artificial Intelligence (ICTAI),
#' pages 594–602. IEEE, Nov 2004. doi: 10.1109/ICTAI.2004.48.
#' @example demo/Democratic.R
#' @export
democratic <- function(
  x, y,
  bclassifs, dist = "matrix", dist.use = NULL
) {

  # Create the enviroment temp for maintains temporal data
  temp <- new.env(parent = emptyenv())
  # Create the list m for maintains the model data
  m <- list()
  m$bclassifs <- bclassifs
  # check common parameters
  dist.is.matrix <- NULL # initialized for checkCommonParams
  checkCommonParams(env = environment(), denv = temp)
  # check dist.use
  m$dist.use <- if(is.null(dist.use)){
    matrix(TRUE, nrow = length(m$bclassifs), ncol = length(temp$dists.l))
  }else{
    if(nrow(dist.use) != length(m$bclassifs) || ncol(dist.use) != length(temp$dists.l)){
      stop("The number of rows and columns of 'dist.use' must be equal to the number of base algorithms and the number of matrices of distances respectively")
    }
    dist.use
  }

  N <- length(m$bclassifs) # cantidad de algoritmos clasificadores
  if (N <= 1) stop("bclassifs must contain at least two base classifiers")

  yy <- y[!is.na(y)]
  m <- normalizeLabels(m, yy)
  indexInstTL <- cbind(which(!is.na(y)), mapLabels(m, yy)) # indexes and clases of labeled instances
  indexInstTU <- which(is.na(y)) # indexes of unlabeled instances

  ncls <- length(yy) # cantidad de clases, las clases estan numeradas de 1..n

  lenU <- length(indexInstTU) #size of U

  e <- c()   # estimate for # of mislabeled exs in Li
  temp$idxInstLi <- list()
  for (i in 1:N){
    temp$idxInstLi[[i]] <- indexInstTL	# instancias etiquetadas para clasificador Ai
    e[i] <- 0
  }

  temp$models <- list()
  for (i in 1:N) {#train all classifiers
    temp$models[[i]] <- runTrain(m$bclassifs[[i]], temp$dists.l[m$dist.use[i,]], inds = temp$idxInstLi[[i]][,1], y = temp$idxInstLi[[i]][,2])
  }

  iter <- 1
  changes <- TRUE
  while (changes) { #while some classifier changes

    changes <- FALSE

    idxInstLiPrima <- vector("list", N)

    out <- internalClassify(m, temp, indexInstTU, ncls)
    pred <- out$pred
    labels <- out$lab
    rm(out)

    # compute the confidence interval over the original set L
    temp$W <- sapply(1:N, function(i){
      confidenceInterval(m, temp, i, indexInstTL)$W}
    )

    for (i in 1:lenU) { #for each unlabeled example x in U
      # is the sum of the mean confidence values of the learners in the majority
      # group greater than the sum of the mean confidence values in the minority group??
      sumW <- rep(0, ncls)
      for (j in 1:N)
        sumW[pred[j,i]] <- sumW[pred[j,i]] + temp$W[j]

      # Calculate the maximum confidence with different label to predicted.
      lab <- labels[[i]][which.max(sumW[labels[[i]]])] #se devuelve la etiqueta mas probable
      tmp <- sumW[lab] # la confidencia total asociada a esa etiqueta
      sumW[lab] <- -Inf #para no volverla a seleccionar
      Max <- which.max(sumW)#la segunda clase con mayor confidencia
      sumW[lab] <- tmp

      if (sumW[lab] > sumW[Max]) {
        # if the classifier i does not label this X unlabeled as predicted, add it to Li.
        for (j in 1:N)
          if (pred[j,i] != lab) {
            idxInstLiPrima[[j]] <- rbind(idxInstLiPrima[[j]],c(indexInstTU[i],lab))
          }
      }
    }# end for each unlabeled example x in U


    # Estimate if adding Li' to Li improves the accuracy

    labeled <- list() # L u L'
    for (i in 1:N) {
      repeated <- intersect(idxInstLiPrima[[i]][,1], temp$idxInstLi[[i]][,1])
      if (length(repeated) != 0){#elimino instancias que ya esten en temp$idxInstLi
        filasToRemove <- sapply( X=repeated, FUN = function(r){
          which(idxInstLiPrima[[i]][,1] == r)
        })
        idxInstLiPrima[[i]] <- idxInstLiPrima[[i]][-filasToRemove,]
      }
      if (!is.null(idxInstLiPrima[[i]]))
        labeled[[i]] <- rbind(temp$idxInstLi[[i]], idxInstLiPrima[[i]])
      else
        labeled[[i]] <- temp$idxInstLi[[i]]
    }

    L <- sapply(1:N, function(i){
      confidenceInterval(m, temp, i, temp$idxInstLi[[i]])$L
    })
    q <- ep <- qp <- NULL
    for (i in 1:N) { # for each classifier
      sizeLi <- nrow(temp$idxInstLi[[i]])
      sizeLLP <- nrow(labeled[[i]])
      if (sizeLLP > sizeLi) { #hay instancias nuevas eb LiPrima
        q[i] <- sizeLi * (1 - 2*(e[i]/sizeLi))^2				# est. of error rate
        ep[i] <- (1 - mean(L[-i])) * nrow(as.matrix(idxInstLiPrima[[i]]))			# est. of new error rate
        qp[i] <- sizeLLP * (1 - 2*(e[i]+ep[i])/sizeLLP)^2		# if Li' added

        if (qp[i] > q[i]) {
          #cat("Add", sizeLLP - sizeLi,"instances to",class(BaseL[[i]]),"\n")
          temp$idxInstLi[[i]] <- labeled[[i]]
          e[i] <- e[i] + ep[i]
          changes <- TRUE
          # entrenar clasificador
          temp$models[[i]] <- runTrain(m$bclassifs[[i]], temp$dists.l[m$dist.use[i,]], inds = temp$idxInstLi[[i]][,1], y = temp$idxInstLi[[i]][,2])
        }
      }
    } # end for each classifier
    iter <- iter + 1
  } # End while

  # Store the model data
  m$models <- temp$models
  m$W <- sapply(1:N, function(i) confidenceInterval(m, temp, i, indexInstTL)$W)
  # determine labeled instances
  m$included.insts <- vector()
  for(i in 1:N){
    m$included.insts <- union(m$included.insts, temp$idxInstLi[[i]][,1])
  }
  # map indexes respect to m$included.insts
  m$indexes <- list()
  for(i in 1:N){
    m$indexes[[i]] <- vapply(temp$idxInstLi[[i]][,1], FUN = function(e){ which(e == m$included.insts)}, FUN.VALUE = 1)
  }
  if(! dist.is.matrix){
    # dist and included.insts are needs for compute the distances before predict
    # Save dist in the model.
    m$dist <- dist
    # Save the labeled instances in the model
    m$tr.insts <- x[m$included.insts, ]
  }
  class(m) <- "democratic"

  m
}

#' @title Model Predictions
#' @details For additional help see \code{\link{democratic}} examples.
#' @description Predicts the label of instances according to Democratic model.
#' @param object Democratic model object built with \code{\link{democratic}} function.
#' @param x A object that can be coerced as matrix.
#' Depending on how was the model built, \code{x} is interpreted as a matrix
#' with the distances between the unseen instances and the selected training instances,
#' or a matrix of instances.
#' @param ... Additional objects that can be coerced as matrix.
#' Depending on how was the model built, this objects are used or ignored.
#' This objects are interpreted as matrices of distances between the unseen instances
#' and the selected training instances.
#' @return Vector with labels of instances.
#' @export
#' @importFrom stats predict
predict.democratic <- function(object, x, ...){

  temp <- new.env(parent = emptyenv())
  temp$dists.l <- getDists(object, c(list(x), list(...)))

  pred <- internalClassify(object, temp, 1:nrow(x), length(object$labels.map), on.train = FALSE)$pred

  clase <- NULL
  N <- length(object$bclassifs)

  for (i in 1:nrow(x)) {#for each example x in U
    pertenece <- wz <- rep(0, length(object$labels.map))

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
    clase[i] <- which.max(countGj)
  }# end for

  # retornar

  restoreLabels(object, clase) # traduce las etiquetas a los valores originales
}

#' @title Compute the 95\% confidence interval of the classifier BaseL
#' @return A vector of the same length of indexInstTU containing the label of these instances.
#' @noRd
confidenceInterval <- function(m, temp, i, indexInstTL) {
  pred <- runPredict(m$bclassifs[[i]]$predClass,
                     temp$models[[i]], temp$dists.l[m$dist.use[i,]],
                     indexInstTL[,1], temp$idxInstLi[[i]][,1])
  W <- length(which(pred == indexInstTL[,2])) / nrow(indexInstTL) #accuracy
  L <- W - 1.96 * sqrt(W*(1-W) / nrow(indexInstTL)) # lowest point of the confidence interval
  list(L = L, W = W)
}

#' @title Classify unseen instances by majority voting
#' @param indexInstTU indices in the data set of the instances to be classified
#' @param ncls the number of ncls
#' @return A vector of the same length of indexInstTU containing the label of these instances.
#' @noRd
internalClassify <- function(m, temp, indexInstTU, ncls, on.train = TRUE) {
  N <- length(m$bclassifs) #number of classifiers
  lenU <- length(indexInstTU) #number of unlabeled instances
  pred <- matrix(nrow = N, ncol = lenU)

  for (i in 1:N){
    if(on.train){
      pred[i,] <- runPredict(m$bclassifs[[i]]$predClass,
                             temp$models[[i]], temp$dists.l[m$dist.use[i,]],
                             indexInstTU, temp$idxInstLi[[i]][,1])
    }else{
      pred[i,] <- runPredict(m$bclassifs[[i]]$predClass,
                             m$models[[i]], temp$dists.l[m$dist.use[i,]],
                             indexInstTU, m$indexes[[i]])
    }
  }

  lab <- vote(pred,ncls) # etiquetas votadas

  output <- list()
  output$pred <- pred #individual labels assigned for each classifier
  output$lab <- lab # voted labels for each instance

  output
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
