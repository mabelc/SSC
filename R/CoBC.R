#' @title Train the Co-bagging model
#' @description Builds and trains a model to predict the label of instances,
#' according to Co-bagging algorithm.
#' @param x A object that can be coerced as matrix.
#' This object have various interpretations depending on the value set in \code{dist} argument.
#' See \code{dist} argument.
#' @param y A vector with the labels of training instances. In this vector the unlabeled instances
#' are specified with the value \code{NA}.
#' @param bclassif Base classifier specification. Default is \code{bClassifOneNN()}. For defining new base classifiers
#' see \code{\link{bClassif}}.
#' @param dist Distance information. Valid options are:
#' \itemize{
#'    \item \code{"matrix"}: this string indicates that \code{x} is a distance matrix.
#'    \item \emph{string}: the name of a distance method available in \code{proxy} package.
#'    In this case \code{x} is interpreted as a matrix of instances.
#'    \item \emph{function}: a function defined by the user that computes the distance between two vectors.
#'    This function is called passing the vectors in the firsts two arguments. If the function have others arguments,
#'    those arguments must be have default values.
#'    In this case \code{x} is interpreted as a matrix of instances.
#' }
#' @param N The number of classifiers used as committee members. All these classifiers are defined using
#' the description provided by \code{bClassif}. Default is 3.
#' @param min.amount Minimum number of unlabeled instances to stop the training process. When the size
#' of unlabeled training instances reaches this number the self-labeling process is stopped. Default is
#' 0.3 * <number of unlabeled instances>.
#' @param u Number of unlabeled instances in the pool. Default is 100.
#' @param max.iter Maximum number of iterations to execute in the self-labeling process. Default is 50.
#' @return The trained model stored in a list with the following named values:
#' \describe{
#'   \item{models}{A list with the models (classifiers) trained. See argument \code{N} for details.}
#'   \item{bclassif}{The value of \code{bclassif} argument.}
#'   \item{dist}{The value of \code{dist} argument.
#'   This value is optionally returned when the argument \code{dist} is different from "matrix".}
#'   \item{included.insts}{The indexes of the training instances used.
#'   This set is the union of the indexes used to train each model in \code{models}.
#'   These indexes are relative to \code{x} argument.}
#'   \item{indexes}{A list of vectors. Each vector indicates the indexes of the instances
#'   used to train the corresponding model in \code{models}. For example, the instances used to train
#'   \code{models[1]} are \code{x[included.insts[indexes[1]], ]}.}
#'   \item{tr.insts}{The training instances corresponding to the indexes returned in \code{included.insts}.
#'    This value is optionally returned when the \code{dist} argument is different from "matrix".}
#'   \item{labels.map}{An internal map used for the labels.}
#' }
#' @references
#' Avrim Blum and Tom Mitchell.\cr
#' \emph{Combining labeled and unlabeled data with co-training.}\cr
#' In Eleventh Annual Conference on Computational Learning Theory, COLT’ 98, pages 92–100, New York, NY, USA, 1998. ACM.
#' ISBN 1-58113-057-0. doi: 10.1145/279943.279962.
#' @examples
#' # This example is part of CoBC demo.
#' # Use demo(CoBC) to see all the examples.
#'
#' ## Load Wine data set
#' data(wine)
#'
#' x <- wine[, -14] # instances without classes
#' y <- wine[, 14] # the classes
#' x <- scale(x) # scale the attributes
#'
#' ## Prepare data
#' set.seed(20)
#' # Use 50% of instances for training
#' tra.idx <- sample(x = length(y), size = ceiling(length(y) * 0.5))
#' xtrain <- x[tra.idx,] # training instances
#' ytrain <- y[tra.idx]  # classes of training instances
#' # Use 70% of train instances as unlabeled set
#' tra.na.idx <- sample(x = length(tra.idx), size = ceiling(length(tra.idx) * 0.7))
#' ytrain[tra.na.idx] <- NA # remove class information of unlabeled instances
#'
#' # Use the other 50% of instances for inductive testing
#' tst.idx <- setdiff(1:length(y), tra.idx)
#' xitest <- x[tst.idx,] # testing instances
#' yitest <- y[tst.idx] # classes of testing instances
#'
#' ## Example: Using the Euclidean distance in proxy package.
#' m <- coBC(xtrain, ytrain, dist = "Euclidean")
#' pred <- predict(m, xitest)
#' caret::confusionMatrix(table(pred, yitest))
#'
#' @export
coBC <- function(
  x, y,
  bclassif = bClassifOneNN(), dist = "matrix",
  N = 3,
  min.amount = ceiling(length(which(is.na(y))) * 0.3),
  u = 100, max.iter = 50
) {
  # Create the enviroment m for maintains the model data
  m <- new.env(parent = emptyenv())
  m$bclassif <- bclassif
  # Check common parameters
  dist.is.matrix <- NULL # initialized for checkCommonParams
  checkCommonParams(env = environment(), denv = m)
  # Check specifics parameters
  if(!(is.numeric(min.amount) && length(min.amount) == 1)) stop("min.amount must be a numeric value")
  if(!(min.amount > 0)) warning("min.amount is not a positive number")
  if(!(is.numeric(max.iter) && length(max.iter) == 1)) stop("max.iter must be a numeric value")
  if(!(max.iter > 0)) stop("max.iter must be a positive value")
  if(!is.numeric(u)) stop("'u' must be numeric")
  if(!is.numeric(N)) stop("'N' must be numeric")
  if(!(N > 0)) stop("'N' must be a positive number")

  #
  yy <- y[!is.na(y)]
  m <- normalizeLabels(m, yy)
  indexInstTL <- cbind(which(!is.na(y)), mapLabels(m, yy)) # indexes and clases of labeled instances
  indexInstTU <- which(is.na(y)) # indexes of unlabeled instances

  # cantidad de clases
  ncls <- length(m$labels.map)
  if(u < ncls) { stop("The pool size must be greater than the number of classes") }

  #estimar las cant por cada clase
  cantClass <- sapply(1:ncls, function(c) length(which(indexInstTL[,2] == c)))
  cantClass <- round(cantClass / min(cantClass)) # divido por el valor minimo
  totalPerIter <- sum(cantClass)

  m$L <- resample(indexInstTL = indexInstTL, N = N)
  #entrenar usando bagging
  m$models <- list()
  m$LO <- list()
  for (i in 1:N) {
    inds <- m$L[[i]][,1]
    m$models[[i]] <- runTrain(m$bclassif, m$dists.l, inds = inds, y = m$L[[i]][,2])
    m$LO[[i]] <- inds
  }

  iter <- 1
  m$HO <- m$models
  while ((length(indexInstTU) > min.amount) && (iter <= max.iter) ){ #para cada iteracion

    labeledPrima <- LPrima <- list()

    yes.break <- TRUE
    for(i in 1:N){#Para cada clasificador
      if(length(indexInstTU) > totalPerIter){#puedo satisfacer totalmente al clasificador i
        #selecciono aleatoriamente un pool de unlabeled instances
        unlabeledPrima <- sample(x = indexInstTU, size = min(u,length(indexInstTU)), replace = FALSE)
        #selecciono las instancias mas competentes
        labeledPrima[[i]] <- selectCompetentExamples(m, k =  i, unlabeledPrima, cantClass)
        #verificacion con el conjunto inicial
        classification <- H.classify(m, indexesQ = labeledPrima[[i]][,1], use.models = FALSE)
        indCoinciden <- which(classification == labeledPrima[[i]][,2])

        #adiciono en el conjunto de etiquetas del clasificador i las nuevas instancias
        LPrima[[i]] <- rbind(m$L[[i]],labeledPrima[[i]][indCoinciden,])
        #las elimino de indexInstTU
        indexInstTU <- setdiff(indexInstTU,labeledPrima[[i]][,1])
      } else {
        yes.break <- TRUE
        break
      }
    }#end para cada clasificador

    if(yes.break){
      end <- i - 1
      LPrima[i:N] <- m$L[i:N]
    } else {
      end <- N
    }
    m$L <- LPrima

    #reentreno el base learning con las nuevas instancias
    for (j in 1:end){
      m$models[[j]] <- runTrain(m$bclassif, m$dists.l, inds = m$L[[j]][,1], y = m$L[[j]][,2])
    }
    #print(paste("iter: ",iter))
    #print(paste("quedan en U:",length(indexInstTU)))
    iter <- iter + 1
  }#end del while principal

  # determine labeled instances
  m$included.insts <- vector()
  for(i in 1:N){
    m$included.insts <- union(m$included.insts, m$L[[i]][,1])
  }
  m$indexes <- list()
  # map indexes respect to m$included.insts
  for(i in 1:N){
    m$indexes[[i]] <- vapply(m$L[[i]][,1], FUN = function(e){ which(e == m$included.insts)}, FUN.VALUE = 1)
  }
  rm(list = c("L", "LO", "HO", "dists.l"), envir = m)
  class(m) <- "coBC"
  if(! dist.is.matrix){
    # dist and included.insts are needs for compute the distances before predict
    # Save dist in the model.
    m$dist <- dist
    # Save the labeled instances in the model
    m$tr.insts <- x[m$included.insts, ]
  }
  m
}

#' @title Model Predictions
#' @description Predicts the label of instances according to coBC model.
#' @details For additional help see \code{\link{coBC}} examples.
#' @param object coBC model object built with \code{\link{coBC}} function.
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
predict.coBC <- function(object, x, ...){

  object$dists.l <- getDists(object, c(list(x), list(...)))

  clase <- H.classify(object, 1:nrow(x))
  object$dists.l <- NULL

  # retornar
  restoreLabels(object, clase) # traduce las etiquetas a los valores originales
}


#' @title Compute the probability assigned by the committee H that xu belongs to class c
#' @param h.prob is the list containing the probability matrix  of each base classifier
#' @param u the unlabeled instance
#' @param c the class
#' @param classes The number of classes
#' @return The probability
#' @noRd
H.xu.wc <- function(h.prob,u,c,classes){
  N <- length(h.prob)
  num <- sum(sapply(X = 1:N, FUN = function(i){h.prob[[i]][u,c]}))
  den <- 0
  for(j in 1:classes){
    den <- den + sum(sapply(X = 1:N, FUN = function(i){h.prob[[i]][u,j]}))
  }
  num/den
}

#' @noRd
H.prob <- function(m, indexesQ, use.models = TRUE){

  h.prob <- lapply(X = 1:length(m$models), FUN = function(i) {
    if(use.models){
      runPredict(m$bclassif$predProb, m$models[[i]], m$dists.l, indexesQ, m$indexes[[i]])
    }else{
      runPredict(m$bclassif$predProb, m$HO[[i]], m$dists.l, indexesQ, m$LO[[i]])
    }
  })

  U <- length(indexesQ)
  #para cada instancia u estimo la probabilidad por clase que le asigna el ensemble
  ncls <- length(m$labels.map)
  H.pro <- matrix(nrow = U, ncol = ncls)

  for(ind in 1:U){
    H.pro[ind,] <- sapply(X = 1:ncls, FUN = function(i){
      H.xu.wc(h.prob=h.prob, u = ind, c = i, classes = ncls)})
  }

  H.pro
}

#' @noRd
H.classify <- function(m, indexesQ, use.models = TRUE){

  MP <- H.prob(m, indexesQ, use.models)

  U <- length(indexesQ)
  H.classes <- sapply(X = 1:U, FUN = function(i){which.max(MP[i,])})

  H.classes
}

#' @title Select the competent instances from the unlabeled pool
#' @param k is the k-th classifier of the committee
#' @param cantClass The number of instances per class
#' @return The competent instances selected per class
#' @noRd
selectCompetentExamples <- function(m, k, unlabeledPrima, cantClass){
  # obtain the committee for the classifier k
  X <- setdiff(1:length(m$models), k) # indexes of models excluding the index of model k
  # obtain the list of probabilities
  h.prob <- lapply(X = X, FUN = function(i) {
    runPredict(m$bclassif$predProb, m$models[[i]], m$dists.l, unlabeledPrima, m$L[[i]][,1]) })

  #para cada instancia u estimo la probabilidad por clase que le asigna el ensemble
  U <- length(unlabeledPrima)
  ncls <- length(m$labels.map)
  H.prob <- matrix(nrow = U, ncol = ncls)

  for(ind in 1:U){
    H.prob[ind,] <- sapply(X = 1:ncls, FUN = function(i){
      H.xu.wc(h.prob = h.prob, u = ind, c = i, classes = ncls) })
  }
  #calculo la clase estimada
  wpred <- sapply(X = 1:U, FUN = function(i){ which.max(H.prob[i,]) })

  LP <- selectInst(cantClass=cantClass, probabilities=H.prob)
  labeledPrima <- matrix(nrow = 0, ncol = 2)

  for (i in 1:length(LP)){
    if (length(LP[[i]]) != 0)
      for(j in 1:length(LP[[i]]))
      {
        labeledPrima <- rbind(labeledPrima, c(unlabeledPrima[LP[[i]][j]], i))
      }
  }
  labeledPrima
}
