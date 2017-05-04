
#' @title Compute matrices of distances
#' @param x a matrix of instances
#' @param y a matrix of instances
#' Should only be defined for calculation of distance matrices between
#' two different databases so default value is NULL.
#' @param dist a list with the distance methods specifications
#' @return a list with the computed matrices of distances
#' @noRd
computeDistances <- function(x, y = NULL, dist){
  r <- list()
  for(i in 1:length(dist)){
    method <- dist[[i]]
    r[[i]] <- as.matrix(
      proxy::dist(x = x, y = y, method = method,
                  diag = TRUE, upper = TRUE, by_rows = TRUE)
    )
  }
  return(r)
}

#' @title  Check Distances List
#' @param nlabels number of labels
#' @param env environment that contains the list of distances matrices
#' @param lname the name of the list in the environment
#' @details In the case of an error, this function throws an exception.
#' @noRd
checkDistsList <- function(nlabels, env, lname = "dists.l"){
  for(i in 1:length(env[[lname]])){
    if(! is.matrix(env[[lname]][[i]])) {
      stop(sprintf("The var %s is not a matrix.", names(env[[lname]])[i]))
    }
    if(nrow(env[[lname]][[i]]) != ncol(env[[lname]][[i]])) {
      stop(sprintf("The number of rows and columns of %s is different. Expected a square matrix.", names(env[[lname]])[i]))
    }
    if( nrow(env[[lname]][[i]]) != nlabels ) {
      stop(sprintf("The rows number of '%s' is different to the length of 'y'.", names(env[[lname]])[i]))
    }
  }
  names(env[[lname]]) <- NULL # para evitar problemas de nombre al ejecutar train, ej: el nombre del primer parametro de train no tiene que ser dists
}

#' @title Check Common Params Of Semi-Supervised methods
#' @param env environment that contains the parameters to check
#' @param denv environment to use for save the list with the matrices of distances
#' @details In the case of an error, this function throws an exception.
#' @noRd
checkCommonParams <- function(env, denv = env){
  # coerce x to matrix
  if(is.list(env$x)){
    for(i in 1:length(env$x)){
      if(! is.matrix(env$x[[i]]) ||
         # dist is the class of an object returned by proxy::dist
         class(env$x[[i]]) != "dist"){
        env$x[[i]] <- as.matrix(env$x[[i]])
      }
    }
  }else{
    env$x <- as.matrix(env$x)
  }
  # check bclassif class
  # bclassif not exists in the snnrce function
  if( exists(x = "bclassif", envir = env, inherits = FALSE) ){
    if (class(env$bclassif) != "bClassif"){
      stop("The 'bclassif' value must be built using the 'bClassif' function.")
    }
  }
  # check y
  if(! ((is.vector(env$y) && ! is.list(env$y)) |
        (is.array(env$y) && !is.matrix(env$y)) |
        is.factor(env$y))){
    stop("'y' must be a non generic vector")
  }
  # check the existence of more than one instance for each class
  if(any(summary(factor(env$y)) < 2)){
    stop("Expected more than one instance for each class. See the 'y' argument.")
  }
  # check dist
  if(! is.list(env$dist)){
    env$dist <- list(env$dist)
  }
  for (e in env$dist){
    if(!( is.character(e) | is.function(e) )){
      stop("All elements of 'dist' must be a string or a function.")
    }
  }
  # check consistency between the number of instances and the number of classes
  if( !is.list(env$x) ){  # can occur for democratic
    if( nrow(env$x) != length(env$y) ){
      stop("The rows number of 'x' must be equal to the length of 'y'")
    }
  }
  # determine how interpret x, as a matrix of distances or as a matrix of instances
  env$dist.is.matrix <-
    is.vector(env$dist) && length(env$dist) == 1 && env$dist == "matrix"
  # create a list with distance matrices
  if(env$dist.is.matrix){
    denv$dists.l <- if(is.list(env$x)){
      env$x # can occur for democratic
    }else{
      list(env$x)
    }
    # check matrices of distances in x
    checkDistsList(length(env$y), denv)
  }else{ # computes distances
    denv$dists.l <- computeDistances(env$x, dist = env$dist)
  }
}

#' @title For use in predict functions
#' @param object the model
#' @param input a list with the arguments passed to predict function with the
#' exception of the model argument
#' @description Check the input, compute distances if necessary and
#' return a list with the distance matrices (or dist objects)
#' @return A list with distance matrices (or dist objects)
#' @noRd
getDists <- function(object, input){
  if(is.null(object[["dist"]])){ # input store the precalculated distances
    for(i in 1:length(input)){
      if(! is.matrix(input[[i]]) ||
         # dist is the class of an object returned by proxy::dist
         class(input[[i]]) != "dist"){
        input[[i]] <- as.matrix(input[[i]])
      }
    }
    input
  }else{  # input store the instances
    if(! is.matrix(input[[1]])){
      input[[1]] <- as.matrix(input[[1]])
    }
    computeDistances(x = input[[1]], y = object$tr.insts, dist = object$dist)
  }
}
