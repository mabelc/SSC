#' @title Statistics calculation
#' @description Calculates the statistics:
#' Accuracy, Cohen's kappa and F-measure
#' @param predicted a vector with the predicted labels
#' @param real a vector with the real labels
#' @return a named list with the statistics values calculated
#' @export
statistics <- function(predicted, real) {
  results <- list()
  lvs = unique(c(real,predicted))
  truth <- factor(levels = lvs,real)
  pred <- factor(levels = lvs,predicted)
  xtab <- table(pred, truth)
  m <- caret::confusionMatrix(xtab)

  accuracy <- m$overall[1]
  kappa <- m$overall[2]
  fmeasure <- F_measureMC(xtab)

  results$kappa <- kappa
  results$accuracy <- accuracy
  results$fmeasure <- fmeasure
  results
}

#' @title F-measure of multi class classifier
#' @param confusionMatrix the matrix containing training instances
#' @return the F-measure value
#' @noRd
F_measureMC <- function(mConfusion){
  #media armonica de la precision y el recall
  #para cada clase calculo precision and recall
  p <- c()
  r <- c()

  for(c in 1:nrow(mConfusion)){
    mc <- list()
    mc$TP <- mConfusion[c,c]
    mc$FN <- sum(mConfusion[,c]) - mc$TP
    mc$FP <- sum(mConfusion[c,]) - mc$TP
    p[c] <- precision(mc)
    r[c] <- recall(mc)
  }

  #calculo macro-precision and macro-recall
  Mp <- mean(p)
  Mr <- mean(r)
  #calculo macro-FMeasure
  Mfmeasure <- 2/(1/Mp + 1/Mr)
  Mfmeasure
}

#' @title precision of classifier
#' @param confusionMatrix the matrix containing training instances
#' @return the precision value
#' @noRd
precision <- function(mConfusion){

  result <- mConfusion$TP/(mConfusion$TP + mConfusion$FP)
  result <- if (is.nan(result)) 1
  else result
}

#' @title recall of classifier
#' @param confusionMatrix the matrix containing training instances
#' @return the recall value
#' @noRd
recall <- function(mConfusion){

  result <- mConfusion$TP/(mConfusion$TP + mConfusion$FN)
  result <- if (is.nan(result)) 1
  else result

}

#' @export
getmode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#' @title Statistical mode
#' @noRd
statisticalMode <- function(a){
  my_mode = as.numeric(names(table(a))[which.max(table(a))])
  my_mode
}
