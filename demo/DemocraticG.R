library(ssc)

## Load Wine data set
data(wine)

cls <- which(colnames(wine) == "Wine")
x <- wine[, -cls] # instances without classes
y <- wine[, cls] # the classes
x <- scale(x) # scale the attributes

## Prepare data
set.seed(20)
# Use 50% of instances for training
tra.idx <- sample(x = length(y), size = ceiling(length(y) * 0.5))
xtrain <- x[tra.idx,] # training instances
ytrain <- y[tra.idx]  # classes of training instances
# Use 70% of train instances as unlabeled set
tra.na.idx <- sample(x = length(tra.idx), size = ceiling(length(tra.idx) * 0.7))
ytrain[tra.na.idx] <- NA # remove class information of unlabeled instances

# Use the other 50% of instances for inductive testing
tst.idx <- setdiff(1:length(y), tra.idx)
xitest <- x[tst.idx,] # testing instances
yitest <- y[tst.idx] # classes of testing instances

## Example A: 
# Training from a set of instances with 
# 1-NN and C-svc (SVM) as base classifiers.

### Define knn base classifier using knn3 from caret package
library(caret)
# learner function
knn <- function(indexes, cls) {
  knn3(x = xtrain[indexes, ], y = cls, k = 1)
}
# function to predict probabilities
knn.prob <- function(model, indexes) {
  predict(model, xtrain[indexes, ])
}

### Define svm base classifier using ksvm from kernlab package
library(kernlab)
library(proxy)
# learner function
svm <- function(indexes, cls) {
  rbf <- function(x, y) {
    sigma <- 0.048
    d <- dist(x, y, method = "Euclidean", by_rows = FALSE)
    exp(-sigma *  d * d)
  }
  class(rbf) <- "kernel"
  ksvm(x = xtrain[indexes, ], y = cls, scaled = FALSE,
       type = "C-svc",  C = 1,
       kernel = rbf, prob.model = TRUE)
}
# function to predict probabilities
svm.prob <- function(model, indexes) {
  predict(model, xtrain[indexes, ], type = "probabilities")
}

### Train
m1 <- democraticG(y = ytrain, 
                  gen.learners = list(knn, svm),
                  gen.preds = list(knn.prob, svm.prob))
### Predict
# predict labels using each classifier
m1.pred1 <- predict(m1$model[[1]], xitest, type = "class")
m1.pred2 <- predict(m1$model[[2]], xitest)
# combine predictions
m1.pred <- list(m1.pred1, m1.pred2)
cls1 <- democraticCombine(m1.pred, m1$W, m1$classes)
table(cls1, yitest)

## Example B: 
# Training from a distance matrix and a kernel matrix with 
# 1-NN and C-svc (SVM) as base classifiers.

### Define knn2 base classifier using oneNN from ssc package
library(ssc)
# Compute distance matrix D
# D is used in knn2.prob
D <- as.matrix(dist(x = xtrain, method = "euclidean", by_rows = TRUE))
# learner function
knn2 <- function(indexes, cls) {
  model <- oneNN(y = cls)
  attr(model, "tra.idxs") <- indexes
  model
}
# function to predict probabilities
knn2.prob <- function(model, indexes)  {
  tra.idxs <- attr(model, "tra.idxs")
  predict(model, D[indexes, tra.idxs], distance.weighting = "none")
}

### Define svm2 base classifier using ksvm from kernlab package
library(kernlab)

# Compute kernel matrix K
# K is used in svm2 and svm2.prob functions
sigma <- 0.048
K <- exp(- sigma * D * D)

# learner function
svm2 <- function(indexes, cls) {
  model <- ksvm(K[indexes, indexes], y = cls, 
                type = "C-svc", C = 1,
                kernel = "matrix", 
                prob.model = TRUE)
  attr(model, "tra.idxs") <- indexes
  model
}
# function to predict probabilities
svm2.prob <- function(model, indexes)  {
  tra.idxs <- attr(model, "tra.idxs")
  sv.idxs <- tra.idxs[SVindex(model)]
  predict(model, 
          as.kernelMatrix(K[indexes, sv.idxs]),
          type = "probabilities") 
}

### Train
m2 <- democraticG(y = ytrain, 
                  gen.learners = list(knn2, svm2),
                  gen.preds = list(knn2.prob, svm2.prob))

### Predict
# Compute distance matrix Ditest
Ditest <- dist(x = xitest, y = xtrain[m2$model.index[[1]],],
               method = "euclidean", by_rows = TRUE)
# predict using classifier 1
m2.pred1 <- predict(m2$model[[1]], Ditest, type = "class")

# Compute kernel matrix Kitest
sv.idxs <- SVindex(m2$model[[2]])
sv.idxs <- m2$model.index.map[[2]][sv.idxs]
Kitest <- Ditest[, sv.idxs]
Kitest <- exp(- sigma * Kitest * Kitest)
Kitest <- as.kernelMatrix(Kitest)
# predict using classifier 2
m2.pred2 <- predict(m2$model[[2]], Kitest)

# Combine predictions
m2.pred <- list(m2.pred1, m2.pred2)
cls2 <- democraticCombine(m2.pred, m2$W, m2$classes)
table(cls2, yitest)


