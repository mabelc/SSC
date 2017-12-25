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

## Example: Training from a set of instances with 1-NN (knn3) as base classifier.
learnerB <- function(indexes, cls) 
  caret::knn3(x = xtrain[indexes, ], y = cls, k = 1)
predB <- function(model, indexes)  
  predict(model, xtrain[indexes, ]) 

# Train
set.seed(1)
md1 <- triTrainingBase(y = ytrain, learnerB, predB)

# Predict
pred <- list()
for(i in 1:3){
  pred[[i]] <- predict(md1$models[[i]], xitest, type = "class")
}
cls1 <- c()
for(i in 1:nrow(xitest)){
  a <- c(pred[[1]][i], pred[[2]][i], pred[[3]][i])
  cls1[i] <- ssc::getmode(a)
}
caret::confusionMatrix(table(cls1, yitest))

## Example: Training from a distance matrix with 1-NN (oneNN) as base classifier.
dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE))
learnerB <- function(indexes, cls) {
  m <- ssc::oneNN(y = cls)
  attr(m, "tra.idxs") <- indexes
  m
}

predB <- function(model, indexes)  {
  tra.idxs <- attr(model, "tra.idxs")
  d <- dtrain[indexes, tra.idxs]
  prob <- predict(model, d, type = "prob",  initial.value = 0) 
  prob
}

# Train
set.seed(1)
md2 <- triTrainingBase(y = ytrain, learnerB, predB)

# Predict
ditest <- proxy::dist(x = xitest, y = xtrain[md2$included.insts,],
                      method = "euclidean", by_rows = TRUE)

pred <- list()
for(i in 1:3){
  m <- md2$models[[i]]
  D <- ditest[, md2$indexes[[i]]]
  pred[[i]] <- predict(m, D, type = "class")
}
cls2 <- c()
for(i in 1:nrow(xitest)){
  a <- c(pred[[1]][i], pred[[2]][i], pred[[3]][i])
  cls2[i] <- ssc::getmode(a)
}
caret::confusionMatrix(table(cls2, yitest))
