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
learnerB1 <- function(indexes, cls) 
  caret::knn3(x = xtrain[indexes, ], y = cls, k = 1)
predB1 <- function(model, indexes)  
  predict(model, xtrain[indexes, ]) 

set.seed(1)
md1 <- coBCG(y = ytrain, learnerB1, predB1)

# Predict probabilities per instances using each model
h.prob <- lapply(
  X = md1$model, 
  FUN = function(m) predict(m, xitest)
)
# Combine probability matrices
prob <- coBCCombine(h.prob, md1$classes)
# Get class per instance
cls1 <- md1$classes[apply(prob, 1, which.max)]
table(cls1, yitest)

## Example: Training from a distance matrix with 1-NN (oneNN) as base classifier.
dtrain <- as.matrix(proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE))
learnerB2 <- function(indexes, cls) {
  m <- ssc::oneNN(y = cls)
  attr(m, "tra.idxs") <- indexes
  m
}

predB2 <- function(model, indexes)  {
  tra.idxs <- attr(model, "tra.idxs")
  d <- dtrain[indexes, tra.idxs]
  prob <- predict(model, d, type = "prob",  initial.value = 0) 
  prob
}

set.seed(1)
md2 <- coBCG(y = ytrain, learnerB2, predB2)

# Predict probabilities per instances using each model
ditest <- proxy::dist(x = xitest, y = xtrain[md2$instances.index,],
                      method = "euclidean", by_rows = TRUE)

h.prob <- list()
ninstances <- nrow(dtrain)
for(i in 1:length(md2$model)){
  m <- md2$model[[i]]
  D <- ditest[, md2$model.index.map[[i]]]
  h.prob[[i]] <- predict(m, D, type = "prob",  initial.value = 0)
}
# Combine probability matrices
prob <- coBCCombine(h.prob, md2$classes)
# Get class per instance
cls2 <- md2$classes[apply(prob, 1, which.max)]
table(cls2, yitest)

