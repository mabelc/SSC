
library(ssc)

## Load Wine data set
data(wine)

x <- wine[, -14] # instances without classes
y <- wine[, 14] # the classes
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

## Example: Using the Euclidean distance in proxy package.
m <- selfTraining(xtrain, ytrain, dist = "Euclidean")
pred <- predict(m, xitest)
caret::confusionMatrix(table(pred, yitest))

## Example: Using a defined distance function
distFun <- function(x, y){
  proxy::dist(x, y, method = "Minkowski", p = 3, by_rows = FALSE)
}
m2 <- selfTraining(xtrain, ytrain, dist = distFun)
pred2 <- predict(m2, xitest)
caret::confusionMatrix(table(pred2, yitest))

## Example: Using distance matrices instead of the instances
# Compute distances between training instances
dtrain <- proxy::dist(x = xtrain, method = "euclidean", by_rows = TRUE)
m3 <- selfTraining(dtrain, ytrain)
# Compute distances between testing instances and training instances
# used to build the model. The testing distances are expected by rows.
# m3$included.insts - indexes of instances used to build the model m3
ditest <- proxy::dist(x = xitest, y = xtrain[m3$included.insts,],
                      method = "euclidean", by_rows = TRUE)
pred3 <- predict(m3, ditest)
caret::confusionMatrix(table(pred3, yitest))
