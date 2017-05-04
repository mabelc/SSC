
# Note: democratic assumes that the classifiers provided are from different
# learning paradigms. In this examples we simulated this requirement by using two
# 1-NN classifiers with different distance measures instead of two different
# learning paradigms.

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

## Example
bclassifs <- list(bClassifOneNN(), bClassifOneNN())
dist <- list("Euclidean", "Manhattan")
dist.use <- matrix(
  data = c(
#   Euclidean Manhattan
    TRUE,     FALSE,    # First  1-NN
    FALSE,    TRUE      # Second 1-NN
  ),
  nrow = 2, byrow = TRUE
)
m <- democratic(x = xtrain, y = ytrain,
                bclassifs, dist, dist.use)
pred <- predict(m, xitest)
caret::confusionMatrix(table(pred, yitest))

## Example: Using distance matrices instead of the instances
dtEuclidean <- proxy::dist(x = xtrain, method = "Euclidean", by_rows = TRUE)
dtManhattan <- proxy::dist(x = xtrain, method = "Manhattan", by_rows = TRUE)
m2 <- democratic(x = list(dtEuclidean, dtManhattan), y = ytrain,
                bclassifs, dist = "matrix", dist.use = dist.use)

ditEuclidean <- proxy::dist(x = xitest, y = xtrain[m2$included.insts,],
                            method = "Euclidean", by_rows = TRUE)
ditManhattan <- proxy::dist(x = xitest, y = xtrain[m2$included.insts,],
                            method = "Manhattan", by_rows = TRUE)
pred <- predict(m2, ditEuclidean, ditManhattan)
caret::confusionMatrix(table(pred, yitest))
