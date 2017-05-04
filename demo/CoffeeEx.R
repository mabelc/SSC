library(ssc)

## Load Coffee data set
data(coffee)

x <- coffee[, -287] # instances without classes
y <- coffee[, 287] # the classes

## Prepare data
set.seed(1)
# Use 50% of instances for training
tra.idx <- sample(x = length(y), size = ceiling(length(y) * 0.5))
xtrain <- x[tra.idx,] # training instances
ytrain <- y[tra.idx]  # classes of training instances
# Use 70% of train instances as unlabeled set
tra.na.idx <- sample(x = length(tra.idx), size = ceiling(length(tra.idx) * 0.7))
ytrain[tra.na.idx] <- NA # remove class information of unlabeled instances

# Use the other 50% of instances for inductive testing
tst.idx <- setdiff(1:length(y), tra.idx)
xitest <- x[tst.idx,] # inductive testing instances
yitest <- y[tst.idx]  # classes of inductive testing instances
# Use the unlabeled examples for transductive testing
xttest <- x[tra.idx[tra.na.idx],] # transductive testing instances
yttest <- y[tra.idx[tra.na.idx]]  # classes of transductive testing instances

library(proxy) # Load the package proxy to use the function dist

dtrain <- dist(x = xtrain, method = "euclidean", by_rows = TRUE)
ditest <- dist(x = xitest, y = xtrain,
               method = "euclidean", by_rows = TRUE)
dttest <- dist(x = xttest, y = xtrain,
               method = "euclidean", by_rows = TRUE)

## Training with selfTrainings
# Using precomputed distances
m1 <- selfTraining(x = dtrain, y = ytrain)
# Using the instances and the Euclidean distance
m2 <- selfTraining(x = xtrain, y = ytrain, dist = "euclidean")
# Using the instances and the DTW pseudo distance
m3 <- selfTraining(x = xtrain, y = ytrain,
                   dist = function(x, y){
                     dtw::dtw(x, y, window.type = "sakoechiba",
                              window.size = 10)$distance
                   })

library(caret)
## Predicting using m1
# Compute distances between testing instances and training instances
# used to build the model. The testing distances are expected by rows.
# m1$included.insts are the indexes of instances used to build the model m1
d <- ditest[, m1$included.insts]
pred1 <- predict(m1, d)
confusionMatrix(table(pred1, yitest))$overall[1:2]

## Predicting using m2
pred2 <- predict(m2, xttest)
confusionMatrix(table(pred2, yttest))$overall[1:2]

## Training with democratic
# Note: democratic assumes that the classifiers provided are from different
# learning paradigms. In this example we simulated this requirement by using three
# 1-NN classifiers with different distance measures instead of three different
# learning paradigms.
bclassifs <- list(bClassifOneNN(), bClassifOneNN(), bClassifOneNN())
dist <- list("Euclidean",
             function(x, y){ dtw::dtw(x, y)$distance },
             function(x, y){ TSdist::EDRDistance(x, y, epsilon = 0.2) })
dist.use <- matrix(
  data = c(
    TRUE, FALSE, FALSE,
    FALSE, TRUE, FALSE,
    FALSE, FALSE, TRUE
  ),
  nrow = 3, byrow = TRUE
)
m4 <- democratic(x = xtrain, y = ytrain, bclassifs, dist, dist.use)
