
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

## Example: Training from a set of instances with 1-NN as base classifier.
learners <- list(caret::knn3, kernlab::ksvm)
learners.pars = list(list(k = 1), list(prob.model = TRUE))
preds = list("predict", kernlab::predict)
preds.pars = list(NULL, list(type = "probabilities"))

set.seed(1)
m1 <- democratic(x = xtrain, y = ytrain, 
                 learners, learners.pars,
                 preds, preds.pars)
pred1 <- predict(m1, xitest)
caret::confusionMatrix(table(pred1, yitest))



