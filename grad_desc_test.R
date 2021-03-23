source("src/measure.R")
source("src/gradient_descent.R")
load("data/jain_train.Rda")
load("data/jain_test.Rda")
jain_train

X <- as.matrix(jain_train[,-ncol(jain_train)])
Y <- as.matrix(jain_train[,ncol(jain_train)])

X_test <- as.matrix(jain_test[,-ncol(jain_test)])
Y_test <- as.matrix(jain_test[,ncol(jain_test)])

model <- gradient_descent(X, Y, 200, 0.05)
y_hat <- predict.gradient_descent(model, X_test)
model2 <- sgd(X, Y, 100, 0.01)
y_hat2 <- predict.gradient_descent(model2, X_test)

measure(y_hat, Y_test)
measure(y_hat2, Y_test)
