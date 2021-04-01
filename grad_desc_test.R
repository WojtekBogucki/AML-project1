source("src/measure.R")
source("src/gradient_descent.R")
source("src/iwls.R")
source("src/util.R")
source("src/optim_logreg.R")
library(microbenchmark)
library(dplyr)
# Jain --------------------------------------------------------------------
load("data/jain_train.Rda")
load("data/jain_test.Rda")
jain_train

X <- as.matrix(jain_train[,-ncol(jain_train)])
Y <- as.matrix(jain_train[,ncol(jain_train)])

X_test <- as.matrix(jain_test[,-ncol(jain_test)])
Y_test <- as.matrix(jain_test[,ncol(jain_test)])

max_iter <- 1000
epsilon <- 1e-4
lr <- 0.01

model <- gradient_descent(X, Y, epsilon, max_iter, 0.75)
y_hat <- predict(model, X_test)
model2 <- sgd(X, Y, epsilon , max_iter, lr)
y_hat2 <- predict(model2, X_test)
model3 <- iwls(X, Y, epsilon, max_iter)
y_hat3 <- predict(model3, X_test)
model4 <- optim_logreg(X, Y, method = "CG", epsilon, max_iter)
y_hat4 <- predict(model4, X_test)
model5 <- optim_logreg(X, Y, method = "BFGS", epsilon, max_iter)
y_hat5 <- predict(model5, X_test)

micro1 <- microbenchmark(gd = gradient_descent(X, Y, epsilon, max_iter, 0.75),
                        sgd = sgd(X, Y, epsilon , max_iter, lr),
                        iwls = iwls(X, Y, epsilon, max_iter),
                        cg = optim_logreg(X, Y, method = "CG", epsilon, max_iter),
                        bfgs = optim_logreg(X, Y, method = "BFGS", epsilon, max_iter),
               times = 10, unit='ms')
time_stats1 <- cbind(summary(micro1)[, c(1,4)], iters=sapply(list(model, model2, model3, model4, model5), function(m) m$iters))
time_stats1 <- time_stats1 %>% mutate(ms_per_iter=mean/iters) %>% rename(mean_ms = mean)
time_stats1
m1 <- measure(y_hat, Y_test)
measure(y_hat2, Y_test)
measure(y_hat3, Y_test)
plot(model)
plot(model2)
plot(model3)
g1 <- compare_models(list(model,model2, model3, model4, model5), labels = c("GD", "SGD", "IWLS", "CG", "BFGS"))
g1
# Spambase ----------------------------------------------------------------
load("data/spambase_train.Rda")
load("data/spambase_test.Rda")
head(spambase_train)
X <- as.matrix(spambase_train[,-ncol(spambase_train)])
Y <- as.matrix(spambase_train[,ncol(spambase_train)])

X_test <- as.matrix(spambase_test[,-ncol(spambase_test)])
Y_test <- as.matrix(spambase_test[,ncol(spambase_test)])

max_iter <- 500
epsilon <- 1e-5
lr <- 0.01

model <- gradient_descent(X, Y, epsilon, max_iter, 0.75)
y_hat <- predict(model, X_test)
model2 <- sgd(X, Y, epsilon , max_iter, lr)
y_hat2 <- predict(model2, X_test)
model3 <- iwls(X, Y, epsilon, max_iter)
y_hat3 <- predict(model3, X_test)
model4 <- optim_logreg(X, Y, method = "CG", epsilon)
y_hat4 <- predict(model4, X_test)
model5 <- optim_logreg(X, Y, method = "BFGS", epsilon, max_iter)
y_hat5 <- predict(model5, X_test)

micro2 <- microbenchmark(gd = gradient_descent(X, Y, epsilon, max_iter, 0.75),
                         sgd = sgd(X, Y, epsilon , max_iter, lr),
                         iwls = iwls(X, Y, epsilon, max_iter),
                         cg = optim_logreg(X, Y, method = "CG", epsilon, max_iter),
                         bfgs = optim_logreg(X, Y, method = "BFGS", epsilon, max_iter),
                         times = 10, unit='s')
time_stats2 <- cbind(summary(micro2)[, c(1,4)], iters=sapply(list(model, model2, model3, model4, model5), function(m) m$iters))
time_stats2 <- time_stats2 %>% mutate(ms_per_iter=1000*mean/iters) %>% rename(mean_s = mean)
time_stats2
cbind(time_stats1, time_stats2[,-1])

measure(y_hat, Y_test)
measure(y_hat2, Y_test)
measure(y_hat3, Y_test)
measure(y_hat4, Y_test)
plot(model)
plot(model2)
plot(model3)
plot(model4)

models <- list()
lr <- c(0.05, 0.01, 0.005, 0.001)
for (i in 1:4){
  models[[2*i-1]] <- gradient_descent(X, Y, epsilon, max_iter, lr[i])
  models[[2*i]] <- sgd(X, Y, epsilon , max_iter, lr[i])
}
compare_models(models, labels = paste(rep(c("GD", "SGD"), 4),"lr =", rep(lr, each=2)))
# mammography -------------------------------------------------------------
load("data/mammography_train.Rda")
load("data/mammography_test.Rda")
head(mammography_train)

X <- as.matrix(mammography_train[,-ncol(mammography_train)])
Y <- as.matrix(mammography_train[,ncol(mammography_train)])

X_test <- as.matrix(mammography_test[,-ncol(mammography_test)])
Y_test <- as.matrix(mammography_test[,ncol(mammography_test)])

max_iter <- 100
epsilon <- 1e-5



model <- gradient_descent(X, Y, epsilon, max_iter, 0.75)
y_hat <- predict(model, X_test)
model2 <- sgd(X, Y, epsilon, max_iter, 0.01)
y_hat2 <- predict(model2, X_test)
model3 <- iwls(X, Y, epsilon, max_iter)
y_hat3 <- predict(model3, X_test)
model4 <- optim_logreg(X, Y, method = "CG", epsilon, max_iter)
y_hat4 <- predict(model4, X_test)
model5 <- optim_logreg(X, Y, method = "BFGS", epsilon, max_iter)
y_hat5 <- predict(model5, X_test)

measure(y_hat, Y_test)
measure(y_hat2, Y_test)
measure(y_hat3, Y_test)
plot(model)
plot(model2)
plot(model3)

table(y_hat, Y_test)
table(y_hat2, Y_test)


# Skin segmentation -------------------------------------------------------
load("data/skin_seg_train.Rda")
load("data/skin_seg_test.Rda")

X <- as.matrix(skin_seg_train[,-ncol(skin_seg_train)])
Y <- as.matrix(skin_seg_train[,ncol(skin_seg_train)])

X_test <- as.matrix(skin_seg_test[,-ncol(skin_seg_test)])
Y_test <- as.matrix(skin_seg_test[,ncol(skin_seg_test)])

max_iter <- 100
epsilon <- 1e-5



model <- gradient_descent(X, Y, epsilon, max_iter, 0.75)
y_hat <- predict(model, X_test)
model2 <- sgd(X, Y, epsilon, max_iter, 0.005)
y_hat2 <- predict(model2, X_test)
model3 <- iwls(X, Y, epsilon, max_iter)
y_hat3 <- predict(model3, X_test)
model4 <- optim_logreg(X, Y, method = "CG", epsilon, max_iter)
y_hat4 <- predict(model4, X_test)
model5 <- optim_logreg(X, Y, method = "BFGS", epsilon, max_iter)
y_hat5 <- predict(model5, X_test)

micro3 <- microbenchmark(gd = gradient_descent(X, Y, epsilon, max_iter, 0.75),
                         sgd = sgd(X, Y, epsilon , max_iter, lr),
                         iwls = iwls(X, Y, epsilon, max_iter),
                         cg = optim_logreg(X, Y, method = "CG", epsilon, max_iter),
                         bfgs = optim_logreg(X, Y, method = "BFGS", epsilon, max_iter),
                         times = 2, unit='s')
time_stats3 <- cbind(summary(micro3)[, c(1,4)], iters=sapply(list(model, model2, model3, model4, model5), function(m) m$iters))
time_stats3 <- time_stats3 %>% mutate(ms_per_iter=1000*mean/iters) %>% rename(mean_s = mean)
time_stats3

measure(y_hat, Y_test)
measure(y_hat2, Y_test)
measure(y_hat3, Y_test)
plot(model)
plot(model2)
plot(model3)
plot(model4)
plot(model5)

table(y_hat, Y_test)
table(y_hat2, Y_test)

# Occupancy ---------------------------------------------------------------

load("data/occupancy_train.Rda")
load("data/occupancy_test.Rda")

X <- as.matrix(occupancy_train[,-ncol(occupancy_train)])
Y <- as.matrix(occupancy_train[,ncol(occupancy_train)])

X_test <- as.matrix(occupancy_test[,-ncol(occupancy_test)])
Y_test <- as.matrix(occupancy_test[,ncol(occupancy_test)])

model <- gradient_descent(X, Y, 1e-4, 1000, 0.01)
y_hat <- predict(model, X_test)
model2 <- sgd(X, Y, 1e-4, 20, 0.02)
y_hat2 <- predict(model2, X_test)
model3 <- iwls(X, Y)
y_hat3 <- predict(model3, X_test)

measure(y_hat, Y_test)
measure(y_hat2, Y_test)
measure(y_hat3, Y_test)
plot(model)
plot(model2)
plot(model3)

table(y_hat, Y_test)
table(y_hat2, Y_test)


