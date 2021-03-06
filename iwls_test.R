source("src/measure.R")
source("src/iwls.R")

# Jain --------------------------------------------------------------------
load("data/jain_train.Rda")
load("data/jain_test.Rda")
jain_train

X <- as.matrix(jain_train[,-ncol(jain_train)])
Y <- as.matrix(jain_train[,ncol(jain_train)])

X_test <- as.matrix(jain_test[,-ncol(jain_test)])
Y_test <- as.matrix(jain_test[,ncol(jain_test)])

model <- iwls(X, Y)
y_hat <- predict(model, X_test)

measure(y_hat, Y_test)

# Spambase ----------------------------------------------------------------
load("data/spambase_train.Rda")
load("data/spambase_test.Rda")

X <- as.matrix(spambase_train[,-ncol(spambase_train)])
Y <- as.matrix(spambase_train[,ncol(spambase_train)])

X_test <- as.matrix(spambase_test[,-ncol(spambase_test)])
Y_test <- as.matrix(spambase_test[,ncol(spambase_test)])

model <- iwls(X, Y)
y_hat <- predict(model, X_test)

measure(y_hat, Y_test)


# mammography -------------------------------------------------------------
load("data/mammography_train.Rda")
load("data/mammography_test.Rda")

X <- as.matrix(mammography_train[,-ncol(mammography_train)])
Y <- as.matrix(mammography_train[,ncol(mammography_train)])

X_test <- as.matrix(mammography_test[,-ncol(mammography_test)])
Y_test <- as.matrix(mammography_test[,ncol(mammography_test)])

model <- iwls(X, Y)
y_hat <- predict(model, X_test)

table(y_hat, Y_test)

measure(y_hat, Y_test)

# Skin segmentation -------------------------------------------------------
load("data/skin_seg_train.Rda")
load("data/skin_seg_test.Rda")

X <- as.matrix(skin_seg_train[,-ncol(skin_seg_train)])
Y <- as.matrix(skin_seg_train[,ncol(skin_seg_train)])

X_test <- as.matrix(skin_seg_test[,-ncol(skin_seg_test)])
Y_test <- as.matrix(skin_seg_test[,ncol(skin_seg_test)])

model <- iwls(X, Y)
y_hat <- predict(model, X_test)

table(y_hat, Y_test)

measure(y_hat, Y_test)

# Occupancy ---------------------------------------------------------------

load("data/occupancy_train.Rda")
load("data/occupancy_test.Rda")

X <- as.matrix(occupancy_train[,-ncol(occupancy_train)])
Y <- as.matrix(occupancy_train[,ncol(occupancy_train)])

X_test <- as.matrix(occupancy_test[,-ncol(occupancy_test)])
Y_test <- as.matrix(occupancy_test[,ncol(occupancy_test)])

model <- iwls(X, Y)
y_hat <- predict(model, X_test)

table(y_hat, Y_test)

measure(y_hat, Y_test)

