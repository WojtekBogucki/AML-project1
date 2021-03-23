load("data/jain_train.Rda")
jain_split_train

X <- as.matrix(jain_split_train[,-ncol(jain_split_train)])


Y <- jain_split_train[,ncol(jain_split_train)]
levels(Y) <- c(0,1)
Y <- as.matrix(as.numeric(as.vector(Y)))
Y

sigmoid <- function(x){
  sigm <- 1/(1 + exp(-x))
  return(sigm)
}

cost_fun <- function(m, g, Y){
  cost <- -sum(Y*log(g + 1e-8) + (1-Y)*log(1-g + 1e-8))/m
  return(cost)
}

update_theta <- function(X,Y, theta, lr, m, g){
  theta <- theta - lr*(t(X) %*% (g - Y)/m)
  return(theta)
}

init_theta <- function(n, type="uniform"){
  if(type=="uniform") theta <- runif(n)
  return(as.matrix(theta))
}

gradient_descent <- function(X,Y, n_iter, lr){
  m <- nrow(X)
  X <- cbind(rep(1, m), X)
  n <- ncol(X)
  theta <- init_theta(n)
  for(i in 1:n_iter){
    g <- sigmoid(X %*% theta)
    cost <- -sum(Y*log(g + 1e-8) + (1-Y)*log(1-g + 1e-8))/m
    theta <- theta - lr*(t(X) %*% (g - Y)/m)
    print(paste0("Cost: ", round(cost,4)))
  }
  return(round(sigmoid(X %*% theta)))
}

stoch_gradient_descent <- function(X,Y, n_iter, lr){
  m <- nrow(X)
  X <- cbind(rep(1, m), X)
  n <- ncol(X)
  theta <- init_theta(n)
  for(i in 1:n_iter){
    idx <- sample(1:m, m)
    x_new <- X[idx,]
    y_new <- Y[idx]
    cost <- 0
    for(i in 1:m){
      x <- x_new[i,]
      y <- y_new[i]
      g <- sigmoid(x %*% theta)
      cost <- cost - (y*log(g + 1e-8) + (1-y)*log(1-g + 1e-8))
      theta <- theta - lr*(x * c(g - y))
    }
    print(paste0("Cost: ", round(cost/m,4)))
  }
  return(round(sigmoid(X %*% theta)))
}

y_hat <- gradient_descent(X, Y, 1000, 0.02)
y_hat2 <- stoch_gradient_descent(X, Y, 100, 0.005)
source("src/measure.R")
measure(y_hat, Y)
measure(y_hat2, Y)
