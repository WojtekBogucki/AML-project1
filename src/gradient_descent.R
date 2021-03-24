
sigmoid <- function(x){
  sigm <- 1/(1 + exp(-x))
  return(sigm)
}


init_theta <- function(n, type="uniform"){
  if(type=="uniform") theta <- runif(n)
  return(as.matrix(theta))
}

gradient_descent <- function(X,Y, epsilon  = 1e-15, max_iter = 100, lr = 0.01){
  X <- cbind(1, X)
  m <- nrow(X)
  n <- ncol(X)
  theta <- init_theta(n)
  for(i in 1:max_iter){
    g <- sigmoid(X %*% theta)
    cost <- -sum(Y*log(g + 1e-8) + (1-Y)*log(1-g + 1e-8))/m
    theta_old <- theta
    theta <- theta - lr*(t(X) %*% (g - Y)/m)
    print(paste0("Epoch: ", i, " Cost: ", round(cost,4)))
    if(all(abs(theta - theta_old) < epsilon)) break
  }
  structure(.Data = list(beta = theta), class = c("gd", "logreg", "model"))
}

predict.gd <- function(object, X, prob = FALSE, ...){
  X <- cbind(1, X)
  p <- sigmoid(X %*% object$beta)
  if(prob) return(p)
  else return(round(p))
}

sgd <- function(X,Y, epsilon = 1e-15, max_iter = 100, lr = 0.01){
  X <- cbind(1, X)
  n <- ncol(X)
  m <- nrow(X)
  theta <- init_theta(n)
  i <- 0
  stop_condition <- FALSE
  for(i in 1:max_iter){
    idx <- sample(1:m, m)
    x_new <- X[idx,]
    y_new <- Y[idx]
    cost <- 0
    theta_old <- theta
    for(j in 1:m){
      x <- x_new[j,]
      y <- y_new[j]
      g <- sigmoid(x %*% theta)
      cost <- cost - (y*log(g + 1e-8) + (1-y)*log(1-g + 1e-8))
      theta <- theta - lr*(x * c(g - y))
    }
    if(all(abs(theta - theta_old) < epsilon)) break
    print(paste0("Epoch: ", i, " Cost: ", round(cost/m,4)))
  }
  structure(.Data = list(beta = theta), class = c("gd", "logreg", "model"))
}




