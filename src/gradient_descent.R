
sigmoid <- function(x){
  sigm <- 1/(1 + exp(-x))
  return(sigm)
}


init_theta <- function(n, type="uniform"){
  if(type=="uniform") theta <- runif(n)
  return(as.matrix(theta))
}

gradient_descent <- function(X,Y, epsilon  = 1e-7, max_iter = 100, lr = 0.01){
  X <- cbind(1, X)
  m <- nrow(X)
  n <- ncol(X)
  theta <- as.matrix(rep(0, length.out = n))
  g <- sigmoid(X %*% theta)
  all_costs <- c(-sum(Y*log(g + 1e-8) + (1-Y)*log(1-g + 1e-8))/m)
  for(i in 1:max_iter){
    theta <- theta - lr*(t(X) %*% (g - Y)/m)
    g <- sigmoid(X %*% theta)
    all_costs[i+1] <- -sum(Y*log(g + 1e-8) + (1-Y)*log(1-g + 1e-8))/m
    print(paste0("Epoch: ", i, " Cost: ", round(all_costs[i+1],4)))
    if(all(abs(all_costs[i+1] - all_costs[i]) < epsilon)) break
  }
  structure(.Data = list(beta = theta, costs = all_costs, iters = i), class = c("gd", "logreg", "model"))
}

predict.gd <- function(object, X, prob = FALSE, ...){
  X <- cbind(1, X)
  p <- sigmoid(X %*% object$beta)
  if(prob) return(p)
  else return(round(p))
}

sgd <- function(X,Y, epsilon = 1e-7, max_iter = 100, lr = 0.01){
  X <- cbind(1, X)
  n <- ncol(X)
  m <- nrow(X)
  theta <- as.matrix(rep(0, length.out = n))
  g <- sigmoid(X %*% theta)
  all_costs <- c(-sum(Y*log(g + 1e-8) + (1-Y)*log(1-g + 1e-8))/m)
  for(i in 1:max_iter){
    idx <- sample(1:m, m)
    x_new <- X[idx,]
    y_new <- Y[idx]
    for(j in 1:m){
      x <- x_new[j,]
      y <- y_new[j]
      g <- sigmoid(x %*% theta)
      theta <- theta - lr*(x * c(g - y))
    }
    g <- sigmoid(x_new %*% theta)
    all_costs[i+1] <- -mean(y_new*log(g + 1e-8) + (1-y_new)*log(1-g + 1e-8))
    print(paste0("Epoch: ", i, " Cost: ", round(all_costs[i+1],4)))
    if(all(abs(all_costs[i+1] - all_costs[i]) < epsilon)) break
  }
  structure(.Data = list(beta = theta, costs = all_costs, iters = i), class = c("gd", "logreg", "model"))
}




