iwls <- function(X, y, epsilon = 1e-15){
  X <- cbind(1, X)
  y0 <- mean(y)
  beta <- as.matrix(rep(0, length.out = ncol(X)))
  beta[1] <- log(y0/(1-y0))
  stop_condition = FALSE
  while(!stop_condition){
    beta_old <- beta
    p <- 1 / (1 + exp(- X %*% beta_old))
    W <- diag(as.numeric(p * (1 - p)))
    z <- X%*%beta_old + solve(W) %*% (y - p)
    beta <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% z
    if(all(abs(beta - beta_old) < epsilon)){
      stop_condition = TRUE
    }
  }
  structure(.Data  = list(beta = beta), class = c("iwls", "logreg", "model"))
}

predict.iwls <- function(object, X, prob = FALSE, ...){
  X <- cbind(1, X)
  p = 1 / (1 + exp(- X %*% object$beta))
  if(prob){
    return(p)
  }else{
    return(round(p))
  }
}
