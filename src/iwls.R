

iwls <- function(X, y, epsilon = 1e-4, max_iter = 100){
  n <- 1:nrow(X)
  X <- cbind(1, X)
  y0 <- mean(y)
  beta <- as.matrix(rep(0, length.out = ncol(X)))
  beta[1] <- log(y0/(1-y0))
  n_iter <- 1
  
  for(i in 1:max_iter){
    
    beta_old <- beta
    p <- 1 / (1 + exp(- X %*% beta_old))
    W <- Matrix::sparseMatrix(n, n, x = as.numeric(p * (1 - p)))
    W_inv <- Matrix::sparseMatrix(n, n, x = as.numeric(1 / (p * (1 - p))))
    z <- X%*%beta_old + W_inv %*% (y - p)
    beta <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% z
    
    if(all(abs(beta - beta_old) < epsilon)) break
    
    if(i == max_iter){
      warning(paste('Algorithm did not converge after max_iter =', max_iter))
      
    }
  }
  structure(.Data  = list(beta = beta, iters = i), class = c("iwls", "logreg", "model"))
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
