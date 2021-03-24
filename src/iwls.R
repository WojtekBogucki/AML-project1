require(Matrix)
source('src/util.R')

iwls <- function(X, y, epsilon = 1e-7, max_iter = 100){
  
  m <- nrow(X)
  m_vec <- 1:m
  X <- cbind(1, X)
  beta <- as.matrix(rep(0, length.out = ncol(X)))
  p <- sigmoid(X %*% beta)
  all_costs <- c(-sum(y*log(p + 1e-8) + (1-y)*log(1-p + 1e-8))/m)
  
  for(i in 1:max_iter){
    p_vec <- as.numeric(p * (1 - p)) + 1e-8
    W <- sparseMatrix(m_vec, m_vec, x = p_vec)
    W_inv <- sparseMatrix(m_vec, m_vec, x = p_vec)
    z <- X%*%beta + W_inv %*% (y - p)
    beta <- solve(t(X) %*% W %*% X) %*% t(X) %*% W %*% z
    p <- sigmoid(X %*% beta)
    
    all_costs[i+1] <- -sum(y*log(p + 1e-8) + (1-y)*log(1-p + 1e-8))/m

    if(all(abs(all_costs[i+1] - all_costs[i]) < epsilon)) break
    
    if(i == max_iter){
      warning(paste('Algorithm did not converge after max_iter =', max_iter))
      
    }
  }
  structure(.Data  = list(beta = beta, iters = i, costs = all_costs), 
            class = c("iwls", "logreg", "model"))
}

