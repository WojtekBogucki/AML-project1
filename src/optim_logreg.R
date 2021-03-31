require(data.table)

optim_logreg <- function(X,Y, method="BFGS", epsilon=1e-7, max_iter=1000){
  X <- cbind(1, X)
  n <- ncol(X)
  m <- nrow(X)
  theta <- as.matrix(rep(0, length.out = n))
  g <- sigmoid(X %*% theta)
  all_costs <<- data.table()
  cost <- function(theta){
    g <- sigmoid(X %*% theta)
    loss <- -sum(Y*log(g + 1e-8) + (1-Y)*log(1-g + 1e-8))/m
    all_costs <<- rbind(all_costs, loss)
    return(loss)
  }
  cost_grad <- function(theta){
    g <- sigmoid(X %*% theta)
    return(t(X) %*% (g - Y)/m)
  }
  if(method=="BFGS") max_iter <- max_iter + 1
  opt <- optim(theta, cost, cost_grad, method = method, control = list(type = 2, trace = 0, maxit=max_iter, reltol=epsilon))
  structure(.Data = list(beta = opt$par, costs = unlist(all_costs), iters=opt$counts[1]-1), class = c("logreg", "model"))
}

# opt <- optim_logreg(x, y)
# opt
# print(measure(y, predict(opt, x)))
# plot(opt)
