require(ggplot2)

predict.logreg <- function(object, X, prob = FALSE, cutoff = 0.5, ...){
  X <- cbind(1, X)
  p <- sigmoid(X %*% object$beta)
  if(prob) return(p)
  else return(as.numeric(p > cutoff))
}

sigmoid <- function(x){
  sigm <- 1/(1 + exp(-x))
  return(sigm)
}

plot.logreg <- function(object){
  df <- data.frame(iters=0:object$iters, loss=object$costs)
  ggplot(data=df, aes(x=iters, y=loss)) + geom_line() + labs(title = "Loss") + xlab("Number of iterations")
}