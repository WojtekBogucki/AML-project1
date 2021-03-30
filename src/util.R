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

plot.logreg <- function(object, title="Loss"){
  df <- data.frame(iters=0:object$iters, loss=object$costs)
  ggplot(data=df, aes(x=iters, y=loss)) + geom_line() + labs(title = title) + xlab("Number of iterations")
}

compare_models <- function(models, labels, title="Loss"){
  df <- data.frame(iters=numeric(), loss=numeric(), label=character())
  for (i in 1:length(models)){
    df <- rbind(df, data.frame(iters=0:models[[i]]$iters, loss=models[[i]]$costs, label=labels[i]))
  }
  ggplot(data=df, aes(x=iters, y=loss, color=label)) + geom_line(size=1) + labs(title = title, color="Algorithm") + xlab("Number of iterations")
}