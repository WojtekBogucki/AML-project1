source('src/measure.R')
source('src/iwls.R')

beta <- as.matrix(c(rep(1, 10), rep(0, 10)))
n <- 1000
d <- 1:20
x <- sapply(d, function(x) rnorm(n))
p <- 1 / (1 + exp(-t(x%*%beta)))
y <- as.matrix(sapply(p, function(x) rbinom(1, 1, x)))

model <- iwls(x, y)

print(measure(y, predict(model, x)))
