########################
##     Libraries      ##
########################
    
library(MASS)
library(lars)
library(elasticnet)

########################
##     Functions      ##
########################
    
"pospart" <- function(a) {
  if(a > 0) {return(a)} else {return(0)}
}
    
########################
##       Data         ##
########################
    
data(diabetes)
X <- diabetes$x
X <- as.matrix(X)
X <- apply(X,2,scale)
Y <- diabetes$y
Y <- scale(matrix(Y), center=T, scale=F)
    
########################
##     Parameters     ##
########################
    
cwenet <- function(X, Y, tol, l1, l2) {
  n <- dim(X)[1]
  p <- dim(X)[2]
    
  # Regularization and Convergence Params #
  
  ########################
  ##       LASSO        ##
  ########################
    
  y <- matrix(Y)
  y <- scale(y, center=T, scale=F)
  X <- as.matrix(X)
  X <- apply(X,2,scale)
  
  # Initialize Betas #
  b <- b_old <- numeric(p)
    
  # Coordinate-wise Fit #
  i <- 0
  del = 1
  while(abs(del) > tol) {
    i <- i+1
    b_old <- rbind(b_old, b)
    for(j in 1:p) {
    rj <- y - X[,-j]%*%b[-j]
    S <- t(X[,j])%*%rj
    b[j] <- (1/n)*(sign(S)*pospart(abs(S) - l1))
    }	
    del <- abs(sum(b-b_old[i,]))/length(b)
  }
  return(b)
}

l <- lars(X, Y, type='lasso')
lc <- predict(l, s=88, mode='norm', type='coef')
print(sum(abs(lc$coef)))
