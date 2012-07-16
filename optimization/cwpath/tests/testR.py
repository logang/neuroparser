# cython functions are not yet seen as packages. import path. 
import sys, os
path_to_cython_packages = os.path.abspath('../.')
print path_to_cython_packages
sys.path.append(path_to_cython_packages)

# R imports
#import rpy2.rpy_classic as rpy
#rpy.set_default_mode(NO_CONVERSION)
import rpy2
from rpy2 import robjects as rpy

# major python libraries
import numpy as np

# local imports 
import cwpath, graphnet #, lasso
	
class AlternativeConvergence(graphnet.Lasso):
    """
    Use same convergence criterion as in R script
    """
    tol = 1.0e-10
    
    def Rfit(self, penalty=None):
        penalty = penalty or self.penalty
        rpy.r.assign('X', self.X.ravel())
        rpy.r("X = matrix(X, %d, %d, byrow=T)" % self.X.shape)
        rpy.r.assign('Y', self.Y)
        rpy.r.assign('tol', self.tol)
        rpy.r.assign('p', penalty)
        return rpy.r("cwenet(X, Y, tol, p, 0)")

    def stop(self, previous, tol=None):
        """
        Uses l1 convergence criteria for coefficients
        """
        
        bold, rold = previous
        bcurrent, rcurrent = self.output()
        err = np.fabs(bold - bcurrent).sum() / bold.shape[0]
        
        if err < self.tol:
            return True
    
        return False

# Preamble for R

data = {}
def setup():
    rpy.r('''
    
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
    b[j] <- (1/(n-1))*(sign(S)*pospart(abs(S) - l1))
    }	
    del <- abs(sum(b-b_old[i,]))/length(b)
    }
    return(b)
    }
    ''')
    
    data['Y'] = np.asarray(rpy.r("Y"))
    data['Y'] =  data['Y'].reshape((data['Y'].shape[0],))
    data['X'] = np.asarray(rpy.r("X"))

def test_Renet():
    raise ValueError('write a test using ENet!')

def test_R():
    X = data['X']
    Y = data['Y']
    l = cwpath.CoordWise((X, Y), AlternativeConvergence)
    p = l.problem.penalty
    l1 = 1000
    l.problem.penalty = l1 / np.sqrt(X.shape[0])

    l.fit(l.problem.penalty)
    print l.current[0], l.problem.Rfit(l.problem.penalty)
    assert np.allclose(l.current[0], l.problem.Rfit(l.problem.penalty))

def test_final():
    X = data['X']
    Y = data['Y']
    l = cwpath.CoordWise((X, Y), AlternativeConvergence)
    l.problem.penalty = 0
    l.tol = 1.0e-14
    l.fit()
    b = np.dot(np.linalg.pinv(l.problem.X), l.problem.Y)
    assert(np.allclose(l.problem.beta, b))
    assert(np.allclose(l.problem.Rfit(l.problem.penalty), b))
