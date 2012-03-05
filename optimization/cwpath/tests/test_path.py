import time
import copy

import rpy
import numpy as np
from cwpath import cwpath, lasso
import pylab

from testR import setup, data

def test():
    X = data['X']
    Y = data['Y']

    l = cwpath.CoordWise((X, Y), lasso.Lasso)
    p = l.problem.penalty

    v = []
    l.results = []
    rpy.r("b <- as.numeric(lm(Y ~ X)$coef)")
    b = rpy.r("b")

    for i in range(500):
        l.fit(p)
        l.append(l.current[0].copy())
        p *= 0.98
        print 'here'

    def plot_path(l):
        b = np.asarray(l.results)
        print b.shape
        l1 = np.sum(np.fabs(b), axis=1)
        for i in range(10):
            pylab.plot(l1, b[:,i])
	

    plot_path(l)
    pylab.title("%0.3e" % np.fabs(l.results[-1]).sum())
    beta = np.dot(np.linalg.pinv(l.problem.X), l.problem.Y)
    pylab.scatter([np.fabs(beta).sum()]*beta.shape[0], beta)
    pylab.show()


    return 
