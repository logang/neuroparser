
import rpy
import numpy as np
from cwpath import cwpath, lasso

import testR

def setup():
    testR.setup()

def test():
    X = testR.data['X']
    Y = testR.data['Y']

    l = cwpath.CoordWise((X, Y), lasso.Lasso)
    l.problem.penalty = 5000.
    
    la = cwpath.CoordWise((X, Y), lasso.LassoArray)
    la.problem.penalty[:] = 5000.

    l.fit()
    la.fit()

    assert(np.allclose(la.problem.beta, l.problem.beta))

def test_differentvals():
    X = testR.data['X']
    Y = testR.data['Y']

    l = cwpath.CoordWise((X, Y), lasso.Lasso)
    l.problem.penalty = 5000.
    
    la = cwpath.CoordWise((X, Y), lasso.LassoArray)
    pa = la.problem.penalty
    pa[:3] = 5000.
    pa[3:] = 0.

    print pa
    la.fit()

def test_inf1():
    X = testR.data['X']
    Y = testR.data['Y']

    la = cwpath.CoordWise((X, Y), lasso.LassoArray)
    pa = la.problem.penalty
    pa[:3] = 500.
    pa[3:] = np.inf

    la.fit()
    print la.problem.beta
    assert(np.allclose(la.problem.beta[3:], 0))

def test_inf2():
    X = testR.data['X']
    Y = testR.data['Y']

    la = cwpath.CoordWise((X, Y), lasso.LassoArray)
    pa = la.problem.penalty
    pa[:3] = 0.
    pa[3:] = np.inf

    la.fit()

    x = X[:,:3]
    b = np.dot(np.linalg.pinv(x), Y)
    print la.problem.beta, b

    assert(np.allclose(la.problem.beta[3:], 0))
    assert(np.allclose(la.problem.beta[:3], b))

def test_inf3():
    X = testR.data['X']
    Y = testR.data['Y']

    la = cwpath.CoordWise((X, Y), lasso.LassoArray)
    pa = la.problem.penalty
    pa[:3] = 500.
    pa[3:] = np.inf

    la.fit()

    x = X[:,:3]
    l = cwpath.CoordWise((x, Y), lasso.Lasso) 
    l.problem.penalty = 500.
    l.fit()

    print la.problem.beta[:3], l.problem.beta
    assert(np.allclose(la.problem.beta[:3], l.problem.beta))

