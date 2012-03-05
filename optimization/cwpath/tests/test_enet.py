

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
    
    e = cwpath.CoordWise((X, Y), lasso.ENet)
    e.problem.penalty = np.array([5000.,0.])

    l.fit()
    e.fit()

    assert(np.allclose(e.problem.beta, l.problem.beta))

def test_inf():
    X = testR.data['X']
    Y = testR.data['Y']

    e = cwpath.CoordWise((X, Y), lasso.ENet)
    e.problem.penalty = np.array([5000.,np.inf])

    e.fit()

    assert(np.allclose(e.problem.beta, 0))

def test_scaling():
    X = testR.data['X']
    Y = testR.data['Y']

    e = cwpath.CoordWise((X, Y), lasso.ENet)
    e.problem.penalty = np.array([5.,1.])

    ne = cwpath.CoordWise((X, Y), lasso.NaiveENet)
    ne.problem.penalty = np.array([5.,1.])

    e.fit()
    ne.fit()
    beta1, r1 = e.problem.output()
    beta2, r2 = ne.problem.output()
    print beta1
    
    assert(np.allclose(beta1, beta2*2.))
    assert(np.allclose(r1, r2))
    assert(np.allclose(e.problem.beta, beta1/2.))
    assert(np.allclose(ne.problem.beta, beta2))


