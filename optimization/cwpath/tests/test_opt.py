
import numpy as np
from cwpath import cwpath, lasso, strategy
import scipy.optimize
from nose.tools import *

import testR

def setup():
    testR.setup()

def test_lasso(l1=500.):

    testR.setup()
    X = testR.data['X']
    Y = testR.data['Y']

    l = cwpath.CoordWise((X, Y), lasso.Lasso)
    l.problem.assign_penalty(l1=l1)
    l.fit()
    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10)
    v = np.asarray(v)
    
    print np.linalg.norm(v - l.current[0]), f(v), f(l.current[0])
    assert_true(np.fabs(f(v) - f(l.current[0])) / np.fabs(f(v) + f(l.current[0])) < 1.0e-04)
    assert_true(np.linalg.norm(v - l.current[0]) / np.linalg.norm(v) < 1.0e-03)


def test_robust_graphnet(l1 = 500., l2 = 2, l3=3.5, delta = 10.):

    testR.setup()
    X = testR.data['X']
    Y = testR.data['Y']

    A = lasso._create_adj(X.shape[1])
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2

    #Xp = np.hstack([X,np.diag(np.ones(X.shape[0]))])
    l = cwpath.CoordWise((X, Y, A), lasso.RobustGraphnetIC)
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta)
    #l.problem.penalty = [l1, l2, l3, delta]
    l.fit()
    
    def huber(r):
        r = np.fabs(r)
        t = np.greater(r, delta)
        return (1-t)*r**2 + t*(2*delta*r - delta**2)        


    def f(beta):
        return huber(Y - np.dot(X, beta)).sum()/2  + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2

    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-14, xtol=1.0e-14)
    v = np.asarray(v)

   #print np.min(np.fabs(Y- np.dot(X,v))), np.mean(np.fabs(Y- np.dot(X,v))),np.median(np.fabs(Y- np.dot(X,v)))
    
    beta = l.results[0]#[range(X.shape[1])]
    print beta.shape, v.shape

    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    print np.linalg.norm(v - beta), f(v), f(beta)
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    #print(np.linalg.norm(v - beta) / np.linalg.norm(v))
    assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-03)

def test_robust_graphnet_reweight(l1 = 500., l2 = 2, l3=3.5, delta = 5.):

    testR.setup()
    X = testR.data['X']
    Y = testR.data['Y']

    print X.shape

    A = lasso._create_adj(X.shape[1])
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2

    l1weights = np.ones(X.shape[1])

    #Change l1 from being a multiple of the 1 vector
    l1weights[range(4)]=17.
    l1weights[-1] = 100000.

    #Xp = np.hstack([X,np.diag(np.ones(X.shape[0]))])
    l = cwpath.CoordWise((X, Y, A), lasso.RobustGraphnetReweight)
    #l.problem.penalty = [l1, l2, l3, delta]
    #l.problem.penalty = l1, l2, l3
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta,l1weights=l1weights,newl1=l1)
    l.fit()
    
    def huber(r):
        r = np.fabs(r)
        t = np.greater(r, delta)
        return (1-t)*r**2 + t*(2*delta*r - delta**2)        


    def f(beta):
        return huber(Y - np.dot(X, beta)).sum()/2  + l1*np.dot(np.fabs(beta),l1weights) + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2

    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-14, xtol=1.0e-14)
    v = np.asarray(v)

   #print np.min(np.fabs(Y- np.dot(X,v))), np.mean(np.fabs(Y- np.dot(X,v))),np.median(np.fabs(Y- np.dot(X,v)))
    
    beta = l.results[0]#[range(X.shape[1])]
    print beta.shape, v.shape

    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    print np.linalg.norm(v - beta), f(v), f(beta)
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    #print(np.linalg.norm(v - beta) / np.linalg.norm(v))
    assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-03)


def test_graphsvm(l1 = 500., l2 = 200, l3=30.5, delta = 0.5):

    testR.setup()
    X = testR.data['X']
    Y = testR.data['Y']

    A = lasso._create_adj(X.shape[1])
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2


    s = strategy.NStep(X.shape[1], nstep=3)
    #s.max_greedyit = 1000
    #s.max_fitit = 1000
    #l = cwpath.CoordWise((X,Y,adj), problem, strategy=s)

    intercept = False
    
    if intercept:
        Xp = np.hstack([X,np.diag(np.ones(X.shape[0])),np.ones(X.shape[0])[:,np.newaxis]])
    else:
        Xp = np.hstack([X,np.diag(np.ones(X.shape[0]))])    

    l = cwpath.CoordWise((X, Y, A), lasso.RobustGraphSVM)#,strategy=s)
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta)
    #l.problem.penalty = [l1, l2, l3,delta]
    l.fit()

    def huber(r):
        r = np.fabs(r)
        t1 = np.greater(r, 1-delta)
        t2 = np.greater(r, 1)
        return t2*(r-1+delta/2)+ (1-t2)*t1*( (r-1+delta)**2/(2*delta) )
    
    Xp2 = np.hstack([np.ones(X.shape[0])[:,np.newaxis],X])

    def f(beta):
        if intercept:
            ind = range(1,len(beta))
            return huber(Y - np.dot(Xp2, beta)).sum()  + np.fabs(beta[ind]).sum()*l1 + l2 * np.linalg.norm(beta[ind])**2/2 + l3 * np.dot(beta[ind], np.dot(Afull, beta[ind]))/2
        else:
            ind = range(len(beta))
            return huber(Y - np.dot(X, beta)).sum()  + np.fabs(beta[ind]).sum()*l1 + l2 * np.linalg.norm(beta[ind])**2/2 + l3 * np.dot(beta[ind], np.dot(Afull, beta[ind]))/2

    
    if intercept:
        v = scipy.optimize.fmin_powell(f, np.zeros(Xp2.shape[1]), ftol=1.0e-15, xtol=1.0e-15)
    else:
        print X.shape
        v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-15, xtol=1.0e-15)

    """
    if intercept:
        ind = list(np.append(range(X.shape[1]),(Xp.shape[1])-1))
        print ind
        beta = l.current[0][ind]
    else:
        beta = l.current[0][range(X.shape[1])]
    """

    beta = l.results[0]

    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    print np.linalg.norm(v - beta), f(v), f(beta)
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    #print(np.linalg.norm(v - beta) / np.linalg.norm(v))
    assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-03)
    

def test_laplacian(l1 = 500., l2 = 2, l3=3.5):

    testR.setup()
    X = testR.data['X']
    Y = testR.data['Y']

    A = lasso._create_adj(X.shape[1])
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2

    print Afull
    l = cwpath.CoordWise((X, Y, A), lasso.NaiveLaplace)
    l.problem.assign_penalty(**{'l1':l1, 'l2':l2, 'l3':l3})
    l.fit()
    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-08, xtol=1.0e-08)
    v = np.asarray(v)

    print np.linalg.norm(v - l.current[0]), f(v), f(l.current[0])
    assert_true(np.fabs(f(v) - f(l.current[0])) / np.fabs(f(v) + f(l.current[0])) < 1.0e-04)
    assert_true(np.linalg.norm(v - l.current[0]) / np.linalg.norm(v) < 1.0e-03)

def test_enet(l1=500., l2=200.):
    testR.setup()
    X = testR.data['X']
    Y = testR.data['Y']

    l = cwpath.CoordWise((X, Y), lasso.NaiveENet)
    l.problem.assign_penalty(l1=l1, l2=l2)
    l.fit()
    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-08, xtol=1.0e-08)
    v = np.asarray(v)

    print np.linalg.norm(v - l.current[0]), f(v), f(l.current[0])
    assert_true(np.fabs(f(v) - f(l.current[0])) / np.fabs(f(v) + f(l.current[0])) < 1.0e-04)
    assert_true(np.linalg.norm(v - l.current[0]) / np.linalg.norm(v) < 1.0e-03)


def test_array(l1=500, l2=200):
    X = testR.data['X']
    Y = testR.data['Y']
    
    l = cwpath.CoordWise((X, Y), lasso.LassoArray)
    p = l.problem.penalty
    p[:3] = l1
    p[3:] = l2
    l.fit()

    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta[:3]).sum()*l1 + np.fabs(beta[3:]).sum()*l2
    
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10)
    v = np.asarray(v)
    
    print np.linalg.norm(v - l.current[0])
    assert_true(np.fabs(f(v) - f(l.current[0])) / np.fabs(f(v) + f(l.current[0])) < 1.0e-04)
    assert_true(np.linalg.norm(v - l.current[0]) / np.linalg.norm(v) < 1.0e-03)

def test_unnormalized(l1=500,l2=200):
    '''
    columns not normalized to 1 -- does it still solve the optimization problem?
    '''
    X = testR.data['X'].copy()
    X[:,:3] *= 2
    X[:,3:] *= 1.5
    Y = testR.data['Y']

    l = cwpath.CoordWise((X, Y), lasso.NaiveENet)
    l.problem.penalty = [l1,l2]
    l.fit()
    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-08, xtol=1.0e-08)
    v = np.asarray(v)
    
    print v, l.current[0]
    assert_true(np.fabs(f(v) - f(l.current[0])) / np.fabs(f(v) + f(l.current[0])) < 1.0e-04)
    assert_true(np.linalg.norm(v - l.current[0]) / np.linalg.norm(v) < 1.0e-03)

