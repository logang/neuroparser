# cython functions are not yet seen as packages. import path. 
import sys
sys.path.append('/Users/logang/Documents/Code/python/neuroparser/optimization/cwpath')

# import major libraries
import numpy as np
import scipy.optimize
from nose.tools import *
import time
import testR

# local imports
import cwpath, graphnet, strategy

def setup():
    testR.setup()

def test_all():
    X = np.load('X.npy')
    Y = np.load('Y.npy')

    #X2 = np.random.normal(0,1,np.prod(X.shape)).reshape(X.shape)
    #X = np.hstack([X,X2])
    #print X.shape
    l1vec = [1,10,100]
    l2vec = [10,20,100]
    l3vec = [30,50,100]

    deltavec = [1000,10,0.1]
    unitdeltavec = [0.25,0.5,0.75]


    #l1 = range(1000,0,-10)
    #l1 = 5
    #test_path(X,Y,l1,1000,1000)
    #test_HuberSVM(X,Y,1,1,1,0.5)

    cwpathtol = 1e-6

    for l1 in l1vec:
        #test_lasso(X,Y,l1,tol=cwpathtol)
        for l2 in l2vec:
            #test_enet(X,Y,l1,l2,tol=cwpathtol)
            #test_array(X,Y,l1,l2,tol=cwpathtol)
            #test_unnormalized(X,Y,l1,l2,tol=cwpathtol)
            for l3 in l3vec:
                #test_graphnet(X,Y,l1,l2,l3,tol=cwpathtol)
                for delta in deltavec:
                    test_robust_graphnet(X,Y,l1,l2,l3,delta,tol=cwpathtol)
                #    #test_robust_graphnet_reweight(X,Y,l1,l2,l3,delta,tol=cwpathtol)
                #for delta in unitdeltavec:
                    #test_HuberSVM(X,Y,l1,l2,l3,delta,tol=cwpathtol/10.)
                    #test_graphsvm(X,Y,l1,l2,l3,delta)


    print "\n\n Congratulations - nothing exploded!"


def adj_array_as_list(adj):
    # Now create the adjacency list

    v = []
    for a in adj:
        v.append(a[np.greater(a, -1)])
    return v
                        
def gen_adj(p):

    Afull = np.zeros((p,p),dtype=int)
    A = - np.ones((p,p),dtype=int)
    counts = np.zeros(p)
    for i in range(p):
        for j in range(p):
            if np.random.uniform(0,1) < 0.3:
                if i != j:
                    if Afull[i,j] == 0:
                        Afull[i,j] = -1
                        Afull[j,i] = -1
                        Afull[i,i] += 1
                        Afull[j,j] += 1
                        A[i,counts[i]] = j
                        A[j,counts[j]] = i
                        counts[i] += 1
                        counts[j] += 1
    return adj_array_as_list(A), Afull
                                                                                                                                                                                                                                    

def test_path(X,Y,l1 = 500., l2 = 2, l3=3.5):

    A = graphnet._create_adj(X.shape[1])
    Afull = np.zeros((X.shape[1], X.shape[1]))
    for i, a in enumerate(A):
        Afull[i,a] = -1
        Afull[a,i] = -1
        Afull[i,i] += 2


    l = cwpath.CoordWise((X, Y, A), graphnet.NaiveGraphNet, initial_coefs = np.array([7.]*10))
    t1 = time.time()
    l.problem.assign_penalty(**{'l1':l1, 'l2':l2, 'l3':l3})
    l.fit()
    l.fit()
    t2 = time.time()
    time1 = t2-t1
    print "Time: ", t2-t1

    """
    time2 = 0
    for i in range(len(l1)):
        l = cwpath.CoordWise((X, Y, A), graphnet.NaiveGraphNet, initial_coefs = np.array([7.]*10))
        t1 = time.time()
        l.problem.assign_penalty(**{'l1':l1[i], 'l2':l2, 'l3':l3})
        l.fit()
        t2 = time.time()
        time2 += t2-t1
    print time2, time1, time2/time1

    if len(l1)>1:
        l1=l1[len(l1)-1]
    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-08, xtol=1.0e-08)
    v = np.asarray(v)

    #print np.linalg.norm(v - l.current[0]), f(v), f(l.current[0])
    assert_true(np.fabs(f(v) - f(l.current[0])) / np.fabs(f(v) + f(l.current[0])) < 1.0e-04)
    #assert_true(np.linalg.norm(v - l.current[0]) / np.linalg.norm(v) < 1.0e-03)
    beta = l.results[0]
    print np.round(1000*v)/1000,'\n', np.round(1000*beta)/1000
    if np.linalg.norm(v) > 1e-8:
        assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-03)
    else:
        assert_true(np.linalg.norm(beta) < 1e-8)
    """


def test_lasso(X,Y,l1=500.,tol=1e-4):


    print "LASSO", l1
    l = cwpath.CoordWise((X, Y), graphnet.Lasso, initial_coefs= np.array([7.]*10))
    l.problem.assign_penalty(l1=l1)
    l.fit(tol=tol)
    beta = l.results[0]


    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
    v = np.asarray(v)
    
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    if np.linalg.norm(v) > 1e-8:
        assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
    else:
        assert_true(np.linalg.norm(beta) < 1e-8)



def test_enet(X,Y,l1=500., l2=200.,tol=1e-4):

    print "ENET", l1, l2

    l = cwpath.CoordWise((X, Y), graphnet.NaiveENet, initial_coefs = np.array([4.]*10))
    l.problem.assign_penalty(l1=l1, l2=l2)
    l.fit(tol=tol)
    beta = l.results[0]

    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)

    #print np.round(100*v)/100,'\n', np.round(100*beta)/100
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    if np.linalg.norm(v) > 1e-8:
        #print np.linalg.norm(v - beta) / np.linalg.norm(v) 
        assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
    else:
        assert_true(np.linalg.norm(beta) < 1e-8)

def test_array(X,Y,l1=500, l2=200,tol=1e-4):
    
    l = cwpath.CoordWise((X, Y), graphnet.LassoArray,initial_coefs= np.array([7.]*10))
    l1 = np.ones(10)*l1
    l1[3:] = l2
    l.problem.assign_penalty(l1=l1)
    l.fit(tol=tol)
    beta = l.results[0]

    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.dot(np.fabs(beta),l1)
        
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)

    #print np.round(100*v)/100,'\n', np.round(100*beta)/100

    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    if np.linalg.norm(v) > 1e-8:
        assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
    else:
        assert_true(np.linalg.norm(beta) < 1e-8)


def test_unnormalized(X,Y,l1=500,l2=200,tol=1e-4):
    '''
    columns not normalized to 1 -- does it still solve the optimization problem?
    '''
    print "Unnormalized", l1,l2
    X = X.copy()
    X[:,:3] *= 2
    X[:,3:] *= 1.5

    l = cwpath.CoordWise((X, Y), graphnet.NaiveENet)
    l.problem.assign_penalty(l1=l1, l2=l2)
    l.fit(tol=tol)
    beta  = l.results[0]
    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)

    #print np.round(100*v)/100,'\n', np.round(100*beta)/100
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    if np.linalg.norm(v) > 1e-8:
        assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
    else:
        assert_true(np.linalg.norm(beta) < 1e-8)



def test_graphnet(X,Y,l1 = 500., l2 = 2, l3=3.5,tol=1e-4):

    print "GraphNet", l1,l2,l3
    A, Afull = gen_adj(X.shape[1])
    print A
    l = cwpath.CoordWise((X, Y, A), graphnet.NaiveGraphNet, initial_coefs = np.array([7.]*10))
    l.problem.assign_penalty(l1=l1, l2=l2, l3=l3)

    l.fit(tol=tol)
    beta = l.results[0]
    
    def f(beta):
        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)

    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    if np.linalg.norm(v) > 1e-8:
        assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
    else:
        assert_true(np.linalg.norm(beta) < 1e-8)


def test_robust_graphnet(X,Y,l1 = 5000., l2 = 2, l3=3.5, delta = 1.,tol=1e-4):

    print "Robust GraphNet", l1, l2, l3, delta

    A,Afull = gen_adj(X.shape[1])
    l = cwpath.CoordWise((X, Y, A), graphnet.RobustGraphNet,initial_coefs=np.array([14.]*10))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta)
    l.fit(tol=tol)
    beta = l.results[0]
    
    def huber(r):
        r = np.fabs(r)
        t = np.greater(r, delta)
        return (1-t)*r**2 + t*(2*delta*r - delta**2)        


    def f(beta):
        return huber(Y - np.dot(X, beta)).sum()/2  + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2

    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
    v = np.asarray(v)

    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    if f(beta) > f(v):
        assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert_true(np.linalg.norm(beta) < 1e-8)



def test_robust_graphnet_reweight(X,Y,l1 = 500., l2 = 2, l3=3.5, delta = 5.,tol=1e-4):

    print "Robust GraphNet reweight", l1, l2, l3, delta
    A,Afull = gen_adj(X.shape[1])

    l1weights = np.ones(X.shape[1])

    #Change l1 from being a multiple of the 1 vector
    l1weights[range(4)]=17.
    l1weights[-1] = 45.
    l1weights[9] = 1.


    l = cwpath.CoordWise((X, Y, A), graphnet.RobustGraphNetReweight, initial_coefs = np.array([77.]*10))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta,l1weights=l1weights,newl1=l1)
    l.fit(tol=tol)
    beta = l.results[0]
    
    def huber(r):
        r = np.fabs(r)
        t = np.greater(r, delta)
        return (1-t)*r**2 + t*(2*delta*r - delta**2)        

    def f(beta):
        return huber(Y - np.dot(X, beta)).sum()/2  + l1*np.dot(np.fabs(beta),l1weights) + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2

    
    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-14, xtol=1.0e-14, maxfun=100000)
    v = np.asarray(v)

    #print np.round(100*v)/100,'\n', np.round(100*beta)/100, '\n', f(beta), f(v)
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    if f(beta) >= f(v):
        if np.linalg.norm(v) > 1e-8:
            assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert_true(np.linalg.norm(beta) < 1e-8)




def test_HuberSVM(X,Y,l1 = 500., l2 = 200, l3=30.5, delta = 0.5,tol=1e-4):

    print "HuberSVM", l1, l2, l3, delta
    Y = 2*np.round(np.random.uniform(0,1,len(Y)))-1
    A, Afull = gen_adj(X.shape[1])
        
    l = cwpath.CoordWise((X, Y, A), graphnet.GraphSVM, initial_coefs=10.*np.array(range(11)*1))
    l.problem.assign_penalty(**{'l1':l1,'l2':l2,'l3':l3,'delta':delta})
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta)
    l.fit(tol=tol)
    beta = l.results[0]
    
    def huber(r):
        t1 = np.greater(r, delta)
        t2 = np.greater(r,0)
        return t1*(r - delta/2) + (1-t1)*t2*(r**2/(2*delta))
    
    Xp2 = np.hstack([np.ones(X.shape[0])[:,np.newaxis],X])
    
    def error(beta):
        r = 1-Y*np.dot(Xp2,beta)
        return huber(r)
    
    def f(beta):
        ind = range(1,len(beta))
        return error(beta).sum()  + np.fabs(beta[ind]).sum()*l1 + l2 * np.linalg.norm(beta[ind])**2/2 + l3 * np.dot(beta[ind], np.dot(Afull, beta[ind]))/2
    
    
    v = scipy.optimize.fmin_powell(f, np.zeros(Xp2.shape[1]), ftol=1.0e-14, xtol=1.0e-14, maxfun=100000)
    v = np.asarray(v)
    
    print np.round(100*v)/100,'\n', np.round(100*beta)/100, '\n', f(beta), f(v)

    if f(beta) > f(v)+1e-10:
        assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
        if np.linalg.norm(v) > 1e-8:
            assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-04)
        else:
            assert_true(np.linalg.norm(beta) < 1e-8)



"""
def test_graphsvm(X,Y,l1 = 500., l2 = 200, l3=30.5, delta = 0.5):

    #testR.setup()
    #X = testR.data['X']
    #Y = testR.data['Y']

    A = graphnet._create_adj(X.shape[1])
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

    l = cwpath.CoordWise((X, Y, A), graphnet.GraphSVMRegression, initial_coefs = np.array([44.]*10))#,strategy=s)
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


    #if intercept:
    #    ind = list(np.append(range(X.shape[1]),(Xp.shape[1])-1))
    #    print ind
    #    beta = l.current[0][ind]
    #else:
    #    beta = l.current[0][range(X.shape[1])]


    beta = l.results[0]

    print np.round(100*v)/100,'\n', np.round(100*beta)/100
    print np.linalg.norm(v - beta), f(v), f(beta)
    assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < 1.0e-04)
    #print(np.linalg.norm(v - beta) / np.linalg.norm(v))
    #assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-03)
    if f(v) < f(beta):
        if np.linalg.norm(v) > 1e-8:
            assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < 1.0e-03)
        else:
            assert_true(np.linalg.norm(beta) < 1e-8)


"""
