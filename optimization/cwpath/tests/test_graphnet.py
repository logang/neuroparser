# cython functions are not yet seen as packages. import path. 
import sys, os
path_to_cython_packages = os.path.abspath('../.')
sys.path.append(path_to_cython_packages)

# import major libraries
import numpy as np
import scipy.optimize
from nose.tools import *
import time, h5py

# local imports
import testR
import cwpath, graphnet, strategy

# functions
from numpy.core.umath_tests import inner1d
from multiprocessing import Pool

# setup some R stuff
def setup():
    testR.setup()

#------------------------------------------------------------------------------------------------------#
# Run all GraphNet tests

def test_all(num_l1_steps=100):
    # get data and constants
    Data = np.load("Data.npz")
    X = Data['X']
    Y = Data['Y']
    G = [None]
    lam_max = get_lambda_max(X,Y)
    cwpathtol = 1e-6
    
    # penalty grid values
    l1vec = np.linspace(0.95*lam_max, 0.2*lam_max, num=num_l1_steps).tolist()
    l2vec = [0.0, 1.0, 10, 100, 1000, 10000, 100000, 1000000]
    l3vec = [0.0, 1.0, 10, 100, 1000, 10000, 100000, 1000000]
    deltavec = [0.0, 0.1. 0.25, 0.5. 0.75. 1.0, 20, 100]
    svmdeltavec = [0.0, 0.1. 0.25, 0.5. 0.75. 1.0, 20, 100]

    # construct parameter grid
    penalties = []
    for l2 in l2vec:
        for l3 in l3vec:
            for delta in deltavec:
                for svmdelta in svmdeltavec:
                    penalties.append((l2,l3,delta,svmdelta))

    # construct problems
    problems = ( [ ( X, Y, G, 
                     penalties[t],
                     num_l1_steps,
                     lam_max,
                     t ) for t in range( len(penalties) ) ] )

    # test imap
    # in_tuple = (X,Y,G,(100,100,1,0.25),10,lam_max,3)
    # _graphnet_imap( in_tuple )
    # 1/0

    # run problems in parallel with imap
    pool = Pool()
    results = pool.imap( _graphnet_imap, problems )

    # write results to h5 file
    outfile = h5py.File("Grid_output.h5",'w')
    for r in results:
        if r[0] not in outfile.keys():
            outfile.create_group(r[0]) # group by problem type
        if str(r[1]) not in outfile[r[0]].keys():
            outfile[r[0]].create_group(str(r[1])) # group by penalty tuple
#        print "------------------------>",r[0], r[1], r[2]
        outfile[r[0]][str(r[1])]['params'] = np.array(r[1])
        outfile[r[0]][str(r[1])]['l1vec'] = np.array(r[2])
        outfile[r[0]][str(r[1])]['results'] = np.array(r[3])
    outfile.close()

    print "\n\n Congratulations - nothing exploded!"

#------------------------------------------------------------------------------------------------------#
# Wrapper for running GraphNet problems using multiprocessing

def _graphnet_imap( in_tuple ):
    """
    Run a graphnet model for a particular tuple (problem_type,X,Y,G,l2,l3,delta,num_l1,lam_max)
    for a grid of l1 parameters.
    """
    X       = in_tuple[0]
    Y       = in_tuple[1]
    G       = in_tuple[2][0]
    pen     = in_tuple[3]
    num_l1  = in_tuple[4]
    lam_max = in_tuple[5]

    l2 = in_tuple[3][0]
    l3 = in_tuple[3][1]
    delta = in_tuple[3][2]
    svmdelta = in_tuple[3][3]

    lam_max = get_lambda_max(X,Y)
    l1vec = np.linspace(0.95*lam_max, 0.2*lam_max, num=num_l1)
    cwpathtol = 1e-6

    results, problemkey = test_graphnet(X,Y,G,l1vec,l2,l3,delta,svmdelta,initial=None,tol=cwpathtol,scipy_compare=False)

    return (problemkey, pen, l1vec, results)

#------------------------------------------------------------------------------------------------------#
# Main Graphnet problem testing function

def test_graphnet(X,Y,G=None,l1=500.,l2=0.0,l3=0.0,delta=0.0,svmdelta=0.0,initial=None,adaptive=False,svm=False,scipy_compare=True,tol=1e-4):
    tic = time.clock()
    # Cases set based on parameters and robust/adaptive/svm flags
    if l2 != 0.0 or l3 != 0.0 or delta != 0.0 or svmdelta != 0.0:
        if l3 != 0.0 or delta != 0.0 or svmdelta != 0.0:
            if G is None:
                A, Afull = gen_adj(X.shape[1])
            else:
                A = G.copy()
            if delta != 0.0:
                if svmdelta != 0.0:                    
                    print "-------------------------------------------HUBER SVM---------------------------------------------------"
                    problemtype = "HuberSVMGraphNet"
                    problemkey = "HuberSVMGraphNet"
                    print "HuberSVM GraphNet with penalties (l1,l2,l3,delta):", l1, l2, l3, delta
                    Y = 2*np.round(np.random.uniform(0,1,len(Y)))-1        
                    l = cwpath.CoordWise((X, Y, A), graphnet.GraphSVM) #, initial_coefs=10.*np.array(range(11)*1))
                    l.problem.assign_penalty(path_key='l1',l1=l1,l2=l2,l3=l3,delta=delta)
                else:
                    print "----------------------------------------ROBUST GRAPHNET------------------------------------------------"
                    problemtype = graphnet.RobustGraphNet
                    problemkey = 'RobustGraphNet'
                    print "Robust GraphNet with penalties (l1, l2, l3, delta)", l1, l2, l3, delta
                    l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs = initial) #,initial_coefs=np.array([14.]*10))
                    l.problem.assign_penalty(path_key='l1',l1=l1,l2=l2,l3=l3,delta=delta)
            else:
                print "-------------------------------------------GRAPHNET---------------------------------------------------"
                problemtype = graphnet.NaiveGraphNet
                problemkey = 'NaiveGraphNet'
                print "Testing GraphNet with penalties (l1,l2,l3):", l1,l2,l3
                l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs = initial)
                l.problem.assign_penalty(path_key='l1',l1=l1, l2=l2, l3=l3)
        else:
            print "-------------------------------------------ELASTIC NET---------------------------------------------------"
            problemtype = graphnet.NaiveENet
            problemkey = 'NaiveENet'
            print "Testing ENET with penalties (l1,l2):", l1, l2
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs = initial) #, initial_coefs = np.array([4.]*10))
            l.problem.assign_penalty(path_key='l1',l1=l1, l2=l2)
    else:
        print "-------------------------------------------LASSO---------------------------------------------------"
        problemtype = graphnet.Lasso
        problemkey = 'Lasso'
        print "Testing LASSO with penalty:", l1
        l = cwpath.CoordWise((X, Y), problemtype, initial_coefs = initial) #, initial_coefs= np.array([7.]*10))
        l.problem.assign_penalty(path_key='l1',l1=l1)

    # fit and get results
    results = l.fit(tol=tol, initial=initial)
    beta = l.results[0]
    print "\t---> Fitting GraphNet problem with coordinate descent took:", time.clock()-tic, "seconds."

    if adaptive:
        tic = time.clock()
        l1weights = 1./beta
        if problemtype != "HuberSVMGraphNet":
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs = initial)
            l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta,l1weights=l1weights,newl1=l1)
            results.append( l.fit(tol=tol, initial=initial) )
            print "\t---> Fitting Adaptive GraphNet problem with coordinate descent took:", time.clock()-tic, "seconds."
    
    # if compare to scipy flag is set,
    # compare the above result with the same problem 
    # solved using a built in scipy solver (fmin_powell).
    if scipy_compare:
        print "\t---> Fitting with scipy for comparison..."
        tic = time.clock()
        if l2 != 0.0 or l3 != 0.0 or delta != 0.0 or svmdelta != 0.0:
            if l3 != 0.0 or delta != 0.0 or svmdelta != 0.0:
                if delta != 0.0:
                    if adaptive: 
                        if svmdelta != 0.0:
                            # HuberSVM Graphnet
                            Xp2 = np.hstack([np.ones(X.shape[0])[:,np.newaxis],X])
                            def f(beta):
                                ind = range(1,len(beta))
                                return huber_svm_error(beta).sum()  + np.fabs(beta[ind]).sum()*l1 + l2 * np.linalg.norm(beta[ind])**2/2 + l3 * np.dot(beta[ind], np.dot(Afull, beta[ind]))/2
                        else:
                            # Robust Adaptive Graphnet
                            def f(beta):
                                return huber(Y - np.dot(X, beta)).sum()/2  + l1*np.dot(np.fabs(beta),l1weights) + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
                    else:
                        # Robust Graphnet
                        def f(beta):
                            return huber(Y - np.dot(X, beta)).sum()/2  + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
                else:
                    # Graphnet 
                    def f(beta):
                        return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta))/2
            else:
                # Elastic Net
                def f(beta):
                    return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2
        else:
            # Lasso
            def f(beta):
                return np.linalg.norm(Y - np.dot(X, beta))**2/2 + np.fabs(beta).sum()*l1
        # optimize
        if problemkey == 'HuberSVMGraphNet':
            v = scipy.optimize.fmin_powell(f, np.zeros(Xp2.shape[1]), ftol=1.0e-14, xtol=1.0e-14, maxfun=100000)
        else:
            v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10,maxfun=100000)
        v = np.asarray(v)
        print "\t---> Fitting GraphNet with scipy took:", time.clock()-tic, "seconds."

        # print np.round(100*v)/100,'\n', np.round(100*beta)/100
        assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < tol)
        if np.linalg.norm(v) > 1e-8:
            assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < tol)
        else:
            assert_true(np.linalg.norm(beta) < 1e-8)

        print "\t---> Coordinate-wise and Scipy optimization agree!"

    return results, problemkey

#------------------------------------------------------------------------------------------------------#
# Adjacency matrix functions

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

#------------------------------------------------------------------------------------------------------#
# For finding starting lambda
def get_lambda_max(X,y):
    """ 
    Find the value of lambda at which all coefficients are set to zero
    by finding the minimum value such that 0 is in the subdifferential
    and the coefficients are all zero.
    """
    subgrads = np.fabs( inner1d(X.T, y))
    return np.max( subgrads )

#------------------------------------------------------------------------------------------------------#
# Some loss functions for tests

def huber(r):
    r = np.fabs(r)
    t = np.greater(r, delta)
    return (1-t)*r**2 + t*(2*delta*r - delta**2)        

def huber_svm(r):
    t1 = np.greater(r, delta)
    t2 = np.greater(r,0)
    return t1*(r - delta/2) + (1-t1)*t2*(r**2/(2*delta))

def huber_svm_error(beta):
    r = 1-Y*np.dot(Xp2,beta)
    return huber(r)

#-------------------------------------------------------------------------------------------------------------------#

"""
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
