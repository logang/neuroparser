import numpy as np
from synthetic_data import gen_correlated_dataset, classification_dataset
from optimization.cwpath import cwpath, graphnet
from optimization.cwpath.mask import prepare_adj
import scipy.optimize

n = 25
#Generate toy data: 3x3x3x3 data on n trials
X_orig = np.random.normal(0,1,n*3**4).reshape((n,3,3,3,3))

#Reshape X to 2 dimensions
p = np.prod(X_orig.shape[1:])
X = np.zeros((n,p))
for i in range(n):
    X[i,:] = X_orig[i].reshape((1,p))

    beta = np.append(5*np.ones(10),np.zeros(71))
    Y = np.dot(X,beta)

    #Prepare adjacency matrix from "mask" of X (usually we'd get it from a mask file, see mask.py)
    Adj = prepare_adj(1+0.*X_orig[0],0,1,1,1)

    l1 = 50.
    l2 = 5.
    l3 = 30.
    delta = 100.

    #Run optimization
    l = cwpath.CoordWise((X, Y, Adj), graphnet.GraphNet, initial_coefs=np.random.normal(0,1,81))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3)#,delta=delta)
    l.fit()

    beta =np.round(1000*l.problem.coefficients)/1000
    print beta


    """
    def huber(r):
            r = np.fabs(r)
                t = np.greater(r, delta)
                    return (1-t)*r**2 + t*(2*delta*r - delta**2)
                def f(b):
                        return huber(Y - np.dot(X, b)).sum()/2  + np.fabs(b).sum()*l1 + l2 * np.linalg.norm(b)**2/2
                    v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=1e6)
                    v = np.round(1000*np.asarray(v))/1000
                    print beta
                    print v
                    """

def test_NaiveGraphNet_classification( n, p, num_obs_class1 = 100, num_obs_class2 = 100 ):
    """
    Test the naive graphnet on a binary classification problem
    """
    # classification data
    class_data = classification_dataset( n, p, num_obs_class1, num_obs_class2, SNR = 1, corr1 = 0.9, corr2 = None )
    X = class_data[1]
    y = class_data[0]
    
    #Prepare adjacency matrix from "mask" of X (usually we'd get it from a mask file, see mask.py)
    Adj = prepare_adj(1+0.*X,0,1,1,0)
    1/0
    l1 = 50.
    l2 = 5.
    l3 = 30.
    delta = 100.

    #Run optimization
    l = cwpath.CoordWise((X, y, Adj), graphnet.GraphNet ) #, initial_coefs=np.random.normal(0,1,81))
    l.problem.assign_penalty(l1=l1,l2=l2,l3=l3)#,delta=delta)
    l.fit()

    beta =np.round(1000*l.problem.coefficients)/1000
    print beta

if __name__ is "__main__":
    test_NaiveGraphNet_classification(100,100)
