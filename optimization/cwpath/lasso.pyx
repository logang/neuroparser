import numpy as np
cimport numpy as np
import time

## Local imports

from regression import Regression
## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

class Lasso(Regression):

    """
    LASSO problem with one penalty parameter
    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - X\beta||^{2}_{2} + \lambda_{1}||\beta||_{1}
       \end{eqnarray}

    as a function of beta.
    """

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """

        self._Xssq = np.sum(self.X**2, axis=0)
        self.penalty = self.default_penalty()

    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        #c = np.fabs(np.dot(self.X.T, self.Y)).max()
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1']]))
        


    def update(self, active, nonzero, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        return _update_lasso(active,
                             self.penalty,
                             nonzero,
                             self.beta,
                             self.r,
                             self.X,
                             self._Xssq)

class LassoArray(Lasso):

    """
    LASSO problem with one penalty parameter per
    coefficient.

    Minimizes

    .. math::
       \begin{eqnarray}
       ||y - X\beta||^{2}_{2} + \sum_{j} \lambda_{j}|\beta_{j}|

    as a function of beta.
    """


    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        c = np.fabs(np.dot(self.X.T, self.Y)).max()
        return np.ones(self.X.shape[1])*c

    def update(self, active, nonzero, permute=False):

        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        return _update_lasso_array(active,
                                   self.penalty,
                                   nonzero,
                                   self.beta,
                                   self.r,
                                   self.X,
                                   self._Xssq)

class NaiveENet(Lasso):

    """
    Naive ENet problem. Minimizes

    (np.sum((Y - np.dot(X, beta))**2) / 2 + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2)) / 2

    as a function of beta.
    """
    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2']]))

    def update(self, active, nonzero, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        v = _update_enet(active,
                         self.penalty,
                         nonzero,
                         self.beta,
                         self.r,
                         self.X,
                         self._Xssq)
        return v


class ENet(NaiveENet):

    """
    ENet problem. Takes NaiveENet solution
    and scales coefficients on self.output() by (1 + l2)
    with l1, l2 = self.penalty.

    The NaiveENet solution minimizes

    (np.sum((Y - np.dot(X, beta))**2) / 2 + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))

    as a function of beta.

    NOTE: For ENet, the coefficients are scaled by (1 + l2)
          as in Zhou & Hastie (2005). This is scaling is not reflected
          in the residual because the update steps use the
          "naive Enet" coefficients.

    NOTE: self.beta corresponds to the NaiveENet solution, not the
          ENet solution.

    """

    def output(self):
        """
        Return the 'interesting' part of the problem arguments.

        In the regression case, this is the tuple (beta, r).

        NOTE: For ENet, the coefficients are scaled by (1 + l2)
              as in Zhou & Hastie (2005). This is scaling is not reflected
              in the residual because the update steps use the
              "naive Enet" coefficients.

        NOTE: self.beta corresponds to the NaiveENet solution, not the
              ENet solution.

        """
        l2 = self.penalty['l2']

        return self.beta * (1 + l2), self.r


class NaiveLaplace(Regression):

    """
    The Naive Laplace problem with three penalty parameters
    l1, l2 and l3 minimizes

    (np.sum((Y - np.dot(X, beta))**2) + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))
     + l3* np.dot(np.dot(beta,np.dot(D-A)),beta)

     as a function of beta,
     where D = diag(N_1, ..., N_p) where N_i is the number
     of neighbors of coefficient i, and A_{ij} = 1(j is i's neighbor)

    """

    def __init__(self, data):
        
        if len(data) != 3:
            raise ValueError('expecting adjacency matrix for Laplacian smoothing')
        _, _, self.adj = data
        Regression.__init__(self, data[:2])

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """

        self._Xssq = np.sum(self.X**2, axis=0)
        self.penalty = self.default_penalty()
        print self.penalty, 'initialize'
        self.nadj = _create_nadj(self.adj)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3']]))
        
    def update(self, active, nonzero, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        print active.shape
        return _update_naive_laplace(active,
                             self.penalty,
                             nonzero,
                             self.beta,
                             self.r,
                             self.X,
                             self._Xssq, 
			     self.adj, 
			     self.nadj)
                       
class Laplace(NaiveLaplace):

    """
    ENet problem. Takes NaiveENet solution
    and scales coefficients on self.output() by (1 + l2)
    with l1, l2, l3 = self.penalty.

    """
    def output(self):
        """
        Return the 'interesting' part of the problem arguments.

        In the regression case, this is the tuple (beta, r).
        """
        l2 = self.penalty['l2']
        
        return self.beta * (1 + l2), self.r

class RobustGraphnetIC(Regression):

    """
    The Naive Laplace problem with three penalty parameters
    l1, l2 and l3 minimizes
    (np.sum((Y - np.dot(X, beta))**2) + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))
     + l3* np.dot(np.dot(beta,np.dot(D-A)),beta)

     as a function of beta,
     where D = diag(N_1, ..., N_p) where N_i is the number
     of neighbors of coefficient i, and A_{ij} = 1(j is i's neighbor)

    """

    def __init__(self, data):
        
        if len(data) != 3:
            raise ValueError('expecting adjacency matrix for Laplacian smoothing')
        _, _, self.adj = data
        Regression.__init__(self, data[:2])

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """
        #3
        self.X = np.hstack([self.X,np.diag(np.ones(self.X.shape[0]))])
        self._Xssq = np.sum(self.X**2, axis=0)
        self.beta = np.zeros(self.X.shape[1])
        self.penalty = self.default_penalty()
        self.nadj = _create_nadj(self.adj)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        #return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3','delta']]))
        d = np.dtype([('l1', np.float),
                      ('l2', np.float),
                      ('l3', np.float),
		      ('delta', np.float),
                      ('newl1', np.float)])
        return np.zeros((), d)        

    def trim_beta(self):
        #Remove extra coefficients from infimal convolution
        return self.beta[range(self.X.shape[1]-self.X.shape[0])]

    def num_coefs(self):
        #Return the number of total coefficients, and the number in the original problem
        return self.X.shape[1], self.X.shape[1]-self.X.shape[0]

    def update(self, active, nonzero, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        print active.shape
        return _update_robust_graphnetIC(active,
                             self.penalty,
                             nonzero,
                             self.beta,
                             self.r,
                             self.X,
                             self._Xssq, 
			     self.adj, 
			     self.nadj)

class RobustGraphnetReweight(Regression):

    """
    The Naive Laplace problem with three penalty parameters
    l1, l2 and l3 minimizes
    (np.sum((Y - np.dot(X, beta))**2) + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))
     + l3* np.dot(np.dot(beta,np.dot(D-A)),beta)

     as a function of beta,
     where D = diag(N_1, ..., N_p) where N_i is the number
     of neighbors of coefficient i, and A_{ij} = 1(j is i's neighbor)

    """

    def __init__(self, data):
        
        if len(data) != 3:
            raise ValueError('expecting adjacency matrix for Laplacian smoothing')
        _, _, self.adj = data
        self._coef_shape = data[0].shape[1]
        Regression.__init__(self, data[:2])

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """
        #3
        self.penalty = self.default_penalty()
        self.X = np.hstack([self.X,np.diag(np.ones(self.X.shape[0]))])
        self._Xssq = np.sum(self.X**2, axis=0)
        self.beta = np.zeros(self.X.shape[1])
        self.nadj = _create_nadj(self.adj)

    def default_penalty(self):
        """
        Default penalty for Lasso: a single
        parameter problem.
        """
        #c = np.fabs(np.dot(self.X.T, self.Y)).max()
        #return np.ones(self.X.shape[1])*c,0.,0.

        d = np.dtype([('l1', np.float),
                      ('l2', np.float),
                      ('l3', np.float),
		      ('delta', np.float),
                      ('l1weights', '(%d,)f8' % self._coef_shape),
                      ('newl1', np.float)])
        return np.zeros((), d)

    def trim_beta(self):
        #Remove extra coefficients from infimal convolution
        return self.beta[range(self.X.shape[1]-self.X.shape[0])]

    def num_coefs(self):
        #Return the number of total coefficients, and the number in the original problem
        return self.X.shape[1], self.X.shape[1]-self.X.shape[0]

    def update(self, active, nonzero, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        print active.shape
        return _update_robust_graphnet_reweight(active,
                             self.penalty,
                             nonzero,
                             self.beta,
                             self.r,
                             self.X,
                             self._Xssq, 
			     self.adj, 
			     self.nadj)



class RobustGraphSVM(Regression):

    """
    The Naive Laplace problem with three penalty parameters
    l1, l2 and l3 minimizes
    (np.sum((Y - np.dot(X, beta))**2) + l1 *
     np.fabs(beta).sum() + l2 * np.sum(beta**2))
     + l3* np.dot(np.dot(beta,np.dot(D-A)),beta)

     as a function of beta,
     where D = diag(N_1, ..., N_p) where N_i is the number
     of neighbors of coefficient i, and A_{ij} = 1(j is i's neighbor)

    """

    def __init__(self, data):
        
        if len(data) != 3:
            raise ValueError('expecting adjacency matrix for Laplacian smoothing')
        _, _, self.adj = data
        Regression.__init__(self, data[:2])

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """
        #With intercept
        #self.X = np.hstack([self.X,np.diag(np.ones(self.X.shape[0])),np.ones(self.X.shape[0])[:,np.newaxis]])
        #Without intercept
        self.X = np.hstack([self.X,np.diag(np.ones(self.X.shape[0]))])
        self._Xssq = np.sum(self.X**2, axis=0)
        self.beta = np.zeros(self.X.shape[1])
        self.penalty = self.default_penalty()
        self.nadj = _create_nadj(self.adj)

    def default_penalty(self):
        """
        Default penalty for Naive Laplace: a single
        parameter problem.
        """
        return np.zeros(1, np.dtype([(l, np.float) for l in ['l1', 'l2', 'l3','delta']]))
          
    def trim_beta(self):
        #Remove extra coefficients from infimal convolution
        #Return with or without intercept
        #return self.beta[list([-1])+range(self.X.shape[1]-self.X.shape[0]-1)]
        return self.beta[range(self.X.shape[1]-self.X.shape[0])]

    def num_coefs(self):
        #Return the number of total coefficients, and the number in the original problem
        return self.X.shape[1], self.X.shape[1]-self.X.shape[0]-1

    def update(self, active, nonzero, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        print active.shape
        return _update_robust_graphSVM(active,
                             self.penalty,
                             nonzero,
                             self.beta,
                             self.r,
                             self.X,
                             self._Xssq, 
			     self.adj, 
			     self.nadj)


class HuberSVM(Regression):

    """
    Compute the solution to the Huberized SVM
    """

    def __init__(self, data):

        if len(data) != 3:
            raise ValueError('expecting adjacency matrix for Laplacian smoothing')
        _, _, self.adj = data
        Regression.__init__(self, data[:2])

    def initialize(self):
        """
        Generate initial tuple of arguments for update.
        """

        self.Z = np.dot(np.diag(self.Y),np.hstack([self.X,np.ones((self.X.shape[0],1))]))
        self.Z = np.hstack([self.Z,np.diag(np.ones(self.X.shape[0]))])
        self.X = self.Z
        self._Xssq = np.sum(self.X**2, axis=0)


        self.beta = np.zeros(self.X.shape[1])
        self.r = np.ones(self.Y.shape)
        self.nadj = _create_nadj(self.adj)
        self.penalty = self.default_penalty()

    def default_penalty(self):
        """

        """
        d = np.dtype([('l1', np.float),
                      ('l2', np.float),
                      ('l3', np.float),
                      ('delta', np.float)])
        return np.zeros((), d)

    def trim_beta(self):
        #Remove extra coefficients from infimal convolution
        return self.beta[range(self.X.shape[1]-self.X.shape[0])]
        #return self.beta[range(self.X.shape[1]+10)]

    def num_coefs(self):
        #Return the number of total coefficients, and the number in the original problem
        return self.X.shape[1], self.X.shape[1]-self.X.shape[0]

    def update(self, active, nonzero, permute=False):
        """
        Update coefficients in active set, returning
        nonzero coefficients.
        """
        if permute:
            active = np.random.permutation(active)
        print active.shape
        return _update_huber_svm(active,
                                 self.penalty,
                                 nonzero,
                                 self.beta,
                                 self.r,
                                 self.X,
                                 self._Xssq,
                                 self.adj,
                                 self.nadj)


def _update_huber_svm(np.ndarray[DTYPE_int_t, ndim=1] active,
                      penalty,
                      nonzero,
                      np.ndarray[DTYPE_float_t, ndim=1] beta,
                      np.ndarray[DTYPE_float_t, ndim=1] r,
                      np.ndarray[DTYPE_float_t, ndim=2] X,
                      np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                      list adj,
                      np.ndarray[DTYPE_int_t,ndim=1] nadj):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """


    cdef double S, lin, quad, new, db, l1, l1wt, l2, l2wt, l3, maxwt
    cdef long q, n, p

    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])

    cdef double delta = float(penalty['delta'])
    q = active.shape[0]
    n = X.shape[0]
    p = X.shape[1]


    cdef long i,j
    for j in range(q):
        i = active[j]
        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i],r)

        if i < p-n-1:
            #The actual coefficients
            l1wt = l1
            l2wt = l2
            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            maxwt = 0.
        elif i == p-n-1:
            #The intercept
            l1wt = 0.
            l2wt = 0.
            lin = 0.
            quad = 0.
            maxwt = 0.
        else:
            #The extra infimal convolution coefficients
            l1wt = 0.
            l2wt = 0.
            lin = 0.
            quad = 0.
            maxwt = 1.

        new = _solve_plinmax_svm(Xssq[i]/(2.*delta) + l3*quad/2. + l2wt/2.,
                          -S/delta + l3*lin/2.,
                          l1wt,
                          maxwt)

        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new




def _update_lasso(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  np.ndarray[DTYPE_float_t, ndim=2] X,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    Optimizes the LASSO penalty

    norm(Y-dot(X,b))**2/2 + penalty*fabs(b).sum()

    as a function of b.

    """
    cdef DTYPE_float_t S, lin, quad, new, db, l1,
    cdef DTYPE_int_t q, i

    l1 = float(penalty['l1'])
    q = active.shape[0]

    for j in range(q):
        i = active[j]
        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i],r)
        new = _solve_plin(Xssq[i]/2,
                          -S,
                          l1)
        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new

def _update_lasso_array(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  np.ndarray[DTYPE_float_t, ndim=2] X,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    Here, penalty is an array of the same shape as beta.

    """

    cdef DTYPE_float_t S, new, db
    cdef DTYPE_int_t q, i
    cdef np.ndarray[DTYPE_float_t, ndim=1] l1

    q = active.shape[0]
    l1 = penalty

    for j in range(q):

        i = active[j]
        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i], r)

        new = _solve_plin(Xssq[i]/2.,
                          -S,
                          l1[i])
        if new != 0:
            nonzero.append(i)

        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new

def _update_enet(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  np.ndarray[DTYPE_float_t, ndim=2] X,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    Optimizes the ENAT penalty

    norm(Y-dot(X,b))**2/2 + penalty[0]*fabs(b).sum() + penalty[1]*norm(b)**2/2
    """

    cdef DTYPE_float_t S, new, db, l1, l2
    cdef DTYPE_int_t q, i
   
    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    q = active.shape[0]
    
    for j in range(q):

        i = active[j]
        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i], r)
        new = _solve_plin(Xssq[i]/2 + l2/2,
                          -S,
                          l1)
        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new


def _update_naive_laplace(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  np.ndarray[DTYPE_float_t, ndim=2] X,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                  list adj,
                  np.ndarray[DTYPE_int_t,ndim=1] nadj):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, i, 


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    q = active.shape[0]

    for j in range(q):
        i = active[j]
        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i],r)

	# JT: an extra factor of 2 is now in the quad term
	# is this right?

        lin, quad = _compute_Lbeta(adj,nadj,beta,i)
        new = _solve_plin(Xssq[i]/2 + l3*quad/2. + l2/2.,
                          -S+l3*lin/2., 
                          l1)
        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new    

def _create_adj(DTYPE_int_t p):
    """
    Create default adjacency list, parameter i having neighbors
    i-1, i+1.
    """
    cdef list adj = []
    adj.append(np.array([p-1,1]))
    for i in range(1,p-1):
        adj.append(np.array([i-1,i+1]))
    adj.append(np.array([p-2,0]))
    return adj

def _create_nadj(list adj):
    """
    Create vector counting the number of neighbors each
    coefficient has.
    """
    cdef np.ndarray[DTYPE_int_t, ndim=1] nadj = np.zeros(len(adj),dtype=DTYPE_int)
    for i in range(len(adj)):
        nadj[i] = len(adj[i])
    return nadj


def _compute_Lbeta(list adj,
                   np.ndarray[DTYPE_int_t, ndim=1] nadj,
                   np.ndarray[DTYPE_float_t, ndim=1] beta,
                   DTYPE_int_t k):
    """
    Compute the coefficients of beta[k] and beta[k]^2 in beta.T*2(D-A)*beta
    """

    cdef double quad_term = nadj[k]
    cdef double linear_term = 0
    cdef np.ndarray[DTYPE_int_t, ndim=1] row
    cdef int i, j

    
    row = adj[k]
    for i in range(row.shape[0]):
        linear_term += beta[row[i]]

    return -2*linear_term, quad_term


cdef DTYPE_float_t _solve_plin(DTYPE_float_t a,
                        DTYPE_float_t b,
                        DTYPE_float_t c):
    """
    Find the minimizer of

    a*x**2 + b*x + c*fabs(x)

    for positive constants a, c and arbitrary b.
    """

    if b < 0:
        if b > -c:
            return 0.
        else:
            return -(c + b) / (2.*a)
    else:
        if c > b:
            return 0.
        else:
            return (c - b) / (2.*a)


cdef DTYPE_float_t _solve_plinmax(DTYPE_float_t a,
                        DTYPE_float_t b,
                        DTYPE_float_t c,
                        DTYPE_float_t d,
                        DTYPE_float_t M):
    """
    Find the minimizer of

    a*x**2 + b*x + c*fabs(x) + d*np.max([0,np.fabs(x)-M])

    for positive constants a, c, d, M and arbitrary b.
    """

    if b < 0:
        if b < -(c+d+2*M*a):
            return -(b+c+d)/(2.*a)
        elif b < -c and b > -c-2*M*a:
            return -(b+c)/(2.*a)
        else:
            return M*np.greater(0,M*a+c+b)
    else:
        if b > c+d+2*M*a:
            return -(b-c-d)/(2.*a)
        elif b > c and b < c+2*M*a:
            return -(b-c)/(2.*a)
        else:
            return -M*np.greater(0,M*a+c-b)

cdef DTYPE_float_t _solve_plinmax_svm(DTYPE_float_t a,
                                      DTYPE_float_t b,
                                      DTYPE_float_t c,
                                      DTYPE_float_t d):
    #Minimize
    # a*(x**2) + b*x + c*np.fabs(x) + d*np.max([0.,x])
    if b > c:
        return (-b+c)/(2.*a)
    elif -(b+c+d)>0:
        return -(b+c+d)/(2.*a)
    else:
        return 0



"""

    if b > 0:
        if b > c:
            return -(b-c)/(2.*a)
        else:
            return 0
    else:
        if b < -(c+d+2*M*a):
            return -(b+c+d)/(2.*a)
        elif b < -c and b > -(c+2*M*a):
            return -(b+c)/(2.*a)
        else:
            return M*np.greater(-(c+M*a),b)

    if b > c + d:
        return (d+c-b)/(2.*a)
    elif b < -c and -(b+c)/(2.*a) > M:
        return -(b+c)/(2.*a)
    elif b < d-c and (d-b-c)/(2.*a) <= M:
        return (d-b-c)/(2.*a)
    else:
        return M*np.greater((d-b-c)/a,M)
"""


def _update_robust_graphnetIC(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  np.ndarray[DTYPE_float_t, ndim=2] X,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                  list adj,
                  np.ndarray[DTYPE_int_t,ndim=1] nadj):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, i, 

    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    delta = float(penalty['delta'])
    q = active.shape[0]
    n = X.shape[0]
    p = X.shape[1]

    

    for j in range(q):
        i = active[j]
        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i],r)

        
        if i < p-n:
            l1wt = l1
            l2wt = l2
            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
        else:
            l1wt = delta
            l2wt = 0.
            lin = 0.
            quad = 0.    

        new = _solve_plin(Xssq[i]/2 + l3*quad/2. + l2wt/2.,
                          -S+l3*lin/2., 
                          l1wt)
        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new    


def _update_robust_graphnet_reweight(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  np.ndarray[DTYPE_float_t, ndim=2] X,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                  list adj,
                  np.ndarray[DTYPE_int_t,ndim=1] nadj):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1,l2, l3
    cdef long q, i, 
    cdef np.ndarray[DTYPE_float_t, ndim=1] l1weights	


    l1 = float(penalty['newl1'])
    print "Penalty newl1:", l1
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    delta = float(penalty['delta'])
    l1weights = penalty['l1weights'] 
    q = active.shape[0]
    n = X.shape[0]
    p = X.shape[1]
    

    for j in range(q):
        i = active[j]

        if i < p-n:
            t = l1weights[i] != np.inf

        if i < p-n and t:
            l1wt = l1weights[i]*l1
            l2wt = l2
            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            #else:
        elif i >= p-n:
            t = True
            l1wt = delta
            l2wt = 0.
            lin = 0.
            quad = 0.    

        if t:
            S = beta[i] * Xssq[i]
            S += np.dot(X[:,i],r)
            new = _solve_plin(Xssq[i]/2 + l3*quad/2. + l2wt/2.,
                              -S+l3*lin/2., 
                              l1wt)
        else:
            new = 0

        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new    

"""
    l1 = penalty[0]
    l2 = penalty[1]
    l3 = penalty[2]
    delta = penalty[3]
    q = active.shape[0]
    
    n = X.shape[0]
    p = X.shape[1]
    

    for j in range(q):
        i = active[j]

        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i],r)        
        if i < p-n:
            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            l1wt = l1
            #S = beta[i] * Xssq[i]
            #S += np.dot(X[:,i],r)        

        else:
            lin = 0.
            quad = 0.
            l1wt = delta
            #S = beta[i] + r[i-n]
        #print i, S

        new = _solve_plin(Xssq[i]/2 + l3*quad/2. + l2/2.,
                          -S+l3*lin/2., 
                          l1wt)
       
        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new    
"""

def _update_robust_graphSVM(np.ndarray[DTYPE_int_t, ndim=1] active,
                  penalty,
                  nonzero,
                  np.ndarray[DTYPE_float_t, ndim=1] beta,
                  np.ndarray[DTYPE_float_t, ndim=1] r,
                  np.ndarray[DTYPE_float_t, ndim=2] X,
                  np.ndarray[DTYPE_float_t, ndim=1] Xssq,
                  list adj,
                  np.ndarray[DTYPE_int_t,ndim=1] nadj):
    """
    Do coordinate-wise update of lasso coefficients beta
    in active set.

    The coordinates that are nonzero are stored in the
    list nonzero.

    """
    cdef double S, lin, quad, new, db, l1, l2, l3
    cdef long q, i, 


    l1 = float(penalty['l1'])
    l2 = float(penalty['l2'])
    l3 = float(penalty['l3'])
    delta = float(penalty['delta'])   
    q = active.shape[0]
    n = X.shape[0]
    p = X.shape[1]

    for j in range(q):
        i = active[j]
        S = beta[i] * Xssq[i]
        S += np.dot(X[:,i],r)

        
        if i < p-n:
        #if i < p-n-1:
            l1wt = l1
            l2wt = l2
            lin, quad = _compute_Lbeta(adj,nadj,beta,i)
            maxwt = 0
        else:
        #elif i < p-1:
            l1wt = 0.
            l2wt = 0.
            lin = 0.
            quad = 0.
            maxwt = 1.
        #else:
        #    l1wt = 0.
        #    l2wt = 0.
        #    lin = 0.
        #    quad = 0.
        #    maxwt = 0.


        new = _solve_plinmax(Xssq[i]/(2.*delta) + l3*quad/2. + l2wt/2.,
                          -S/delta + l3*lin/2., 
                          l1wt,
                          maxwt,
                          1.-delta)
        if new != 0:
            nonzero.append(i)
        db = beta[i] - new
        r += db * X[:,i]
        beta[i] = new    



