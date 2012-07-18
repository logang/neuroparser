import numpy as np
cimport numpy as np
import time

## Local imports

## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

"""
Functions specific to regression problems.
stop: convergence criterion based on residuals.
"""

import numpy as np
import strategy

class Regression(object):

    npath = 0

    def __init__(self, data, initial_coefs=None):
        
        self.X, self.Y = [np.asarray(x) for x in data]
        self.r = self.Y.copy()
        self.initial_coefs = initial_coefs
        self.beta = np.zeros(self.X.shape[1])

        self.initialize()

    def coefficientCheck(self,
                         np.ndarray[DTYPE_float_t, ndim=1] bold,
                         np.ndarray[DTYPE_float_t, ndim=1] bnew,
                         DTYPE_float_t tol):

        #Check if all coefficients have relative errors < tol

        cdef long N = len(bold)
        cdef long i,j

        for i in range(N):
            if bold[i] == 0.:
                if bnew[i] != 0.:
                    return False
            if np.fabs(np.fabs(bold[i]-bnew[i])/bold[i]) > tol:
                return False
        return True

    def coefficientCheckVal(self,
                         np.ndarray[DTYPE_float_t, ndim=1] bold,
                         np.ndarray[DTYPE_float_t, ndim=1] bnew,
                         DTYPE_float_t tol):
        
        #Check if all coefficients have relative errors < tol
        
        cdef long N = len(bold)
        cdef long i,j
        cdef DTYPE_float_t max_so_far = 0.
        cdef DTYPE_float_t max_active = 0.
        cdef DTYPE_float_t ratio = 0.

        for i in range(N):
            if bold[i] ==0.:
                if bnew[i] !=0.:
                    max_so_far = 10.
            else:
                ratio = np.fabs(np.fabs(bold[i]-bnew[i])/bold[i])
                if ratio > max_active:
                    max_active = ratio

        if max_active > max_so_far:
            max_so_far = max_active

        return max_so_far < tol, max_active

            
    def initialize(self):
        """
        Abstract method for initialization of regression problems.
        """
        pass

    def stop(self,
             previous,
             DTYPE_float_t tol=1e-4,
             DTYPE_int_t return_worst = False):
        """
        Convergence check: check whether 
        residuals have not significantly changed or
        they are small enough.

        Both old and current are expected to be (beta, r) tuples, i.e.
        regression coefficent and residual tuples.
    
        """


        cdef np.ndarray[DTYPE_float_t, ndim=1] bold, bcurrent
        bold, _ = previous
        bcurrent, _ = self.output()

        if return_worst:
            status, worst = self.coefficientCheckVal(bold, bcurrent, tol)
            if status:
                return True, worst
            return False, worst
        else:
            status = self.coefficientCheck(bold, bcurrent, tol)
            if status:
                return True
            return False


    def output(self):
        """
        Return the 'interesting' part of the problem arguments.
        
        In the regression case, this is the tuple (beta, r).
        """
        return self.coefficients, self.r

    def final_stop(self, previous, tol=1.0e-5):
        """
        Need a better way to check this?
        """
        if self.npath > 500:
            return True
        self.npath += 1
##         Y = self.Y
##         beta, r = [np.asarray(x) for x in previous]
##         R2 = (r**2).sum() / (Y**2).sum()
##         if R2 < tol:
##             return True

##         R2 = ((r - self.r)**2).sum() / (self.r**2).sum()
##         if R2 < tol:
##             return True

        return False
        
    def copy(self):
        """
        Copy relevant output.
        """

        cdef np.ndarray[DTYPE_float_t, ndim=1] coefs, r
        coefs, r = self.output()
        return (coefs.copy(), r.copy())

    def update(self, active, nonzero):
        """
        Update coefficients in active set, returning nonzero coefficients.

        Abstract method for update step.
        """
        raise NotImplementedError
    
    def initial_strategy(self):
        """
        Initial strategy in the pathwise search.
        """
        return strategy.NStepBurnin(self.total_coefs, nstep=8, burnin=2)
    
    def default_strategy(self):
        """
        Default strategy.
        """
        return strategy.Strategy(self.total_coefs)

    def default_active(self):
        """
        Default active set.
        """
        return np.arange(self.total_coefs)

    def default_penalty(self):
        """
        Abstract method for default penalty.
        """
        raise NotImplementedError

    def assign_penalty(self, path_key=None, **params):
        """
        Abstract method for assigning penalty parameters.
        """
        if path_key is None:
            path_length = 1
        else:
            path_length = len(params[path_key])
        penalty_list = []
        for i in range(path_length):
#            penalty = self.penalty.copy()
            penalty = dict()
            for key in params:
                if key==path_key:
                    penalty[key] = params[key][i]
                else:
                    penalty[key] = params[key]
            penalty_list.append(penalty)
        if path_length == 1:
            penalty_list = penalty_list[0]
        self.penalty = penalty_list    


