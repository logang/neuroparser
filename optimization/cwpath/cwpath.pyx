import numpy as np
cimport numpy as np

## Local imports


## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t








"""
This module implements large scale 'pathwise coordinate optimization'
techniques, as described in 

Hasti, paper here

The optimization problems are determined by the following callables:

update(active, penalty, nonzero, *problem_args):

initialize(data):
  return initial value of *problem_args based on problem data

stop(olddata, newdata): convergence check comparing current candidate to old candidate.

output(*problem_args): return the "intersting part" of the arguments, i.e. the parts to be stored once the algorithm has found a solution

copy(*problem_args): return a copy of output(*problem_args) to compare to a new solution


"""

import numpy as np
import time, copy


class CoordWise(object):

    """
    Solve a convex regression minimization problem using
    coordinate wise descent.

    """

    def __init__(self, data, problemtype, strategy=None, penalty=None, initial_coefs=None):
        self.data = data
        self.problem = problemtype(data,initial_coefs)
        self.strategy = strategy or self.problem.default_strategy()
        #self.initial_coefs = initial_coefs

    def update(self, active=None, permute=False):
        """
        Go through each coordinate in "active" set,
        updating coefficients beta and residual r.
        
        Returns:
        --------

        nonzero: coordinates not set to zero in update

        """

        if active is None:
            active = self.active

        nonzero = []
        #print active.astype(np.int)
        self.problem.update(active.astype(np.int), nonzero, permute=permute)

        return np.asarray(nonzero)

    def _getcurrent(self):
        return self.problem.output()
    current = property(_getcurrent)

    def _getresults(self):
        """
        XXX (brad): New function to strip extra coefficients as necessary
        """
        if hasattr(self.problem,'get_coefficients'):
            return self.problem.coefficients,self.current[1]
        else:
            return self.current
    results = property(_getresults)

    def _get_num_coefs(self):
        """
        XXX (brad): New function to strip extra coefficients as necessary
        """
        if hasattr(self.problem,'num_coefs'):
            return self.problem.num_coefs()
        else:
            return None
    num_coefs = property(_get_num_coefs)


    def fit(self, penalty=None, active=None, initial=None, tol=1e-4, refit=False):
        """
        Fit a pathwise coordianate optimization model with initial estimates
        and a guess at the active set.

        It applies an optional initial strategy, if supplied.

        """
        #print "B", self.problem.coefficients, np.std(self.problem.coefficients), np.max(self.problem.coefficients)
        original_penalty = self.problem.penalty.copy()
        if penalty is not None:
            self.problem.penalty = penalty
        #Check if there is a list of penalties for a path

        if type(self.problem.penalty)!=type([]):
            #Create a list of length one
            penalties = [self.problem.penalty]
        else:
            penalties = self.problem.penalty

        self.conv = 10.    

        cdef long path_length = len(penalties)

        if not refit:
            if initial is not None:
                strategy, self.strategy = self.strategy, initial

            if active is None:
                self.active = self.problem.default_active()
            else:
                self.active = active

        results = []
        cdef long i
        cdef float old_tol = tol
        for i in range(path_length):
            self.fitit = 0
            self.updateit = 0
            tol = old_tol

            self.problem.penalty = penalties[i]

            self.greedy(tol=tol)

            if initial is not None:
                self.strategy = strategy

            #print "Problem size:", self.problem.X.shape
            while True:
                self.fitit += 1
                self.greedy(tol=tol)
                old = self.problem.copy()
                active = self.update(active=self.strategy.all())
                self.active = self.update(active=active)
                finished, worst = self.problem.stop(old,tol=tol,return_worst=True)
                print '\tFit iteration: %d, Greedy Iteration: %d, Number active: %d, Max relative change: %g' % (self.fitit, self.greedyit, self.active.shape[0], worst)
                self.conv = worst
                if finished or self.fitit > self.strategy.max_fitit:
                    break
            results.append(self.current)
        self.problem.penalty = original_penalty
        return results
    
    def greedy(self, active=None, tol=None, DTYPE_int_t min_iter=5):
        """
        Greedy algorithm: update active set at each iteration
        using a given strategy
        and continue until convergence.
        """
        
        self.greedyit = 0
        if active is not None:
            self.active = active
    
        while True:
            self.greedyit += 1
            old = self.problem.copy()
            nactive = self.update()
            if self.greedyit >= min_iter:
                if self.problem.stop(old,tol=tol) or self.greedyit > self.strategy.max_greedyit:
                    break
            self.active = self.strategy(self.greedyit, self.active, nactive)
            
        return self.current

    def append(self, value):
        self.results.append(value)
        
