"""
Strategies for changing active set based on current iteration,
the current active set and a candidate set of non-zero coefficients.
"""
import numpy as np
import scipy.linalg as la

# should we replace this with cython?
from numpy.core.umath_tests import inner1d

class Strategy:

    """
    Strategy for choosing active set for p
    variables, given current active set and a 
    candidate active set consisting of all coordinates
    that were updated in coordinate wise algorithm.

    Information on the current iteration the algorithm
    is on may also be used.
    """

    max_greedyit = np.inf
    max_fitit = 1e4

    def __init__(self, p):
        """
        Default strategy is to include all variables in
        the active set.
        """
        self.p = p

    def __call__(self, iteration, current, candidate):
        return self.all()

    def all(self):
        """
        All variables are active.
        """
        return np.arange(self.p)

    def SAFE(self, lam_max, lam, y, X):
        """
        Screen variables using the SAFE rule.
        """
        resid_prod = np.fabs( inner1d(X.T,resid) )
        idx = resid_prod >= lam - la.norm(X[:,i])*la.norm(y)*((lam_max-lam)/lam_max)
        return np.where(idx)[0]

    def STRONG(self, lam_max, lam, resid, X):
        """
        Screen variables using the STRONG rule.
        """
        resid_prod = np.fabs( inner1d(X.T,resid) )
        idx = resid_prod >= 2*lam_max - lam
        return np.where(idx)[0]

class NStep(Strategy):

    __doc__ = Strategy.__doc__    

    def __init__(self, p, nstep=5):
        """
        Update the active set active set with the candidate
        if iteration % nstep == 0.
        """

        self.p = p
        self.nstep = nstep

    def __call__(self, iteration, current, candidate):
        if iteration % self.nstep == 0:
            current = np.asarray(candidate)
        return np.asarray(current)

class NStepBurnin(Strategy):

    __doc__ = Strategy.__doc__
    
    def __init__(self, p, nstep=5, burnin=1):
        """
        Update the active set with the candidate
        if it % self.nstep == 0, unless it==self.burnin, in which
        case also return the candidate.

        Implicitly assumes that the
        initial active set is "large", and one update is
        enough to get a very good idea of the active set.
        
        Further iterations, can still drop variables from the active set
        after every nstep iterations.
        
        """
        self.p = p
        self.nstep = nstep
        self.burnin = burnin
        if burnin >= nstep:
            raise ValueError, 'expecting burnin < nstep'

    def __call__(self, it, current, candidate):
        if it % self.nstep == 0 or it == self.burnin:
            current = np.asarray(candidate)
        return np.asarray(current)
