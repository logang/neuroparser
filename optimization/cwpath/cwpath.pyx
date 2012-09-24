#!/usr/bin/env python
# encoding: utf-8
# filename: cwpath.pyx
# cython: profile=True

"""
------------------------------------------------------------------------------------------------------------------------------

This optimization module implements large scale 'Pathwise coordinate optimization'
techniques for graph-constrained sparse linear models using active set methods, warm starts, 
SAFE/STRONG rules for variable screening, and infimal convolution, as described in 

Friedman J., Hastie, T., Hofling, H., and Tibshirani, R. Pathwise coordinate optimization. Annals of Applied Statistics, 2007. 
[other active set refs]
[SAFE]
[STRONG]
[Rockafeller]
[our paper]

------------------------------------------------------------------------------------------------------------------------------

The optimization problems solved by the CoordWise class are determined 
by the following callables:

CoordWise.update(active, penalty, nonzero, *problem_args):

CoordWise.initialize(data):
  return initial value of *problem_args based on problem data

CoordWise.stop(olddata, newdata): 
  convergence check comparing current candidate to old candidate.

CoordWise.output(*problem_args): 
  return the "intersting part" of the arguments, i.e. the parts to be stored once the algorithm has found a solution.

CoordWise.copy(*problem_args): 
  return a copy of output(*problem_args) to compare to a new solution.

------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np
cimport numpy as np
import time, copy

## Compile-time datatypes
DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

# should we replace this with cython?
from numpy.core.umath_tests import inner1d

# ----------------------------------------------------------------------------------------------------------------------------

class CoordWise(object):
    """
    Solve a (nonsmooth) convex regression minimization problem using
    pathwise coordinate descent.

    Inputs:
    -------
	-- data: a tuple (X,Y,G), where X are the independent regression variables ("inputs", "features"), 
	   and Y the dependent variable ("output", "target"). G is an optional sparse graph (such as a graph Laplacian) 
	   used in the regularized regression, and represented as a list of lists of indices of adjacent voxels.  
	-- problemtype: a problem object, such as graphnet.GraphNet, that takes data and and optional initial set of coefficients.
	-- strategy: s
	-- penalty: A single set of penalty values for the penalized regression problem, or a list of penalties if 
	   problem is to be solved over a path of penalty parameters (recommended). 
	-- initial_coefs: Defaults to None. 
	-- debug: a flag specifying whether debugging information should be printed to screen. Defaults to False.
    """

    def __init__(self, data, problemtype, strategy=None, penalty=None, initial_coefs=None, debug=False):
        self.data = data
        self.problem = problemtype(data,initial_coefs)
        self.strategy = strategy or self.problem.default_strategy()
        self.debug = debug
#        self.initial_coefs = initial_coefs
#        self.screen_type = "STRONG"
        self.screen_type = "all"
        self.KKT_checking_ON = False

   # ---- Main update and fitting methods ---- #

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
        if self.debug:
            print "Active set:", active.astype(np.int)

	# Update the active set, permuting the coordinate cycle order if permute is True.
	# Return the nonzero coefficients.  
        self.problem.update(active.astype(np.int), nonzero, permute=permute)
        return np.asarray(nonzero)

    def fit(self, penalty=None, active=None, initial=None, tol=1e-6, repeat_update=1, refit=False, debug=False):
        """
        Fit a pathwise coordinate optimization model with initial estimates
        and a guess at the active set.

        It applies an optional initial strategy, if supplied (see strategy.py).
        """
        if debug:
            print "Coefficients descriptive statistics:"
            print "\t--> Standard deviation:", np.std(self.problem.coefficients)
            print "\t--> Range: [",np.min(self.problem.coefficients),np.max(self.problem.coefficients)," ]"
            print "\t--> Mean:", np.mean(self.problem.coefficients) 
            print "\t--> Median:", np.median(self.problem.coefficients) 
            print "\t--> Mode:", np.mode(self.problem.coefficients) 

	# keep copy of penalty around
        # penalty = self.problem.penalty.copy()

        if penalty is not None:
            self.problem.penalty = penalty

#        else:	
#            raise ValueError("No penalty supplied.")

        # Check if there is a list of penalties to be used as a path.
	# If there is not, make the given value a list with one element,
	# otherwise, use the given list. 
        if type(self.problem.penalty)!=type([]):
            penalties = [self.problem.penalty]
        else:
            penalties = self.problem.penalty

	# initialize conv, the worst difference ratio between 
	# sequential estimates of nonzero coefficients  
        self.conv = 10.    
	
	# declare path length
        cdef long path_length = len(penalties)

	# ---- main loop over penalties ---- #
        # main fitting loop for a particular set of penalites
        # and eligible set
        # Does:
        # (1) Set penalties
        # (2) Run "repeat_update" updates of active set
        # (3) Check for convergence of active set. 
        # (4) If KKT_checking_ON is True:
        #     If active set converged, check coordinate-wise for violations of KKT conditions in the eligible set.
        #     Add any violations to active set and repeat (2)-(3) until KKT conditions are not violated.
        # (5) If KKT_checking_ON is True:
        #     Check coordinate-wise for violations of KKT conditions in the entire set of variables. 
        #     If KKT conditions satisfied, proceed to next fit in path. 
        #     Otherwise, add violations to active set and repeat (2)-(4).

	# initialize
        coefficients = []
        residuals = []
        cdef long i
        cdef float old_tol = tol
        cdef float alpha = 1.0

	# loop over penalty path
        for i in range(path_length):

            # inner loop initialization
            self.fitit = 0
            self.updateit = 0
            tol = old_tol

            # set penalties
            self.problem.penalty = penalties[i]
            self.lam_max = self.get_lambda_max()

            # set starting active set as eligible set
            if i == 0:
                if self.KKT_checking_ON:
                    self.eligible = []
                    while len(self.eligible) < 1:
                        self.eligible = self._get_eligible_set( alpha*self.lam_max, penalties[i]['l1'], "STRONG", self.problem.penalty )
                        alpha *= 0.99
                        print alpha
                    self.active = self.eligible.copy()
                else:
                    if initial is None:
                        print "\t---> Initial pass through all coefficients..."
                        self.active = self.update( np.array(range(self.problem.X.shape[1])) )
                        print "\t---> Done with initial pass."
                    else:
                        print "\t---> Setting active set to last solution in path."
                        self.active = initial
            else:
                print "\t---> Path index", i
                if self.KKT_checking_ON:
                    self.eligible = self._get_eligible_set(penalties[i-1], penalties[i]['l1'], self.screen_type, self.problem.penalty)
                    self.active = self.eligible.copy()

	    # if debugging print problem size
            if self.debug:
               print "Problem size:", self.problem.X.shape

            while True:
	    	# keep count of fit iterations
                self.fitit += 1

		# store the last problem solution
                old = self.problem.copy()

		# repeat update on just the current active variables
                for rep in xrange(repeat_update):
                    self.active = self.update(active=self.active)
		
		# Check for convergence of active set, and if finished set 'finished' to True
		# If return_worst is True, return largest difference ratio between an active variable 
		# on this and the previous iteration.
                finished, worst = self.problem.stop(old,tol=tol,return_worst=True)
                if finished:
                    print "\t---> Active set of size", len(self.active), "converged."
 
                # If active set has converged, check KKT conditions for eligible set for SAFE/STRONG.
                # If there are no violations on the eligible set, run KKT on all variables. If there
                # is a violation in either case, add the violating variables to the active set and continue
                # updates (set finished=False). 
                if finished and self.KKT_checking_ON:
                    if self.screen_type == "STRONG" or self.screen_type == "SAFE":
                        KKT_violations, KKT_violvals, eligible = self.check_KKT("eligible")
                        if len(KKT_violations) > 1:
                            print "KKT violations:", KKT_violations
                            print "KKT violation values:", KKT_violvals
                            self.active = np.unique( np.append( self.active, KKT_violations ))
                            print "New active set:", self.active
                            finished = False
                        else:
                            print "No KKT violations on eligible set", eligible
                            KKT_violations, KKT_violvals, eligible = self.check_KKT("all")
                            if len(KKT_violations) > 1:
                                print "KKT violations:", KKT_violations
                                print "KKT violation values:", KKT_violvals
                                self.active = np.unique( np.append( self.active, KKT_violations ))
                                print "New active set:", self.active
                                finished = False
                            else:
                                print "No KKT violations on full variable set."
                elif finished and not self.KKT_checking_ON:
                       self.current_active = self.active.copy()
                       self.active = np.array( range(self.problem.X.shape[1]) )
                       self.active = self.update(self.active)
                       if len( np.setdiff1d(self.current_active, self.active) ) > 0:
                           print "Added ", np.setdiff1d(self.current_active, self.active), " to active set."
                           finished = False

                if self.debug:
                    print '\tFit iteration: %d, Number active: %d, Max relative change: %g' % (self.fitit, self.active.shape[0], worst) 
#                  print '\tFit iteration: %d, Greedy Iteration: %d, Number active: %d, Max relative change: %g' % (self.fitit, self.greedyit, self.active.shape[0], worst)

		# Store worst difference ratio between nonzero coefficients
                self.conv = worst

		# if finished is true or if the maximum number of fit iterations has been reached, break.
                if finished or self.fitit > self.strategy.max_fitit:
                    break

	    # collect results
            coefficients.append(self.current[0])
            residuals.append(self.current[1])

        self.problem.penalty = penalty
        return coefficients, residuals

    def check_KKT(self, KKT_type="all", conv_eps=0.1):
        if KKT_type == "all":
            eligible = np.setdiff1d( np.array(range( self.problem.X.shape[1] )), self.active)
#            eligible = [i for i in np.array(range( self.problem.X.shape[1] )) if i not in self.active]
            print "KKT type: all"
        elif KKT_type == "eligible":
            eligible = np.setdiff1d( self.eligible, self.active)
#            eligible = [i for i in self.eligible if i not in self.active] 
            print "KKT type: eligible."
        if len(eligible) > 0:
            # check KKT on eligible set
            beta_hat, r = self.current
            n = self.problem.X.shape[0]
            p = self.problem.X.shape[1]
            X = self.problem.X[:,eligible]
            y = self.data[1]
            G = [self.data[2][i] for i in eligible]
            l1 = self.problem.penalty['l1']; l2 = self.problem.penalty['l2']; l3 = self.problem.penalty['l3']
            resid_prod = inner1d(X.T,r)/n
            pen_prod =  l3*np.array([ len(G[i])*beta_hat[eligible[i]] - 0.5*np.sum(beta_hat[G[i]]) for i in xrange(len(G))]) #/p
            subgrad = l1*np.sign(beta_hat[eligible])
            print "subgrad, resid_prod, pen_prod", subgrad, resid_prod, pen_prod
            print "Resid prod shape",resid_prod.shape, pen_prod.shape, subgrad.shape
            KKT = subgrad + resid_prod + pen_prod
            idx = np.where( np.fabs(KKT) > conv_eps )[0]
            return idx, KKT[idx], eligible
        else:
            return [],[],[]

    def get_lambda_max(self):
        """ 
        Find the value of lambda at which all coefficients are set to zero
        by finding the minimum value such that 0 is in the subdifferential
        and the coefficients are all zero.
        """
        subgrads = np.fabs( inner1d(self.problem.X.T, self.data[1]) )
        return np.max( subgrads )

    # ---- Methods for getting results and coefficients ---- #

    def _getcurrent(self):
        return self.problem.output()
    current = property(_getcurrent)

    def _get_eligible_set(self, lam_max, lam, type="STRONG", penalties=None):
        if type != "all":
            if self.fitit == 0:
               resids = self.data[1]
            else:
               _, resids = self.current
        if type == "STRONG":
           if lam_max == None or lam == None:
              raise ValueError("Lambda parameters not given.")
           eligible = self.strategy.STRONG(lam_max,lam,resids,self.problem.X) 
        elif type == "SAFE":
           if lam_max == None or lam == None:
              raise ValueError("Lambda parameters not given.")
           eligible = self.strategy.SAFE(lam_max,lam,resids,self.problem.X)
        elif type == "all":
           eligible = self.strategy.all()
        else:
           raise ValueError("Strategy type does not exist")
        return eligible
	  
    def _getresults(self):
        """
        Function to strip extra coefficients as necessary
        """
        if hasattr(self.problem,'get_coefficients'):
            return self.problem.coefficients,self.current[1]
        else:
            return self.current
    results = property(_getresults)

    def _get_num_coefs(self):
        """
        Function to strip extra coefficients as necessary
        """
        if hasattr(self.problem,'num_coefs'):
            return self.problem.num_coefs()
        else:
            return None
    num_coefs = property(_get_num_coefs)

    # ---- Methods for choosing sets of variables to search over ---- #

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
        
# ----------------------------------------------------------------------------------------------------------------------------
# EOF
