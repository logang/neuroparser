#!/usr/bin/env python
# encoding: utf-8
# filename: profile.py
# cython: profile=True

# import profiling stuff
import pstats, cProfile

# cython functions are not yet seen as packages. import path. 
import sys, os, time
path_to_cython_packages = os.path.abspath('../.')
sys.path.append(path_to_cython_packages)

# scons still building packges funny, get path to graphs
path_to_graphs = os.path.abspath('../../graphs/.')
sys.path.append(path_to_graphs)

# import major libraries
import numpy as np
import time

# local imports
# import testR
import cwpath, graphnet, strategy
from graph_laplacian import construct_adjacency_list
from test_graphnet import train_all, test_graphnet, get_lambda_max

#-------------------------------------------------------------------------------------------------------------

def profile_test_graphnet():
    # get training data and constants
    Data = np.load("Data.npz")
    X = Data['X'][0:1000,:]
    Y = Data['Y'][0:1000]
    G = None
    lam_max = get_lambda_max(X,Y)
    cwpathtol = 1e-6
    
    # penalty grid values
    l1vec = np.linspace(0.95*lam_max, 0.0001*lam_max, num=100).tolist()
    results, problemkey = test_graphnet(X, Y, G, l1vec, 1e4, 1e6, 0.1, -999.0,initial=None,tol=cwpathtol,scipy_compare=False)

def profile_cwpath_robust_graphnet():
    # get training data and constants
    Data = np.load("Data.npz")
    X = Data['X'][0:1000,:]
    print "Data matrix size:",X.shape
    Y = Data['Y'][0:1000]
    nx = np.sqrt(X.shape[1])
    ny = np.sqrt(X.shape[1])
    A = construct_adjacency_list(nx,ny,1)
    lam_max = get_lambda_max(X,Y)
    tol = 1e-6
    initial=None

    # choose penalty grid
    l1 = np.linspace(4*lam_max, 0.2*lam_max, num=100)
    l2 = 100.
    l3 = 1000.
    delta = 1.0

    # setup problem
    problemtype = graphnet.RobustGraphNet
    problemkey = 'RobustGraphNet'
    print "Robust GraphNet with penalties (l1, l2, l3, delta)", l1, l2, l3, delta
    l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial) #,initial_coefs=np.array([14.]*10))
    l.problem.assign_penalty(path_key='l1',l1=l1,l2=l2,l3=l3,delta=delta)
    coefficients, residuals = l.fit(tol=tol, initial=initial)



if __name__ == "__main__":

#    cProfile.runctx("train_all()", globals(), locals(), "Profile.prof")
    cProfile.runctx("profile_cwpath_robust_graphnet()", globals(), locals(), "Profile.prof")

    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
