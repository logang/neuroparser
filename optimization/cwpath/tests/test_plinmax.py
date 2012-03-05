
import numpy as np
from cwpath import cwpath, lasso
import scipy.optimize
from nose.tools import *

import testR

def setup():
    testR.setup()

def plin(a,b,c):

    if b > 0.:
        if b > c:
            return (c-b)/(2*a)
        else:
            return 0
    else:
        if b < -c:
            return -(b+c)/(2*a)
        else:
            return 0


def plinmax(a,b,c,d,M):


    if b < 0:
        if b < -(c+d+2*M*a):
            return -(b+c+d)/(2.*a)
        elif b < -c and b > -c-2*M*a:
            return -(b+c)/(2.*a)
        else:
            print "A"
            return M*np.greater(0,M*a+c+b)
    else:
        if b > c+d+2*M*a:
            return -(b-c-d)/(2.*a)
        elif b > c and b < c+2*M*a:
            return -(b-c)/(2.*a)
        else:
            print "B"
            return -M*np.greater(0,M*a+c-b)
            
            

"""
    if b > 0:
        if b > c:
            return -(b-c)/(2.*a)
        else:
            return 0
    else:
        if b < -(c+d+2*M*a):
            return -(b+c+d)/(2.*a)
        if b < -c and b > -(c+2*M*a):
            return -(b+c)/(2.*a)
        else:
            print "AAAAAAA"
            return M*np.greater(-(c+M*a),b)
"""
"""
    if b > c + d:
        return (d+c-b)/(2.*a)
    elif b < -c and -(b+c)/(2.*a) > M:
        return -(b+c)/(2.*a)
    elif b < d-c and (d-b-c)/(2.*a) <= M:
        return (d-b-c)/(2.*a)
    else:
        return M*np.greater((d-b-c)/a,M)
"""


def test_plinmax():

    N = 500
    for i in range(N):
        a,b,c,d,M = np.random.normal(0,10,5)
        a = np.fabs(a)
        c = np.fabs(c)
        d = np.fabs(d)
        M = np.fabs(M)/10
        print a,b,c,d,M

        def f(x):
            #return a*(x**2) + b*x + c*np.fabs(x)
            #return a*(x**2) + b*x + c*np.fabs(x) + d*np.max([0.,M-x])
            return a*(x**2) + b*x + c*np.fabs(x) + d*np.max([0.,np.fabs(x)-M])
        

        v = scipy.optimize.fmin_powell(f, np.zeros(1), ftol=1.0e-10, xtol=1.0e-10)
        v = np.asarray(v)
        #print v#, f(v), f(plinmax(a,b,c,d,M))
        #print np.fabs(v-plinmax(a,b,c,d,M))
        print v
        assert(np.fabs(v-plinmax(a,b,c,d,M))<1e-3)

def test_objective(N=10):
        
    for i in range(N):
        
        delta = np.random.uniform(0,1)
        r = np.random.normal(0,15)
        
        def huber(r):
            r = np.fabs(r)
            t1 = np.greater(r, 1-delta)
            t2 = np.greater(r, 1)
            return t2*(r-1+delta/2)+ (1-t2)*t1*( (r-1+delta)**2/(2*delta) )
        
        
        def f(b):
            return (r-b)**2/(2*delta) + np.max([0.,np.fabs(b)-1+delta])
        
        
        v = scipy.optimize.fmin_powell(f, np.zeros(1), ftol=1.0e-12, xtol=1.0e-12)
        v = np.asarray(v)
        print v, f(v), huber(r)
        assert( np.fabs(f(v)-huber(r)) < 1e-5)


def test_robust_objective(N=1):

    for i in range(N):
        
        delta = np.random.uniform(0,1)
        r = np.random.normal(0,150)

        def huber(r):
            r = np.fabs(r)
            t = np.greater(r, delta)
            return ((1-t)*r**2 + t*(2*delta*r - delta**2))/2

        def f(b):
            return (r-b)**2/2 + delta*np.fabs(b)
        
        
        v = scipy.optimize.fmin_powell(f, np.zeros(1), ftol=1.0e-12, xtol=1.0e-12)
        v = np.asarray(v)
        print v, f(v), huber(r)
        assert( np.fabs(f(v)-huber(r)) < 1e-5)
