# cython functions are not yet seen as packages. import path. 
import sys, os
path_to_cython_packages = os.path.abspath('../.')
sys.path.append(path_to_cython_packages)

# scons still building packges funny, get path to smoothers
path_to_smoothers = os.path.abspath('../../smoothers/.')
sys.path.append(path_to_smoothers)

# import major libraries
import numpy as np
import scipy.optimize
from nose.tools import *
import time, h5py

# for plotting 
import matplotlib
matplotlib.use('agg')
import pylab as pl
pl.ion()

# local imports
import testR
import cwpath, graphnet, strategy
from graph_laplacian import construct_adjacency_list

# functions
from numpy.core.umath_tests import inner1d
from multiprocessing import Pool

# setup some R stuff
def setup():
    testR.setup()

#------------------------------------------------------------------------------------------------------#
# Run all GraphNet tests

def train_all(num_l1_steps=100,test_imap=False):
    # get training data and constants
    Data = np.load("Data.npz")
    X = Data['X'][0:1000,:]
    Y = Data['Y'][0:1000]
    G = [None]
    lam_max = get_lambda_max(X,Y)
    cwpathtol = 1e-6
    
    # penalty grid values
    l1vec = np.linspace(0.95*lam_max, 0.0001*lam_max, num=num_l1_steps).tolist()

    # # test grid
    l2vec = [0.0, 1e6]
    l3vec = [1e6, 1e12] # [0.0, 100, 1e6]
    deltavec = [-999.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1e10] #, 0.1, 0.5, 1., 1e10]
    svmdeltavec = [-999.0] #, 1]

    # big grid
    # l2vec = [0.0, 1.0, 10, 100, 1000, 1e4, 1e6, 1e8]
    # l3vec = [0.0, 1.0, 10, 100, 1000, 1e4, 1e6, 1e8]
    # deltavec = [-999.0, 0.25, 0.5, 1.0, 10, 100]
    # svmdeltavec = [-999.0, 0.25, 0.5, 1.0, 10, 100]

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
    if test_imap:
        in_tuple = (X,Y,G,(100,1e10,-999.0,-999.0),500,lam_max,80)
        out_tuple = _graphnet_imap( in_tuple )
        results = out_tuple[3]
        coefs = results[0][499]
        pl.imsave('imtest.png',coefs.reshape((60,60),order='F'))
        #1/0

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
        outfile[r[0]][str(r[1])]['params'] = np.array(r[1])
        outfile[r[0]][str(r[1])]['l1vec'] = np.array(r[2])
        outfile[r[0]][str(r[1])]['coefficients'] = np.array(r[3][0])
        outfile[r[0]][str(r[1])]['residuals'] = np.array(r[3][1])
        print "Mean and median residuals:", np.mean(r[3][1]), np.median(r[3][1])
    # TODO: save parameter grid to hdf5 file
    outfile.close()
    
    print "\n\n Congratulations - nothing exploded!"

def validate_all(h5file):
    # get validation data
    Data = np.load("Data.npz")
    X = Data['X'][1000:2000,:]
    Y = Data['Y'][1000:2000]
    y = Y.copy()
    y[Y>0] = 1.0
    y[Y<=0] = -1.0
    y.shape = (1000,1)
    
    # # big grid
    # l1_len = 100
    # l2vec = [0.0, 1.0, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]
    # l3vec = [0.0, 1.0, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]
    # deltavec = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2, 10, 100]
    # svmdeltavec = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2, 10, 100]

    # test grid
    l2vec = [0.0, 1e6]
    l3vec = [0.0, 1e6]
    deltavec = [0.0, 0.01, 0.04, 1]
    svmdeltavec = [0.0, 0.4, 1]
    
    rate_arr_dims = (l1_len, len(l2vec), len(l3vec), len(deltavec), len(svmdeltavec))

    fits = h5py.File(h5file)
    for model in fits.keys():
        current_model = fits[model]
        rate_arr = np.zeros(rate_arr_dims)
        for params in current_model.keys():
            print model, params
            current_fit = current_model[params]
            coefs = np.array(current_fit['coefficients']).T
            params = np.array(current_fit['params'])
            if model == 'HuberSVMGraphNet':
                coefs = coefs[1::,:]
            preds = np.dot(X,coefs)
            preds[preds>0] = 1.0
            preds[preds<=0] = -1.0
            errs = y-preds != 0.0
            err_rates = np.sum(errs,axis=0)/1000.0
            #1/0
            fits[model+'/'+str(tuple(params.tolist()))+'/'+'err_rates'] = err_rates
            print '\t---> Best rate:', 1.-np.min(err_rates)
            rate_arr[:,np.where(params[0]==l2vec)[0][0],np.where(params[1]==l3vec)[0][0],np.where(params[2]==deltavec)[0][0],np.where(params[3]==svmdeltavec)[0][0]] = err_rates
        fits[model+'/'+'err_rate_array'] = rate_arr
    fits.close()
    print "Error rates on validation data have been added to file."

#------------------------------------------------------------------------------------------------------#
# Wrapper for running GraphNet problems using multiprocessing

def _graphnet_imap( in_tuple ):
    """
    Run a graphnet model for a particular tuple (X,Y,G,(l2,l3,delta,svmdelta),num_l1,lam_max)
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

def test_graphnet(X,Y,G=None,l1=500.,l2=-999.0,l3=-999.0,delta=-999.0,svmdelta=-999.0,initial=None,adaptive=False,svm=False,scipy_compare=True,tol=1e-5):
    tic = time.clock()
    # Cases set based on parameters and robust/adaptive/svm flags
    if l2 != -999.0 or l3 != -999.0 or delta != -999.0 or svmdelta != -999.0:
        if l3 != -999.0 or delta != -999.0 or svmdelta != -999.0:
            if G is None:
                nx = 60
                ny = 60
                A, Afull = construct_adjacency_list(nx,ny,1,return_full=True)
#                A, Afull = gen_adj(X.shape[1])
            else:
                A = G.copy()
            if delta != -999.0:
                if svmdelta != -999.0:                    
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
    coefficients, residuals = l.fit(tol=tol, initial=initial)
    print "\t---> Fitting GraphNet problem with coordinate descent took:", time.clock()-tic, "seconds."

    if adaptive:
        tic = time.clock()
        l1weights = 1./beta
        l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs = initial)
        l.problem.assign_penalty(l1=l1,l2=l2,l3=l3,delta=delta,l1weights=l1weights,newl1=l1)
        adaptive_coefficients, adaptive_residuals = l.fit(tol=tol, initial=initial) 
        print "\t---> Fitting Adaptive GraphNet problem with coordinate descent took:", time.clock()-tic, "seconds."
    
    # if compare to scipy flag is set,
    # compare the above result with the same problem 
    # solved using a built in scipy solver (fmin_powell).
    if scipy_compare:
        print "\t---> Fitting with scipy for comparison..."
        tic = time.clock()
        l1 = l1[-1] # choose only last l1 value
        beta = coefficients[-1] # coordinate-wise coefficients
        if l2 != -999.0 or l3 != -999.0 or delta != -999.0 or svmdelta != -999.0:
            if l3 != -999.0 or delta != -999.0 or svmdelta != -999.0:
                if delta != -999.0:
                    if adaptive: 
                        if svmdelta != -999.0:
                            # HuberSVM Graphnet
                            Xp2 = np.hstack([np.ones(X.shape[0])[:,np.newaxis],X])
                            def f(beta):
                                ind = range(1,len(beta))
                                return huber_svm_error(beta,Y,Xp2,delta).sum()  + np.fabs(beta[ind]).sum()*l1 + l2 * np.linalg.norm(beta[ind])**2/2 + l3 * np.dot(beta[ind], np.dot(Afull, beta[ind]))/2
                        else:
                            # Robust Adaptive Graphnet
                            def f(beta):
                                return huber(Y - np.dot(X, beta),delta).sum()/2  + l1*np.dot(np.fabs(beta),l1weights) + l2*np.linalg.norm(beta)**2/2 + l3*np.dot(beta, np.dot(Afull, beta))/2
                    else:
                        # Robust Graphnet
                        def f(beta):
                            try:
                                return huber(Y - np.dot(X, beta.T),delta).sum()/2  + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta.T))/2
                            except:
                                return huber(Y - np.dot(X, beta),delta).sum()/2  + np.fabs(beta).sum()*l1 + l2 * np.linalg.norm(beta)**2/2 + l3 * np.dot(beta, np.dot(Afull, beta).T)/2
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

    return (coefficients, residuals), problemkey

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

def huber(r,delta):
    r = np.fabs(r)
    t = np.greater(r, delta)
    return (1-t)*r**2 + t*(2*delta*r - delta**2)        

def huber_svm(r,delta):
    t1 = np.greater(r, delta)
    t2 = np.greater(r,0)
    return t1*(r - delta/2) + (1-t1)*t2*(r**2/(2*delta))

def huber_svm_error(beta,Y,Xp2,delta):
    r = 1-Y*np.dot(Xp2,beta)
    return huber(r,delta)

#-------------------------------------------------------------------------------------------------------------------#
# plotting functions

def plot_coefficient_images(h5file, output_dir, data_file='Data.npz', x=None, y=None,problemtype="RobustGraphNet"):
    """
    Iterate through hdf5 file of fits, plotting the coefficients as images and slices of images.
    """
    # get ground truth
    Data = np.load(data_file)
    true_im = Data['sig_im']

    # get fit results
    f = h5py.File(h5file,'r')
    results = f[problemtype]

    # make appropriate directories for saving images
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for k in results.keys():
        local_dir = output_dir + k
        if not os.path.isdir(local_dir):
            os.makedirs(local_dir)
            os.makedirs(local_dir + "/slice_plots/")
        # get coefficients and l1 values
        solution = results[k+'/coefficients'].value
        l1_path= results[k+'/l1vec'].value
        if x is None and y is None:
            x = np.sqrt(solution.shape[1])
            y = x # image is square
        # make plots
        for i in xrange(solution.shape[0]):
            im = solution[i,:].reshape((x,y),order='F')
            pl.imsave(local_dir + "/l1=" + str(l1_path[i]) + ".png", im)
            print "\t---> Saved coefficient image", i
            plot_image_slice(im, true_im, x_slice=45, out_path=local_dir+"/slice_plots/l1="+str(l1_path[i])+".png")
            print "\t---> Saved coefficient image slice", i

def plot_image_slice(im, true_im, x_slice, out_path):
    im_slice = im[x_slice,:]
    true_im_slice = true_im[x_slice,:]
    pl.clf()
    pl.plot(im_slice)
    pl.plot(true_im_slice,'r--')
    pl.savefig(out_path)

#-------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    pass

#EOF
