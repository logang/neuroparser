
import os, sys, time
import numpy as np 
import scipy.optimize
from nose.tools import *
import h5py
#import matplotlib
import pylab as pl 
pl.ion()
from nipy.io.api import load_image
from nipy.core.api import Image


''' TODO: THESE IMPORTS NEED TO BE FIXED:

from optimization.cwpath import cwpath, strategy
from optimization.cwpath.cwpath import inner1d
from optimization.graphs.graph_laplacian import construct_adjacency_list

path_to_graphnetC_packages = os.path.abspath('./graphnetc/.')
sys.path.append(path_to_graphnetC_packages)
import graphnet
'''

from datamanager import NiftiTools, simple_normalize
from crossvalidation import CVObject





def adj_from_nii(maskfile,num_time_points,numt=0,numx=1,numy=1,numz=1,regions=None):
    """
    Construct adjacency array from .nii mask file

    INPUT:

    maskfile: Path to mask file (.nii)

    Other parameters are passed directly to prepare_adj (see that function for docs)

    OUTPUT:

    adj: An array containing adjacency information
    """
    mask = load_image(maskfile)._data
    newmask = np.zeros(np.append(num_time_points,mask.shape))
    for i in range(num_time_points):
        newmask[i] = mask
    adj = prepare_adj(newmask,numt,numx,numy,numz,regions)
    adj = convert_to_array(adj)
    return adj




def prepare_adj(mask,numt=0,numx=1,numy=1,numz=1,regions=None, gm_mask=None, verbose=True):
    """
    Return adjacency list, where the voxels are considered
    neighbors if they fall in a ball of radius numt, numx, numy, and numz
    for time, x position, y position, and z position respectively.

    INPUT:

    X: a 5-dimensional ndarray. The first index is trial, the second index is time,
    the third index is x position, the fourth index is y position and the fifth
    position is z position.

    mask: a binary 4-dimensional ndarray, the same size as X[0,:,:,:,:] where
    1 indicates that the voxel-timepoint is included and 0 indicates that it is
    excluded. NOTE: Usually the mask is thought of as a 3-dimensional ndarray, since
    it is uniform across time. 

    regions: a multivalued array the same size as the mask that indicates different
    regions in the spatial structure. No adjacency edges will be made across region
    boundaries.

    numt: an integer, the radius of the "neighborhood ball" in the t direction
    numx: an integer, the radius of the "neighborhood ball" in the x direction                                                                
    numy: an integer, the radius of the "neighborhood ball" in the y direction                                                                
    numz: an integer, the radius of the "neighborhood ball" in the z direction                                                                
                                                                    
    OUTPUT:

    newX: The matrix X reshaped as a 2-dimensional array for analysis
    adj: The adjacency list associated with newX

    """
    
    #Create map going from X to predictor vector indices. The entries of
    # this array are -1 if the voxel is not included in the mask, and the 
    # index in the new predictor corresponding to the voxel if the voxel
    # is included in the mask.

    if regions == None:
        regions = np.zeros(mask.shape)
    regions.shape = mask.shape
    reg_values = np.unique(regions)
    
    vmap = np.cumsum(mask).reshape(mask.shape)
    mask = np.bool_(mask.copy())
    vmap[~mask] = -1
    vmap -= 1 # now vmap's values run from 0 to mask.sum()-1
    
    
    if gm_mask is not None:
        gm = True
    else:
        gm = False

    if verbose:
        if gm:
            print 'Constructing adjacency matrix with greymatter mask'

    # Create adjacency list
    
    adj = []
    #gm_adj = []

    nt,nx,ny,nz = mask.shape

    for t in range(nt):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if mask[t,i,j,k]:
                        local_map = vmap[max((t-numt),0):(t+numt+1),
                                         max((i-numx),0):(i+numx+1),
                                         max((j-numy),0):(j+numy+1),
                                         max((k-numz),0):(k+numz+1)]
                        
                        if gm:
                            local_gm = gm_mask[max((t-numt),0):(t+numt+1),
                                               max((i-numx),0):(i+numx+1),
                                               max((j-numy),0):(j+numy+1),
                                               max((k-numz),0):(k+numz+1)]
                        
                        local_reg = regions[max((t-numt),0):(t+numt+1),
                                            max((i-numx),0):(i+numx+1),
                                            max((j-numy),0):(j+numy+1),
                                            max((k-numz),0):(k+numz+1)]
                        
                        region = regions[t,i,j,k]
                        ind = (local_map>-1)*(local_reg == region)
                        ind = np.bool_(ind)
                        adjrow = np.array(local_map[ind], dtype=int)
                        
                        if gm:
                            gmrow = np.array(local_gm[ind], dtype=float)
                            adj.append([[adjr, gmr] for adjr, gmr in zip(adjrow, gmrow)])

                        else:
                            adj.append([[adjr, 1.] for adjr in adjrow])
                        #else:
                            #gmrow = np.ones(len(adjrow), dtype=float)
                            
                        

                        #adj.append(np.array(local_map[ind],dtype=int))
                        #if gm:
                        #    gm_adj.append(np.array(vgm_mask[ind], dtype=float))
                        #adj.append(local_map[ind])
                        
    
    #accum = []
    #for i, a in enumerate(adj):
    #    for [ax, g] in a:
    #        accum.append(g)
            
    #print np.unique(g), np.unique(vgm_mask)
    #print np.sum(g), np.sum(vgm_mask)
    #stop
                        
    for i, a in enumerate(adj):
        for j, [ax, g] in enumerate(a):
            if ax == i:
                a[j] = [-1, g]
        adj[i] = a
        
        #a[np.equal(a,i)] = -1
        #adj[i] = a.tolist()
        
    #if gm:
    #    for i, g in enumerate(gm_adj):
    #        gm_adj[i] = g.tolist()
    #return convert_to_array(adj)
    
    #if gm:
    #    return adj, gm_adj
    #else:
    #    return adj


    return adj




def convert_to_array(adj):
    num_ind = np.max([len(a) for a in adj])
    adjarray = -np.ones((len(adj),num_ind))
    for i in range(len(adj)):
        for j in range(len(adj[i])):
            adjarray[i,j] = adj[i][j]
    return adjarray

   
def test_prep(nt=0,nx=1,ny=1,nz=1):
    """
    Let's make this into a proper test...... what should newa, adj be in this case?
    """
    a = np.array(range(1,1+2*3*4*4*4)).reshape((2,3,4,4,4))
    mask = a[0]*0
    mask[:,0,0,0] = 1
    mask[:,1,1,:] = 1
#    print mask[0]
#    print a[0,0]
    adj = prepare_adj(mask,nt,nx,ny,nz)
#    print newa[0,0], adj[0], newa[0,adj[0]]


class GraphnetInterface(CVObject):


    def __init__(self, data_obj=None, variable_dict=None, folds=None):
        super(GraphnetInterface, self).__init__(variable_dict=variable_dict, data_obj=data_obj)
        self.set_folds(folds)
        self.niftitools = NiftiTools()
        
        self.X = getattr(self.data, 'X', None)
        self.Y = getattr(self.data, 'Y', None)
        self.trial_mask = getattr(self.data, 'trial_mask', None)
        if not self.Y is None:
            self.replace_Y_negative_ones()
        self.indices_dict = getattr(self.data, 'subject_trial_indices', None)
        
        
    def huber(self, r, delta):
        r = np.fabs(r)
        t = np.greater(r, delta)
        return (1-t)*r**2 + t*(2*delta*r - delta**2)
        
        
    def huber_svm(self, r, delta):
        t1 = np.greater(r, delta)
        t2 = np.greater(r, 0)
        return t1*(r - delta/2) + (1-t1)*t2*(r**2/(2*delta))
        
        
    def huber_svm_error(self, beta, Y, Xp2, delta):
        r = 1-Y*np.dot*(Xp2, beta)
        return self.huber(r, delta)
        
        
    def get_lambda_max(self, X, y):
        """ 
        Find the value of lambda at which all coefficients are set to zero
        by finding the minimum value such that 0 is in the subdifferential
        and the coefficients are all zero.
        """
        subgrads = np.fabs(inner1d(X.T, y))
        return np.max(subgrads)
        
    '''
    def adj_array_as_list(self, adj):
        v = []
        for a in adj:
            v.append(a[np.greater(a, -1)])
        return v
    
    
    def gen_adj(self, p):
        print 'generating adjacency matrix'
        Afull = np.zeros((p, p), dtype=int)
        A = -np.ones((p, p), dtype=int)
        counts = np.zeros(p)
        for i in range(p):
            for j in range(p):
                if np.random.uniform(0, 1) < 0.3:
                    if i != j:
                        if Afull[i,j] == 0:
                            Afull[i,j] = -1
                            Afull[j,i] = -1
                            Afull[i,i] += 1
                            Afull[j,j] += 1
                            A[i, counts[i]] = j
                            A[j, counts[j]] = i
                            counts[i] += 1
                            counts[j] += 1
        return self.adj_array_as_list(A), Afull
    '''
    
    def regression_type_selector(self, l1, l2, l3, delta, svmdelta):
        print l1, l2, l3, delta, svmdelta
        l1b = all(l1)
        if (l1b != False) and (l2 != None) and (l3 != None) and (delta != None) and (svmdelta != None):
            return 'HuberSVMGraphNet'
        elif (l1b != False) and (l2 != None) and (l3 != None) and (delta != None):
            return 'RobustGraphNet'
        elif (l1b != False) and (l2 != None) and (l3 != None):
            return 'NaiveGraphNet'
        elif (l1b != False) and (l2 != None):
            return 'NaiveENet'
        elif (l1b != False):
            return 'Lasso'
        else:
            return None
        
        
    def setup_crossvalidation(self, folds=None, subject_indices=None, leave_mod_in=False):
        if subject_indices:
            self.subject_indices = subject_indices
        if getattr(self, 'subject_indices', None):
            self.prepare_folds(folds=folds, indices_dict=self.subject_indices, leave_mod_in=leave_mod_in)
        else:
            print 'no subject indices set, cant setup cv folds'
                        
            
    def crossvalidate(self, train_kwargs_dict, use_memmap=False):
                
        trainresults, testresults = self.traintest_crossvalidator(self.train_graphnet,
                                                                  self.test_graphnet,
                                                                  self.trainX, self.trainY,
                                                                  self.testX, self.testY,
                                                                  train_kwargs_dict=train_kwargs_dict,
                                                                  use_memmap=use_memmap)
        
        self.accuracies = testresults
        self.average_accuracies = []
        for i in range(len(self.accuracies[0])):
            accs = []
            for j in range(len(self.accuracies)):
                accs.append(self.accuracies[j][i])
            self.average_accuracies.append(sum(accs)/len(accs))
                
        #self.average_accuracies = [sum(x)/len(x) for x in self.accuracies]
        print 'Average accuracies: ', self.average_accuracies
        
        # trainresults: list of coefficients for each l1 by fold
        # AS OF NOW JUST TAKING THE COEFS FOR ONE OF THE FOLDS:
        sub_tresults = trainresults[0]
        self.non_zero_coefs = [len([x for x in tr if x != 0.]) for tr in sub_tresults]
        
        return self.accuracies, self.average_accuracies, self.non_zero_coefs
                
        
        
    def test_graphnet(self, X, Y, coefs):
        
        X = simple_normalize(X)
        accuracies = []
        
        for i, coefset in enumerate(coefs):
            
            correct = []
            print 'Checking accuracy for test group'
            
            if self.problemkey == 'RobustGraphNet':
                coefset = coefset[:-self.trainX_shape[0]]
            
            for trial, outcome in zip(X, Y):
                predict = trial*coefset
                #print np.sum(predict)
                Ypredsign = np.sign(np.sum(predict))
                if Ypredsign < 0.:
                    Ypredsign = 0.
                else:
                    Ypredsign = 1.
                #print Ypredsign, outcome, (Ypredsign == outcome)
                correct.append(Ypredsign == outcome)
                
            fold_accuracy = np.sum(correct) * 1. / len(correct)
            
            print 'coef number:', i
            print 'fold accuracy: ', fold_accuracy
            accuracies.append(fold_accuracy)
            
            
        return accuracies
    
    
    def train_graphnet(self, X, Y, trial_mask=None, G=None, l1=None, l2=None, l3=None, delta=None,
                      svmdelta=None, initial=None, adaptive=False, svm=False,
                      scipy_compare=False, tol=1e-5, greymatter_mask=None, initial_l1weights=None,
                      use_adj_time=True):
                
        if not type(l1) in [list, tuple]:
            l1 = [l1]
                
        X = simple_normalize(X)
        
        tic = time.clock()
        
        #problemkey = self.regression_type_selector(*[bool(x) for x in [l1, l2, l3, delta, svmdelta]])
        
        problemkey = self.regression_type_selector(l1, l2, l3, delta, svmdelta)
        
        self.problemkey = problemkey
        self.trainX_shape = X.shape
        
        if problemkey in ('HuberSVMGraphNet', 'RobustGraphNet', 'NaiveGraphNet'):
            if G is None:
                #nx = 60
                #ny = 60
                #A, Afull = construct_adjacency_list(nx, ny, 1, return_full=True)
                #A, Afull = self.gen_adj(X.shape[1])
                #if greymatter_mask is not None:
                #    A, GMA = prepare_adj(trial_mask, numt=1, gm_mask=greymatter_mask)
                #else:
                #    A = prepare_adj(trial_mask, numt=1)
                #    GMA = None
                if use_adj_time:
                    A = prepare_adj(trial_mask, numt=1, gm_mask=greymatter_mask)
                else:
                    A = prepare_adj(trial_mask, numt=0, gm_mask=greymatter_mask)
                
            else:
                A = G.copy()
                
        if initial_l1weights is not None:
            newl1 = l1
        else:
            newl1 = None
        
        if problemkey is 'RobustGraphNet':
            problemtype = graphnet.RobustGraphNet
            print 'Robust GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)#, gma=GMA)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta, l1weights=initial_l1weights,
                                     newl1=newl1)
        
        elif problemkey is 'HuberSVMGraphNet':
            problemtype = graphnet.GraphSVM
            print 'HuberSVM GraphNet with penalties (l1, l2, l3, delta): ', l1, l2, l3, delta
            Y = 2*np.round(np.random.uniform(0, 1, len(Y)))-1
            l = cwpath.CoordWise((X, Y, A), problemtype)#, gma=GMA)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta, l1weights=initial_l1weights,
                                     newl1=newl1)
            
        elif problemkey is 'NaiveGraphNet':
            problemtype = graphnet.NaiveGraphNet
            print 'Testing GraphNet with penalties (l1, l2, l3): ', l1, l2, l3
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)#, gma=GMA)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, l1weights=initial_l1weights,
                                     newl1=newl1)
            
        elif problemkey is 'NaiveENet':
            problemtype = graphnet.NaiveENet
            print 'Testing ENET with penalties (l1, l2): ', l1, l2
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l1weights=initial_l1weights,
                                     newl1=newl1)
            
        elif problemkey is 'Lasso':
            problemtype = graphnet.Lasso
            print 'Testing LASSO with penalty (l1): ', l1
            l = cwpath.CoordWise((X, Y), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l1weights=initial_l1weights, newl1=newl1)
            
        else:
            print 'Incorrect parameters set (no problem key).'
            return False
        
        # Solve the problem:
        print 'Solving the problem...'
        
        coefficients, residuals = l.fit(tol=tol, initial=initial)
        
        self.coefficients = coefficients
        self.residuals = residuals
        
        print '\t---> Fitting problem with coordinate decesnt took: ', time.clock()-tic, 'seconds.'
        
        if adaptive:
            tic = time.clock()
            safety = 1e-5
            l1weights = 1./(self.coefficients[-1]+safety)
            l = cwpath.CoordWise((X, Y, A), problemtype, initial_coefs=initial)
            l.problem.assign_penalty(path_key='l1', l1=l1, l2=l2, l3=l3, delta=delta, l1weights=l1weights, newl1=l1)
            adaptive_coefficients, adaptive_residuals = l.fit(tol=tol, initial=initial)
            print '\t---> Fitting Adaptive GraphNet problem with coordinate descent took: ', time.clock()-tic, 'seconds.'
            
            self.firstpass_coefficients = self.coefficients
            self.firstpass_residuals = self.residuals
            self.coefficients = adaptive_coefficients
            self.residuals = adaptive_residuals
        
        '''
        if scipy_compare:
            
            l1 = l1[-1]
            beta = self.coefficients[-1]
        
            print '\t---> Fitting with scipy for comparison...'
            
            tic = time.clock()
            
            if problemkey is 'RobustGraphNet':
                def f(beta):
                    huber_sum = self.huber(Y - np.dot(X, beta), delta).sum()/2
                    beta_l1 = l1*np.dot(np.fabs(beta), l1weights)
                    beta_l2 = l2*np.linalg.norm(beta)**2/2
                    beta_l3 = l3*np.dot(beta, np.dot(Afull, beta))/2
                    return huber_sum + beta_l1 + beta_l2 + beta_l3
                
            elif problemkey is 'HuberSVMGraphNet':
                Xp2 = np.hstack([np.ones(X.shape[0])[:,np.newaxis], X])
                def f(beta):
                    ind = range(1, len(beta))
                    huber_err_sum = self.huber_svm_error(beta, Y, Xp2, delta).sum()
                    beta_l1 = np.fabs(beta[ind]).sum()*l1
                    beta_l2 = l2*(np.linalg.norm(beta[ind])**2/2)
                    beta_l3 = l3*(np.dot(beta[ind], np.dot(Afull, beta[ind])))/2
                    return huber_error_sum + beta_l1 + beta_l2 + beta_l3
                
            elif problemkey is 'NaiveGraphNet':
                def f(beta):
                    beta_XY = np.linalg.norm(Y - np.dot(X, beta))**2/2
                    beta_l1 = l1*np.fabs(beta).sum()
                    beta_l2 = l2*np.linalg.norm(beta)**2/2
                    beta_l3 = l3*np.dot(beta, np.dot(Afull, beta))/2
                    return beta_XY + beta_l1 + beta_l2 + beta_l3
                
            elif problemkey is 'NaiveENet':
                def f(beta):
                    beta_XY = np.linalg.norm(Y - np.dot(X, beta))**2/2
                    beta_l1 = l1*np.fabs(beta).sum()
                    beta_l2 = np.linalg.norm(beta)**2/2
                    
            elif problemkey is 'Lasso':
                def f(beta):
                    beta_XY = np.linalg.norm(Y - np.dot(X, beta))**2/2
                    beta_l1 = l1*np.fabs(beta).sum()
                    
            if problemkey is 'HuberSVMGraphNet':
                v = scipy.optimize.fmin_powell(f, np.zeros(Xp2.shape[1]), ftol=1.0e-14, xtol=1.0e-14, maxfun=100000)
            else:
                v = scipy.optimize.fmin_powell(f, np.zeros(X.shape[1]), ftol=1.0e-10, xtol=1.0e-10, maxfun=100000)
                
            v = np.asarray(v)
            
            print '\t---> Fitting GraphNet with scipy took: ', time.clock()-tic, 'seconds.'
            
            assert_true(np.fabs(f(v) - f(beta)) / np.fabs(f(v) + f(beta)) < tol)
            if np.linalg.norm(v) > 1e-8:
                assert_true(np.linalg.norm(v - beta) / np.linalg.norm(v) < tol)
            else:
                assert_true(np.linalg.norm(beta) < 1e-8)
                
            print '\t---> Coordinate-wise and Scipy optimization agree.'
            '''


        return self.coefficients
                
        
        
        
class Gridsearch(object):
    
    def __init__(self, savedir=os.getcwd()):
        super(Gridsearch, self).__init__()
        self.verbose = True
        self.savedir = savedir
        
        #self.l1_range = []                    
        #self.l2_range = []
        #self.l3_range = []
        
        self.folds = 5
        
        self.searches = []
        
        st = time.localtime()
        timestr = str(st.tm_mon)+'_'+str(st.tm_mday)+'_'+str(st.tm_hour)+'_'+str(st.tm_min)
        
        self.logfile_name = 'fgrid_'+timestr+'.json'
        
        self.records = {}
        
        
    def generate_l1_values(self, l1_lower, l1_upper, granularity, round_to_int=True,
                           inclusive_max=True):
        
        distance = float(l1_upper)-float(l1_lower)
        
        step = distance*granularity
        if round_to_int:
            step = round(step)
        
        if inclusive_max:
            l1_values = [l1_lower+(x*step) for x in range(int(round(1.*granularity))+1)]
        else:
            l1_values = [l1_lower+(x*step) for x in range(int(round(1.*granularity)))]
        
        if self.verbose:
            print 'l1_range:', l1_lower, l1_upper
            print 'distance:', distance
            print 'granularity:', granularity
            print 'step size:', step
            print 'l1 values:', l1_values
            
        return l1_values, step
    
    
    def simple_generate_l1_range(self, l1min, l1max, stepsize, no_negative=True):
        
        l1min, l1max, stepsize = float(l1min), float(l1max), float(stepsize)
        
        l1_range = [l1min]
        while l1_range[-1]+stepsize < l1max:
            l1_range.append(l1_range[-1]+stepsize)
        l1_range.append(l1max)
        
        if no_negative:
            l1_range = [x for x in l1_range if x > 0.]
        
        return l1_range
        
    
    
    def log_progress(self):
        
        jsonpath = os.path.join(self.savedir, self.logfile_name)
        jfid = open(jsonpath,'w')
        simplejson.dump(self.records, jfid)
        jfid.close()
        
        
    def run_naive_gnet(self, csearch, l1_list=None, use_memmap=False, greymatter_mask=None,
                       adaptive=False, test_run=False):
        
        cparams = csearch['parameters']
        
        print cparams
        
        if l1_list:
            print 'l1s:',l1_list
            train_kwargs = {'trial_mask':self.gnet.trial_mask, 'l1':l1_list,
                            'l2':cparams['l2'], 'l3':cparams['l3'], 'greymatter_mask':greymatter_mask,
                            'adaptive':adaptive, 'delta':cparams['delta']}
        else:
            train_kwargs = {'trial_mask':self.gnet.trial_mask, 'l1':cparams['l1'],
                            'l2':cparams['l2'], 'l3':cparams['l3'],'greymatter_mask':greymatter_mask,
                            'adaptive':adaptive, 'delta':cparams['delta']}
            
        
        self.gnet.setup_crossvalidation(subject_indices=self.gnet.subject_indices, folds=self.folds)
        
        if not test_run:
            accuracies, average_accuracies, nz_coefs = self.gnet.crossvalidate(train_kwargs, use_memmap=use_memmap)
        
        else:
            accuracies = [[random.random() for x in range(len(l1_list))] for x in range(5)]
            average_accuracies = []
            for i in range(len(accuracies[0])):
                accs = []
                for j in range(len(accuracies)):
                    accs.append(accuracies[j][i])
                average_accuracies.append(sum(accs)/len(accs))
            nz_coefs = [random.randint(0,1000) for x in range(len(l1_list))]


        
        self.accuracies = accuracies
        self.average_accuracies = average_accuracies
        self.non_zero_coefs = nz_coefs
        
        if l1_list:
            self.csearches = []
            for ind, l1 in enumerate(l1_list):
                nsearch = {}
                nsearch['parameters'] = {'l1':l1, 'l2':cparams['l2'], 'l3':cparams['l3'], 'delta':cparams['delta']}
                nsearch['parameters']['l1'] = l1
                group_accuracies = []
                for i in range(len(self.accuracies)):
                    group_accuracies.append(self.accuracies[i][ind])
                nsearch['accuracies'] = group_accuracies
                nsearch['average_accuracy'] = average_accuracies[ind]
                nsearch['non_zero_coefs'] = nz_coefs[ind]
                nsearch['search_iter'] = csearch['search_iter'] + ind
                
                #pprint(nsearch)
                
                self.csearches.append(nsearch)
            return self.csearches
        else:  
            csearch['accuracies'] = accuracies[0]
            csearch['average_accuracy'] = average_accuracies[0]
            csearch['non_zero_coefs'] = nz_coefs[0]
            return csearch
        
        


    def _multi_l1_pass(self, l1_range, l2, l3, delta=None, reverse_range=True,
                       use_memmap=False, adaptive=False, greymatter_mask=None,
                       verbose=True, test_run=False):

        cur_l1_range = l1_range[:]
        if reverse_range:
            cur_l1_range.reverse()

        cur_params = {'l1':[], 'l2':l2, 'l3':l3, 'delta':delta}
        csearch = {}
        csearch['search_iter'] = self.search_count
        csearch['parameters'] = cur_params

        if self.verbose:
            print '\nPREFORMING NEXT MULTI-SEARCH GRAPHNET\n'
            print 'l1 range:', cur_l1_range
            print 'l2', l2
            print 'l3', l3
            print 'delta', delta

                    
        csearches = self.run_naive_gnet(csearch, l1_list=cur_l1_range,
                                        use_memmap=use_memmap, greymatter_mask=greymatter_mask,
                                        adaptive=adaptive, test_run=test_run)
                
        for cs in csearches:
            self.searches.append(cs)
            self.search_count += 1
        self.records['current_iter'] = self.search_count
        self.records['searches'] = self.searches
        
        for srec in self.searches:
            cacc = srec['average_accuracy']
            if cacc > self.best_acc:
                self.best_acc = cacc
                self.best_parameters = srec['parameters']
                print 'new best parameters:', self.best_parameters
        
        self.records['best_acc'] = self.best_acc
        self.records['best_parameters'] = self.best_parameters
        
        self.log_progress()


    def _zoom_determine_l1minmax(self):
        best_l1 = self.best_parameters['l1']
        half_dist = float(self.current_l1_distance)/2.
        temp_min = best_l1 - half_dist
        temp_max = best_l1 + half_dist

        if min(temp_min, self.l1_hard_min) == self.l1_hard_min:
            return self.l1_hard_min, self.l1_hard_min+self.current_l1_distance
        else:
            return temp_min, temp_max


    def _zoom_cut_priorl1s(self, l1_list, prior_parameters):

        for parameters in prior_parameters:
            prior_l1 = parameters[0]
            l1_list = [x for x in l1_list if x != prior_l1]

        return l1_list        


    def zoom_gridsearch(self, gnet, name='zoom_gsearch', adaptive=False, use_memmap=False, 
                         greymatter_mask=None, test_run=False, verbose=True):

        self.gnet = gnet
        self.records['title'] = name
        if name:
            st = time.localtime()
            timestr = str(st.tm_mon)+'_'+str(st.tm_mday)+'_'+str(st.tm_hour)+'_'+str(st.tm_min)
            self.logfile_name = name+'_'+timestr+'.json'

        defaults = {'initial_l1_min':5.,
                    'initial_l1_max':65.,
                    'l1_stepsizes':[6.,3.,1.],
                    'l1_hard_min':5.,
                    'l1_shrink_coef':.5,
                    'l2_range':[1.,10.,100.,1000.,10000.],
                    'l3_range':[1.,10.,100.,1000.,10000.],
                    'deltas':[]}


        for gs_var, var_val in defaults.items():
            if getattr(self, gs_var, None) in [None, [], {}, 0., 0, False]:
                setattr(self, gs_var, var_val)
            self.records[gs_var] = getattr(self, gs_var, None)

        if greymatter_mask is not None:
            self.greymatter_mask = greymatter_mask


        
        self.records['adaptive'] = adaptive
        self.records['use_memmap'] = use_memmap
        self.records['greymatter_mask'] = hasattr(self, 'greymatter_mask')

        self.records['folds'] = self.folds
        self.records['current_iter'] = 0
        self.records['searches'] = self.searches
        
        self.search_count = 0
        self.best_acc = 0.
        self.best_parameters = {}
        self.parameter_tracker = []
        self.initial_l1_distance = self.initial_l1_max - self.initial_l1_min
        self.records['l1_distances'] = []
        self.records['l1_ranges'] = []

        self.log_progress()

        
        # l1 step zooms:
        for zoom_n, l1_step in enumerate(self.l1_stepsizes):

            if zoom_n == 0:
                self.current_l1_distance = self.initial_l1_distance
                self.current_l1_min = self.initial_l1_min
                self.current_l1_max = self.initial_l1_max
            else:
                self.current_l1_distance = self.l1_shrink_coef*self.current_l1_distance
                self.current_l1_min, self.current_l1_max = self._zoom_determine_l1minmax()

            self.records['l1_distances'].append(self.current_l1_distance)


            if verbose:
                print 'Preforming zoom pass...'
                print 'zoom distance', self.current_l1_distance


            self.l1_range = self.simple_generate_l1_range(self.current_l1_min,
                                                          self.current_l1_max,
                                                          l1_step)

            sparse_l1_range = self._zoom_cut_priorl1s(self.l1_range, self.parameter_tracker)

            self.records['l1_ranges'].append(sparse_l1_range)

            for l3 in self.l3_range:
                for l2 in self.l2_range:
                    if not self.deltas:
                        self._multi_l1_pass(sparse_l1_range, l2, l3, test_run=test_run, delta=None, adaptive=adaptive,
                                            greymatter_mask=greymatter_mask)
                        for l1 in sparse_l1_range:
                            self.parameter_tracker.append([l1,l2,l3])
                    else:
                        for delta in self.deltas:
                            self._multi_l1_pass(sparse_l1_range, l2, l3, delta=delta, test_run=test_run, adaptive=adaptive,
                                                greymatter_mask=greymatter_mask)




    
        
    def standard_gridsearch(self, gnet, reverse_range=True, name='standard_gsearch', adaptive=False, 
                            use_memmap=False, greymatter_mask=None):
        
        self.gnet = gnet
        self.records['title'] = name

        if name:
            st = time.localtime()
            timestr = str(st.tm_mon)+'_'+str(st.tm_mday)+'_'+str(st.tm_hour)+'_'+str(st.tm_min)
            self.logfile_name = name+'_'+timestr+'.json'

        self.records['l1_range'] = self.l1_range
        self.records['l2_range'] = self.l2_range
        self.records['l3_range'] = self.l3_range

        self.records['folds'] = self.folds
        self.records['current_iter'] = 0
        self.records['searches'] = self.searches
        
        
        search_count = 0
        l1min = self.l1_range[0]
        l1max = self.l1_range[-1]
        best_acc = 0.
        best_l1 = -1
        best_l2 = -1
        best_l3 = -1
        cur_distance = l1max-l1min
        
        self.log_progress()
        
        for l3 in self.l3_range:
            for l2 in self.l2_range:
                cur_l1_range = self.l1_range[:]
                                                  
                if reverse_range:
                    cur_l1_range.reverse()
                                                    
                cur_params = {'l1':[], 'l2':l2, 'l3':l3}
                csearch = {}
                csearch['search_iter'] = search_count
                self.records['current_iter'] = search_count
                csearch['parameters'] = cur_params
                
                if self.verbose:
                    print '\nPREFORMING NEXT MULTI-SEARCH GRAPHNET\n'
                    print 'l1 range:', cur_l1_range
                    print 'l2', l2
                    print 'l3', l3
                    
                csearches = self.run_naive_gnet(csearch, l1_list=cur_l1_range,
                                                use_memmap=use_memmap, greymatter_mask=greymatter_mask,
                                                adaptive=adaptive)
                
                for cs in csearches:
                    self.searches.append(cs)
                    search_count += 1
                self.records['current_iter'] = search_count
                self.records['searches'] = self.searches
                
                for srec in self.searches:
                    cacc = srec['average_accuracy']
                    if cacc > best_acc:
                        best_acc = cacc
                        best_parameters = srec['parameters']
                
                self.records['best_acc'] = best_acc
                self.records['best_parameters'] = best_parameters
                
                self.log_progress()
                    
    


                
                
                
                
                
                
                
            
            
                
            
        
