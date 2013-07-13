
import numpy as np
import random
import itertools
import functools
from pprint import pprint
from process import Process




class CVObject(Process):
    
    
    def __init__(self, variable_dict=None, data_obj=None):
        super(CVObject, self).__init__(variable_dict=variable_dict)
        self.crossvalidator = Crossvalidation()
        if data_obj:
            self.data = data_obj
            self.subject_indices = getattr(self.data, 'subject_indices', None)
            self.indices_dict = self.subject_indices
        else:
            self.data = None
        self.crossvalidation_ready = False
    
    
    def set_folds(self, folds):
        self.crossvalidator.folds = folds
        self.folds = folds
        
        
    def replace_Y_vals(self, Y, original_val, new_val):
        replace = lambda val: new_val if (val==original_val) else val
        return np.array([replace(v) for v in Y])
        
        
    def replace_Y_negative_ones(self):
        self.Y = self.replace_Y_vals(self.Y, -1., 0.)
        
        
    def prepare_folds(self, indices_dict=None, folds=None, leave_mod_in=False):
        # indices dict must be a python dictionary with keys corresponding to
        # some kind of grouping (typically keys for subjects/brains).
        # the values for those keys in the dict are the indices of the X and Y
        # matrices in the data object "attached" to these subjects.
        # this allows for flexible and decently clear upsampling, downsampling,
        # and crossvalidation across folds of these "keys"
        
        # if no indices dict specified, try and get it from self.data's
        # subject_indices, assuming it has been made.
        if not indices_dict:
            if not self.data:
                print 'Unable to find indices_dict, quitting crossvalidation preparation'
                return False
            else:
                if not getattr(self.data, 'subject_indices', None):
                    print 'Unable to find indices_dict, quitting crossvalidation preparation'
                    return False
                else:
                    indices_dict = self.data.subject_indices
        
        # set folds:
        if folds:
            self.set_folds(folds)
        else:
            self.folds = None
            
        # have the crossvalidator object make training and testing dicts:
        self.crossvalidator.create_crossvalidation_folds(indices_dict=indices_dict,
                                                         folds=self.folds,
                                                         leave_mod_in=leave_mod_in)
            
        # reassign variables of CVObject from the crossvalidator:
        self.folds = self.crossvalidator.folds
        self.train_dict = self.crossvalidator.train_dict
        self.test_dict = self.crossvalidator.test_dict
        
        #if (getattr(self, 'X', None) is not None) and (getattr(self, 'Y', None) is not None):
        print 'assigning to trainX, trainY, testX, testY...'
        #self.cv_group_XY(self.X, self.Y)
        self.cv_group_XY()
        self.crossvalidation_ready = True
        #else:
        #    print 'X and Y matrices, unset.. run cv_group_XY(X, Y) when ready to get cv groups'
        
        
        return True
            
            
    def subselect(self, data, indices):
        return np.array([data[i].tolist() for i in indices])
        
        
    def subselect_from_memmap(self, indices, X_memmap_path=None, X_memmap_shape=None,
                              verbose=True):
        
        if X_memmap_path:
            self.data.X_memmap_path = X_memmap_path
        if X_memmap_shape:
            self.data.X_memmap_shape = X_memmap_shape
            
        #bm = raw_input('Before memmap')
        X_memmap = np.memmap(self.data.X_memmap_path, dtype='float64', mode='r', shape=self.data.X_memmap_shape)
        
        if verbose:
            print X_memmap.shape, np.sum(X_memmap)
        
        #bf = raw_input('Before subset')
        subset =  np.array([X_memmap[i] for i in indices])
        
        if verbose:
            print subset.shape, np.sum(subset)
            #print 'original X:'
            #origX_subset = np.array([self.X[i].tolist() for i in indices])
            #print np.sum(origX_subset)
        
        #bd = raw_input('Before delete')
        del X_memmap
        #ad = raw_input('After Delete')
        
        return subset

    
    def cv_group_XY(self):
        if getattr(self, 'train_dict', None) and getattr(self, 'test_dict', None):
            fold_inds = self.train_dict.keys()
            assert fold_inds == self.test_dict.keys()
            
            #self.trainX = [self.subselect(X, self.train_dict[tg]) for tg in fold_inds]
            #self.trainY = [self.subselect(Y, self.train_dict[tg]) for tg in fold_inds]
            #self.testX = [self.subselect(X, self.test_dict[tg]) for tg in fold_inds]
            #self.testY = [self.subselect(Y, self.test_dict[tg]) for tg in fold_inds]
            
            self.trainX = [self.train_dict[tg] for tg in fold_inds]
            self.trainY = [self.train_dict[tg] for tg in fold_inds]
            self.testX = [self.test_dict[tg] for tg in fold_inds]
            self.testY = [self.test_dict[tg] for tg in fold_inds]
            
            self.subjects_in_folds = [self.crossvalidator.cv_sets[i] for i in fold_inds]
            
            self.crossvalidation_ready = True
            print 'completed groupings into trainX/Y, testX/Y'
        else:
            print 'Could not make train/test X Y matrices'
        
        
    
    def statsfunction_over_folds(self, statsfunction, Xgroups, Ygroups, **kwargs):
        # the statsfunction ported in must contain ONLY 2 NON-KEYWORD ARGUMENTS:
        # X data and Y data. The rest of the arguments MUST BE KEYWORDED.
        # you can pass the keyword arguments to this function that you would
        # have passed to the statsfunction originally. Note that the keyword
        # arguments (obviously) have to have the same name as they did in
        # statsfunction since they will be passed along to statsfunction soon
        # enough.
        results = []
        for X, Y in zip(Xgroups, Ygroups):
            statspartial = functools.partial(statsfunction, X, Y, **kwargs)
            results.append([statspartial()])
        return results
        
        
        
        
        
    def traintest_crossvalidator(self, trainfunction, testfunction, trainXgroups,
                                 trainYgroups, testXgroups, testYgroups, train_kwargs_dict={},
                                 test_kwargs_dict={}, X=None, Y=None, use_memmap=False, verbose=True):
        
        if not use_memmap:
            if X:
                fullX = X
            elif hasattr(self, 'X'):
                fullX = self.X
            else:
                print 'no X (either specified or in class)'
                return False
        
        if Y:
            fullY = Y
        elif hasattr(self, 'Y'):
            fullY = self.Y
        else:
            print 'no Y (either specified or in class)'
            return False
        
        #print fullY

        trainresults = []
        testresults = []
        
        
        for trainX, trainY, testX, testY in zip(trainXgroups, trainYgroups,
                                                testXgroups, testYgroups):
            if verbose:
                print 'Crossvalidating next group.'
                
            # assert independence of indices:
            for train_index in trainX:
                assert train_index not in testX
            for dependent_index in trainY:
                assert dependent_index not in testY
                
            if not use_memmap:
                subX = self.subselect(fullX, trainX)
            else:
                subX = self.subselect_from_memmap(trainX)
            #nothing = raw_input('subX loaded.')
            
            #print fullY, trainY
            subY = self.subselect(fullY, trainY)
            #print subY
            #nothing = raw_input('subY loaded.')
                
            trainpartial = functools.partial(trainfunction, subX, subY, **train_kwargs_dict)
            trainresult = trainpartial()
            
            if not use_memmap:
                subX = self.subselect(fullX, testX)
            else:
                subX = self.subselect_from_memmap(testX)
                
            subY = self.subselect(fullY, testY)
            
            testpartial = functools.partial(testfunction, subX, subY, trainresult, **test_kwargs_dict)
            testresult = testpartial()
            
            if verbose:
                print 'this groups\' test result:'
                pprint(testresult)
                
            trainresults.append(trainresult)
            testresults.append(testresult)
            
            #if verbose:
            #    print sum(testresults)/len(testresults)
            
        return trainresults, testresults
    



class Crossvalidation(object):
    
    def __init__(self, indices_dict=None, folds=None):
        super(Crossvalidation, self).__init__()
        self.indices_dict = indices_dict
        self.folds = folds
            
            
    def chunker(self, indices, chunksize):
        # chunker splits indices into equal sized groups, returns dict with IDs:
        groups = [indices[i:i+chunksize] for i in range(0, len(indices), chunksize)]
        
        # assert that each group is the same size:
        for g in groups:
            assert len(g) == chunksize
            
        # assert that indices are not repeated in other groups:
        # for each group:
        for i, g in enumerate(groups):
            # for each group not the same number as g:
            for j, o in enumerate(groups):
                if j != i:
                    # for each item in the other group:
                    for o_x in o:
                        # assert the item is not in the original group:
                        assert o_x not in g
        
        cv_sets = {}
        for i, group in enumerate(groups):
            cv_sets[i] = group
        return cv_sets
        
        
    def excise_remainder(self, indices, folds):
        modulus = len(indices) % folds
        random.shuffle(indices)
        return indices[modulus:], indices[0:modulus]
        
        
    def generate_sets(self, cv_sets, perms, mod_keys, include_mod):
        
        train_dict = {}
        test_dict = {}
        
        self.test_subject_byfold = []
        
        for p, groups in enumerate(perms):
            
            train_dict[p] = []
            test_dict[p] = []
            
            training_keys = []
            testing_keys = []
            
            for gkey in groups:
                training_keys.extend(cv_sets[gkey])
            
            for tr_key in training_keys:
                train_dict[p].extend(self.indices_dict[tr_key])
            
            for cv_key in cv_sets.keys():
                if cv_key not in groups:
                    testing_subjects = cv_sets[cv_key]
                    if include_mod:
                        testing_subjects.extend(mod_keys)
            
            for te_key in testing_subjects:
                test_dict[p].extend(self.indices_dict[te_key])
                
            # assert that there are no repeated indices in the training and
            # testing sets for this fold:
            for test_index in test_dict[p]:
                assert test_index not in train_dict[p]
                
            self.test_subject_byfold.append(testing_subjects)
                
        return train_dict, test_dict
        
        
        
    def create_crossvalidation_folds(self, indices_dict=None, folds=None, leave_mod_in=False,
                                     verbose=True):
        
        self.folds = folds or getattr(self,'folds',None)
        self.indices_dict = indices_dict or getattr(self,'indices_dict',None)
        
        self.train_dict = {}
        self.test_dict = {}
        
        if self.indices_dict is None:
            print 'No indices dictionary provided, exiting...'
            return False
        
        index_keys = self.indices_dict.keys()
        if verbose:
            print index_keys
            
        if len(index_keys) == 1:
            print 'Cannot do crossvalidation with just 1 subject!'
            return False
        
        if self.folds is None:
            print 'Folds unset, defaulting to leave one out crossvalidation...'
            self.folds = len(index_keys)
        
        divisible_keys, remainder_keys = self.excise_remainder(index_keys, self.folds)
        
        for rk in remainder_keys:
            assert rk not in divisible_keys
        
        # cv_sets is a dict with group IDs and indices:
        self.cv_sets = self.chunker(divisible_keys, len(divisible_keys)/self.folds)
        
        # ensure that the number of sets is equal to the number of folds:
        assert len(self.cv_sets) == self.folds
        
        # find the permutations of the group IDs, leaving one out:
        set_permutations = itertools.combinations(self.cv_sets.keys(), len(self.cv_sets.keys())-1)
        
        
        self.train_dict, self.test_dict = self.generate_sets(self.cv_sets, set_permutations,
                                                             remainder_keys, leave_mod_in)
        
        
        
        
        
        
        
        
        
        
        