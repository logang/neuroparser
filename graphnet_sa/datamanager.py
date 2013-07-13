
import functools
import os, glob, shutil
import subprocess
import sys, time
import numpy as numpy
import scipy as sp 
import nibabel as nib
from baseclasses import Process
from afnifunctions import AfniWrapper
from directorytools import glob_remove


def simple_normalize(X, axis=0):
    
    print 'normalizing X'
    print 'previous X sum', np.sum(X)
    
    std_devs = np.std(X, axis=axis)
    means = np.mean(X, axis=axis)
    
    Xnorm = np.zeros(X.shape)
    Xnorm = X-means
    Xnorm = Xnorm/std_devs

    # remove any nans or infinities:
    Xnorm = np.nan_to_num(Xnorm)
    
    print 'post-normalization X sum', np.sum(Xnorm)
    
    return Xnorm




class NiftiTools(object):
    
    
    def __init__(self):
        super(NiftiTools, self).__init__()
        self.afni = AfniWrapper()
        
        
    def adwarp_to_template_talairach(self, input_path, output_path_prefix, talairach_path, dxyz=1.):
        self.afni.adwarp(talairach_path, input_path, output_path_prefix, dxyz=dxyz)
                
        
    def convert_to_nifti(self, dataset_in, dataset_out):
        glob_remove(dataset_out)
        subprocess.call(['3dAFNItoNIFTI', '-prefix', dataset_out, dataset_in])
        
        
    def convert_to_afni(self, nifti_in, dataset_out):
        try:
            glob_remove(dataset_out+'+orig')
        except:
            pass
        try:
            glob_remove(dataset_out+'+tlrc')
        except:
            pass
        subprocess.call(['3dcopy', nifti_in, dataset_out])
        
        
    def refit(self, functional, anatomical):
        subprocess.call(['3drefit', '-apar', anatomical, functional])
        
        
    def adwarp_to_subject_talairach(self, dataset_in, dataset_out, anatomical, dxyz):
        glob_remove(dataset_out)
        subprocess.call(['adwarp', '-apar', anatomical, '-dpar', dataset_in,
                         '-dxyz', str(dxyz), '-prefix', dataset_out])
        
        
    def create_mask(self, mask_dset_path, output_prefix_path, clfrac=.3):
        self.afni.automask(mask_dset_path, output_prefix_path, clfrac=clfrac)
        
        
    def create_talairach_niftis(self, subject_dirs, functional_name, anatomical_name,
                                dxyz, talairach_path, output_name, within_subject_warp=True,
                                to_template_warp=True):
        
        if not output_name.endswith('.nii'):
            output_name = output_name+'.nii'
        
        # iterate subject directories:
        for s in subject_dirs:
            
            functional = os.path.join(s, functional_name)
            anatomical = os.path.join(s, anatomical_name)
            
            # refit the functional to the anatomical:
            self.afni.refit_apar(anatomical+'+tlrc', functional+'+orig')
            #self.refit(functional, anatomical)
                        
            # adwarp the functional to the anatomical
            if within_subject_warp:
                adwarp_temp = os.path.join(s, 'temp_adwarp_func')
                self.afni.adwarp(anatomical+'+tlrc', functional+'+orig', adwarp_temp, dxyz=dxyz)
            else:
                adwarp_temp = functional
            
            # warp the warped functional to the talairach template:
            if to_template_warp:
                template_temp = os.path.join(s, 'temp_template_func')
                self.afni.adwarp(talairach_path, adwarp_temp+'+tlrc', template_temp, dxyz=dxyz,
                                 force=True)
            else:
                template_temp = adwarp_temp
                
            # make the nifti files
            nifti_path = os.path.join(s, output_name)
            try:
                glob_remove(nifti_path)
            except:
                pass
            self.convert_to_nifti(template_temp+'+tlrc', nifti_path)
            
            if within_subject_warp:
                glob_remove(adwarp_temp)
            if to_template_warp:
                glob_remove(template_temp)
                
        return [os.path.join(s, output_name) for s in subject_dirs]
        
    
    def save_nifti(self, data, affine, filepath):
        if not filepath.endswith('.nii'):
            filepath = filepath+'.nii'
        glob_remove(filepath)
        nii = nib.Nifti1Image(data, affine)
        nii.to_filename(filepath)
    
        
    def output_nifti_thrumask(self, data, mask, mask_3dshape, trlen, affine, output_filepath):
        
        shaped_matrix = np.zeros((mask_3dshape[0], mask_3dshape[1], mask_3dshape[2],
                                  trlen))
        shaped_matrix[np.where(mask)] = data
        self.save_nifti(shaped_matrix, affine, output_filepath)
        
        
        
    def load_nifti(self, nifti_path):
        image = nib.load(nifti_path)
        shape = image.get_shape()
        idata = image.get_data()
        affine = image.get_affine()
        return idata, affine, shape



class DataManager(Process):
    ''' 
    DataManager is the basic superclass for both the CsvData and BrainData classes
    (currently). It contains functions that both of those classes can use, particularly
    construction of the X and Y matrices.
    
    '''
    
    def __init__(self, variable_dict=None):

        super(DataManager, self).__init__(variable_dict=variable_dict)
        self.verbose = True
        
            
            
    def recode_variable(self, variable_list=[], variable_dict={}, allow_unspecified=True,
                        as_string=False):
        '''
        recode variable takes a list of variables to change and a dictionary that
        has old values as keys and new values as values. it iterates 
        through the variable_list and makes a new list of recoded varaibles.
        A bit clunky, yes? needs to be redone or taken out perhaps.

        allow_unspecified : this flag if set to true (a good move) will just keep a
        variable the same if it doesn't find it in the dict. a True value here will stop
        the function entirely.

        as_string: True here makes the variables strings, otherwise they are returned as
        floats.
        '''
        
        recoded = []
        for var in variable_list:
            if var in variable_dict.keys():
                nval = variable_dict[var]
                if as_string:
                    recoded.append(str(nval))
                else:
                    recoded.append(float(nval))
            else:
                if allow_unspecified:
                    if as_string:
                        recoded.append(string(var))
                    else:
                        recoded.append(float(var))
                else:
                    print 'variable not found in replacement dict'
                    return False
                
        if as_string:
            return recoded
        else:
            return np.array(recoded)
        
        
                
                
    def _xy_matrix_tracker(self, Ybinary):
        '''
        A function for verbosity in the create_XY_matrices.
        '''

        print 'X (trials) length: ', len(self.X)
        print 'Y (responses) length: ', len(self.Y)
        print 'positive responses: ', self.Y.count(Ybinary[0])
        print 'negative responses: ', self.Y.count(Ybinary[1])
        

    def _xy_matrix_tracker_multiclass(self, classes):

        print 'X (trials) length: ', len(self.X)
        print 'Y (responses) length: ', len(self.Y)
        for ycls in classes:
            print 'class', ycls, 'responses:', self.Y.count(ycls)
        
        
    def create_XY_matrices_multiclass(self, subject_design=None, downsample_type=None, with_replacement=False,
                                      replacement_ceiling=None, random_seed=None, classes=[]):

        required_vars = {'subject_design':subject_design}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        
        self.random_seed = random_seed or getattr(self,'random_seed',None)
        
        
        if self.random_seed:
            print self.random_seed
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)

        self.X = []
        self.Y = []
        self.subject_indices = {}

        if not downsample_type:

            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []

                if not with_replacement:
                    for trial, response in zip(trials, responses):
                        
                        self.subject_indices[subject].append(len(self.X))
                        self.X.append(trial)
                        self.Y.append(float(response))

                elif with_replacement:

                    class_dict = {}
                    class_trials = {}
                    for c in classes:
                        class_dict[int(c)] = 0
                        class_trials[int(c)] = []
                    
                    for trial, response in zip(trials, responses):
                        class_dict[int(response)] += 1
                        class_trials[int(response)].append(trial)

                    delete_sub = False
                    for c, v in class_dict.items():
                        if v == 0:
                            delete_sub = True

                    if delete_sub:
                        del self.subject_indices[subject]
                        print 'deleting', subject, 'from indices... zero of one class type'
                    
                    else:
                        if not replacement_ceiling:
                            upper = max(class_dict.values())
                        else:
                            upper = replacement_ceiling

                        for c, cl_set in class_trials.items():
                            random.shuffle(cl_set)

                            for i, trial in enumerate(cl_set):
                                if i < upper:
                                    self.subject_indices[subject].append(len(self.X))
                                    self.X.append(trial)

                            for rep_trial in [random.sample(cl_set,1)[0] for i in range(upper-len(cl_set))]:
                                self.subject_indices[subject].append(len(self.X))
                                self.X.append(rep_trial)

                            self.Y.extend([c for x in range(upper)])

                self._xy_matrix_tracker_multiclass(classes)


        elif downsample_type == 'subject':
            
            for subject, [trials, responses] in self.subject_design.items():
                print subject, len(trials), len(responses)
                self.subject_indices[subject] = []
                
                subclass_dict = {}
                subclass_trials = {}
                for c in classes:
                    subclass_dict[int(c)] = 0
                    subclass_trials[int(c)] = []
                
                for trial, response in zip(trials, responses):
                    #print response, len(trial)
                    subclass_dict[int(response)] += 1
                    subclass_trials[int(response)].append(trial)
                        
                if min(subclass_dict.values()) == 0:
                    del self.subject_indices[subject]
                    print 'deleting', subject, 'from indices... zero of one class type'

                else:
                    for c, trials in subclass_trials.items():
                        random.shuffle(trials)

                    if not with_replacement:
                        for i in range(min(subclass_dict.values())):
                            #pprint(subclass_trials)
                            for c in subclass_trials.keys():
                                self.subject_indices[subject].append(len(self.X))
                                self.X.append(subclass_trials[c][i])
                                self.Y.append(c)

                    elif with_replacement:
                        if not replacement_ceiling:
                            upper = max(self.subclass_dict.values())
                        else:
                            upper = replacement_ceiling

                        for c, subcls_set in subclass_trials.items():

                            for i, trial in enumerate(subcls_set):
                                if i < upper:
                                    self.subject_indices[subject].append(len(self.X))
                                    self.X.append(trial)
                                    self.Y.append(c)

                            for trial in [random.sample(subcls_set, 1)[0] for i in range(upper-len(subcls_set))]:
                                self.subject_indices[subject].append(len(self.X))
                                self.X.append(trial)
                                self.Y.append(c)

                self._xy_matrix_tracker_multiclass(classes)
                    
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        #pprint(self.Y)


        
        
    def create_XY_matrices(self, subject_design=None, downsample_type=None, with_replacement=False,
                           replacement_ceiling=None, random_seed=None, Ybinary=[1.,-1.], Yreplace=[1.,-1],
                           verbose=True):
        '''
        create_XY_matrices: the X and Y matrix creating super-function.

        this function will convert the subject_design dictionary into an X and Y matrix. It requires that
        subject_design be formatted properly. keys in subject_design must be some indicator of the subject,
        values are tuples or lists containing the trials list at [0] and the responses list at [1].

        ex:
        subject_design['ab040313'] = [trials, responses]

        downsample_type     :   downsample type can either be set to None/False, 'subject', or 'trial'.
                                'subject' downsampling will equalize the number of yes and no responses
                                within each subject (key) in the subject design dictionary.
                                'trial' downsampling will equalize the number of yes and no responses over 
                                the cumulative trials of all the subjects in the subject design dict.

        '''
        
        required_vars = {'subject_design':subject_design}
        self._assign_variables(required_vars)
        if not self._check_variables(required_vars): return False
        
        
        self.random_seed = random_seed or getattr(self,'random_seed',None)
        
        
        if self.random_seed:
            print self.random_seed
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            
            
        self.X = []
        self.Y = []
        self.subject_indices = {}
        
        
        if not downsample_type:
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                if not with_replacement:
                    for trial, response in zip(trials, responses):
                        self.subject_indices[subject].append(len(self.X))
                        self.X.append(trial)
                        if float(response) == float(Ybinary[0]):
                            self.Y.append(Yreplace[0])
                        elif float(response) == float(Ybinary[1]):
                            self.Y.append(Yreplace[1])
                        else:
                            print 'y value unaccounted for, setting to 0...'
                            self.Y.append(0.)
                        #self.Y.append(response)
                        
                elif with_replacement:
                    
                    positive_trials = []
                    negative_trials = []
                    
                    for trial, response in zip(trials, responses):
                        if float(response) == float(Ybinary[0]):
                            positive_trials.append(trial)
                        elif float(response) == float(Ybinary[1]):
                            negative_trials.append(trial)
                            
                    if min(len(positive_trials), len(negative_trials)) == 0:
                        del self.subject_indices[subject]
                    
                    else:
                        if not replacement_ceiling:
                            upper_length = max(len(positive_trials), len(negative_trials))
                        else:
                            upper_length = replacement_ceiling
                        
                        for set in [positive_trials, negative_trials]:
                            random.shuffle(set)
                            
                            for i, trial in enumerate(set):
                                if i < upper_length:
                                    self.subject_indices[subject].append(len(self.X))
                                    self.X.append(trial)
                                
                            for rep_trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                                self.subject_indices[subject].append(len(self.X))
                                self.X.append(rep_trial)
                                
                        self.Y.extend([Yreplace[0] for x in range(upper_length)])
                        self.Y.extend([Yreplace[1] for x in range(upper_length)])
        
                if verbose:
                    self._xy_matrix_tracker(Yreplace)
                    
                    
                    
        elif downsample_type == 'group':
            
            positive_trials = []
            negative_trials = []
            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                for trial, response, in zip(trials, responses):
                    if float(response) == float(Ybinary[0]):
                        positive_trials.append([subject,trial])
                    elif float(response) == float(Ybinary[1]):
                        negative_trials.append([subject,trial])
                        
            random.shuffle(positive_trials)
            random.shuffle(negative_trials)
            
            if not with_replacement:
                for i in range(min(len(positive_trials), len(negative_trials))):
                    [psub, ptrial] = positive_trials[i]
                    [nsub, ntrial] = negative_trials[i]
                    self.subject_indices[psub].append(len(self.X))
                    self.X.append(ptrial)
                    self.subject_indices[nsub].append(len(self.X))
                    self.X.append(ntrial)
                    self.Y.extend([Yreplace[0],Yreplace[1]])
                    
                if verbose:
                    self._xy_matrix_tracker(Ybinary)
                    
            elif with_replacement:
                
                if not replacement_ceiling:
                    upper_length = max(len(positive_trials), len(negative_trials))
                else:
                    upper_length = replacement_ceiling
                
                for set in [positive_trials, negative_trials]:
                    random.shuffle(set)
                    
                    for i, (sub, trial) in enumerate(set):
                        if i < upper_length:
                            self.subject_indices[sub].append(len(self.X))
                            self.X.append(trial)
                                                
                    for sub, trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                        self.subject_indices[sub].append(len(self.X))
                        self.X.append(trial)
                        
                self.Y.extend([Yreplace[0] for x in range(upper_length)])
                self.Y.extend([Yreplace[1] for x in range(upper_length)])
                
                if verbose:
                    self._xy_matrix_tracker(Yreplace)
                    
                    
                    
        elif downsample_type == 'subject':

            
            for subject, [trials, responses] in self.subject_design.items():
                self.subject_indices[subject] = []
                
                subject_positives = []
                subject_negatives = []
                
                for trial, response in zip(trials, responses):
                    if float(response) == float(Ybinary[0]):
                        subject_positives.append(trial)
                    elif float(response) == float(Ybinary[1]):
                        subject_negatives.append(trial)
                        
                random.shuffle(subject_positives)
                random.shuffle(subject_negatives)
                
                if min(len(subject_positives), len(subject_negatives)) == 0:
                    del self.subject_indices[subject]
                    
                else:
                    if not with_replacement:
                        for i in range(min(len(subject_positives), len(subject_negatives))):
                            self.subject_indices[subject].append(len(self.X))
                            self.X.append(subject_positives[i])
                            self.subject_indices[subject].append(len(self.X))
                            self.X.append(subject_negatives[i])
                            self.Y.extend([Yreplace[0],Yreplace[1]])
                            
                    elif with_replacement:
                        if not replacement_ceiling:
                            upper_length = max(len(subject_positives), len(subject_negatives))
                        else:
                            upper_length = replacement_ceiling
                        
                        for set in [subject_positives, subject_negatives]:
                            random.shuffle(set)
                            
                            for i, trial in enumerate(set):
                                if i < upper_length:
                                    self.subject_indices[subject].append(len(self.X))
                                    self.X.append(trial)
                                
                            print upper_length
                            print len(set)
                                
                            for trial in [random.sample(set, 1)[0] for i in range(upper_length-len(set))]:
                                self.subject_indices[subject].append(len(self.X))
                                self.X.append(trial)
                                
                        self.Y.extend([Yreplace[0] for x in range(upper_length)])
                        self.Y.extend([Yreplace[1] for x in range(upper_length)])
                        
                if verbose:
                    self._xy_matrix_tracker(Yreplace)
                    
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        
        
    def merge_datamanagers(self, mergeable_dm):
        
        # this assumes that the mergeable dm has an X, Y, and subject_indices dict.
        # those are the only things that are assimmilated.
        
        oX = mergeable_dm.X
        oY = mergeable_dm.Y
        oSI = mergeable_dm.subject_indices
        
        merge_rows = len(oX)
        if merge_rows == len(oY):
            pass
        else:
            print 'merge X and merge Y have different lengths ?!'
            return False
        
        self.X.tolist()
        self.Y.tolist()
        
        self.X.extend(oX.tolist())
        self.Y.extend(oY.tolist())
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
                
        for subj, inds in oSI.items():
            ninds = [x+merge_rows for x in inds]
            if subj in self.subject_indices:
                self.subject_indices[subj].extend(ninds)
            else:
                print 'merged subject not in original subject_indices dict'
                self.subject_indices[subj] = ninds
        
        
        
    def delete_subject_design(self, checkpoint=False):
        if checkpoint:
            inp = raw_input('Press any key to delete subject design (preserves self.X and self.Y)')
        del self.subject_design
        
        
    def X_to_memmap(self, memmap_filepath, empty_X=True, verbose=True,
                    testmode_bypass_overwrite=False):
        
        if verbose: print 'Creating X memmap'
        self.X_memmap_path = memmap_filepath
        self.X_memmap_shape = self.X.shape
        
        if testmode_bypass_overwrite:
            print '!!! TEST MODE: BYPASSING MEMMAP OVERWRITE !!!'
        else:
            try:
                if verbose: print 'attempting to delete old memmap...'
                os.remove(self.X_memmap_path)
                if verbose: print 'old memmap deleted.'
            except:
                if verbose: print 'no memmap to delete'
                
            
            if verbose: print 'writing new memmap...'
            X_memmap = np.memmap(self.X_memmap_path, dtype='float64', mode='w+', shape=self.X_memmap_shape)
            X_memmap[:,:] = self.X[:,:]
            
            del X_memmap
        
        
    def empty_X(self):
        print 'emptying X'
        self.X = []
            
        
    def normalizeX(self):
        print 'normalizing X'
        print 'previous X sum', np.sum(self.X)
        #self.X = preprocessing.normalize(self.X, axis=1)
        self.X = simple_normalize(self.X, axis=0)
        print 'post-normalization X sum', np.sum(self.X)
        
        
    def scaleX(self):
        print 'scaling X'
        self.X = preprocessing.scale(self.X)
        
        
    def normalize_within_subject(self, normalizeY=False):
        '''
        Iterates through the subject trials and normalizes the X matrices
        individually for each subject. This should obviously be done prior
        to calling create_XY_matrices.
        
        Optionally normalizes the Y vector as well.
        '''
        
        for subject, [trials, resp_vec] in self.subject_design.items():
            print 'normalizing within subject: ', subject
            self.subject_design[subject][0] = preprocessing.normalize(trials)
            if normalizeY:
                print 'also normalizing Y...'
                self.subject_design[subject][1] = preprocessing.normalize(resp_vec)
                
                
                
    def scale_within_subject(self, scaleY=False):
        '''
        Iterates through subject X and Y matrices and scales the X matrix. Can
        also optionally scale the Y matrix. Should be done prior to calling
        create_XY_matrices, if done at all.
        '''
        
        for subject, [trials, resp_vec] in self.subject_design.items():
            print 'scaling within subject:', subject
            self.subject_design[subject][0] = preprocessing.scale(trials)
            if scaleY:
                print 'also scaling Y...'
                self.subject_design[subject][1] = preprocessing.scale(resp_vec)
        

    def recodeY(self, oldvalues, newvalues):
        '''
        Will iterate through oldvalues and newvalues (paired), replacing the old
        values with the new values.
        
        Keep in mind this replacement is iterative, so if you first replace 1 with
        0, then later replace 0 with -1, you will have -1s for 1s in the end.
        Thus the values that come last have precedence in replacement.
        '''
        
        if len(oldvalues) == len(newvalues):
            for ov, nv in zip(oldvalues, newvalues):
                self.Y = [nv if y == ov else y for y in self.Y]



   
    
class BrainData(DataManager):
    '''
    BrainData
    ------------
    Recoded BrainData class in style of Jonathan Taylor's masking scheme. It
    preforms faster than my old one so I decided to go with this. However,
    his originally forced a matrix transposition (reversal) which may not
    be ideal. This has the option of either using his matrix format or
    preserving (as best as possible) the dimensionality when loading niftis
    using nipy or nibabel.
    
    '''
    
    def __init__(self, variable_dict=None):
        
        super(BrainData, self).__init__(variable_dict=variable_dict)
        self.subject_data_dict = {}
        self.nifti = NiftiTools()
        self.vector = VectorTools()
    
    
    
    def create_niftis(self, subject_dirs=None, functional_name=None, anatomical_name=None,
                      dxyz=None, talairach_template_path=None, nifti_name=None,
                      within_subject_warp=True, to_template_warp=False):
        
            
        if not nifti_name.endswith('.nii'):
            nifti_name = nifti_name+'.nii'
        
        self.nifti.create_talairach_niftis(subject_dirs, functional_name,
                                           anatomical_name, dxyz,
                                           talairach_template_path, nifti_name,
                                           within_subject_warp, to_template_warp)
        
        
        
        
    def load_niftis_vectors(self, directory, verbose=True):
        '''
        Loads niftis and response vectors from a directory. This function is
        fairly specific. The nifti files should be named in this manner:
        
        prefix_***.nii
        
        Such that prefix denotes a subject and an underscore splits this subject
        from the rest of the nifti filename.
        
        Likewise, the response vector file should be coded:
        
        prefix_***.1D
        
        Such that the prefix matches a prefix for a nifti file!
        NO DUPLICATE PREFIXES - WILL CHOOSE INDISCRIMINATELY
        '''
        
        nifti_paths = sorted(glob.glob(os.path.join(directory, '*.nii')))
        vector_paths = sorted(glob.glob(os.path.join(directory, '*.nii')))
        
        npre = [os.path.split(n)[1].split('_')[0] for n in nifti_paths]
        vpre = [os.path.split(v)[1].split('_')[0] for v in vector_paths]
        
        pairs = []
        for nifti, np in zip(nifti_paths, npre):
            for vector, vp in zip(vector_paths, vpre):
                if np == vp:
                    pairs.append([nifti, self.vector.read(vector)])
                    break
        
        return pairs
        

        
        
        
    def load_niftis_fromdirs(self, subject_dirs, nifti_name, response_vector,
                             verbose=True):
        '''
        Iterates through subject directories, parses the response vector,
        and appends the path to the nifti file for loading later.
        
        Basic support for multiple niftis per subject (just added as different
        key in the subject_data_dict).
        
        '''
        for subject in subject_dirs:
                
            nifti = os.path.join(subject, nifti_name)
            vec = os.path.join(subject, response_vector)

            if not os.path.exists(nifti):
                if verbose:
                    print 'nifti not found: ', nifti
            elif not os.path.exists(vec):
                if verbose:
                    print 'respnse vector not found: ', vec
            else:
                
                respvec = self.vector.read(vec, usefloat=True)
                subject_key = os.path.split(subject)[1]
                
                if verbose:
                    pprint(nifti)
                    print 'appending raw data for subject: ', subject_key
                
                if not subject_key in self.subject_data_dict:
                    self.subject_data_dict[subject_key] = [nifti, respvec]
                    
                else:
                    tag = 2
                    while subject_key+'_'+str(tag) in self.subject_data_dict:
                        tag += 2
                    self.subject_data_dict[subject_key+'_'+str(tag)] = [nifti, respvec]
                
        
        
        
    def parse_trialsvec(self, trialsvec):
        '''
        Simple function to find the indices where Y is not 0. Returns the indices
        vector and the stripped Y vector. Used by masked_data().
        '''
        
        inds = [i for i,x in enumerate(trialsvec) if x != 0.]
        y = [x for x in trialsvec if x != 0]
        return inds, y
    
    
        
    def unmask_Xcoefs(self, Xcoefs, time_points, mask=None, reverse_transpose=True,
                      verbose=True, slice_off_back=0, slice_off_front=0):
        '''
        Reshape the coefficients from a statistical method back to the shape of
        the original brain matrix, so it can be output to nifti format.
        '''
        if mask is None:
            mask = self.original_mask
            
        unmasked = [np.zeros(mask.shape) for i in range(time_points)]
        
        print 'xcoefs sum', np.sum(Xcoefs)
        
        if slice_off_back:
            print np.sum(Xcoefs[:-slice_off_back]), np.sum(Xcoefs[-slice_off_back:])
            Xcoefs = Xcoefs[:-slice_off_back]
        if slice_off_front:
            Xcoefs = Xcoefs[slice_off_front:]
        
        print 'xcoefs sum', np.sum(Xcoefs)
        Xcoefs.shape = (time_points, -1)
        print 'Xcoefs shape', Xcoefs.shape
        
        for i in range(time_points):
            print 'raw coef time sum', np.sum(Xcoefs[i])
            print 'mask, xind shapes', mask.shape, Xcoefs[i].shape
            
            unmasked[i][np.asarray(mask).astype(np.bool)] = np.squeeze(np.array(Xcoefs[i]))
            #unmasked[i][np.asarray(mask).astype(np.bool)] = np.squeeze(np.ones(np.sum(np.asarray(mask).astype(np.bool))))
            
            print 'time ind coef sum', np.sum(unmasked[i])
            if reverse_transpose:
                unmasked[i] = np.transpose(unmasked[i], [2, 1, 0])
        
        unmasked = np.transpose(unmasked, [1, 2, 3, 0])
        
        if verbose:
            print 'Shape of unmasked coefs: ', np.shape(unmasked)
        
        return np.array(unmasked)
        
        
        
    def save_unmasked_coefs(self, unmasked, nifti_filename, affine=None,
                            talairach_template_path='./TT_N27+tlrc.'):
        '''
        Simple function to save the unmasked coefficients to a specified nifti.
        Affine is usually self.mask_affine, but can be specified.
        '''
        
        if affine is None:
            affine = self.mask_affine
            
        if self.verbose:
            print 'erasing old files with prefix:', nifti_filename#[:-4]
            
        glob_remove(nifti_filename)#[:-4])
            
        self.nifti.save_nifti(unmasked, affine, nifti_filename)
        
        time.sleep(0.25)
        
        self.nifti.convert_to_afni(nifti_filename, nifti_filename)#[:-4])
        
        time.sleep(0.25)
        
        subprocess.call(['3drefit','-view','tlrc',nifti_filename+'+orig.'])
        
        
        
    def make_masks(self, mask_path, ntrs, reverse_transpose=True, verbose=True):
        
        '''
        A function that makes the various mask objects.
        '''
        if verbose:
            if reverse_transpose:
                print 'using time-first reverse transposition of nifti matrix'
            else:
                print 'preserving dimensionality of nifti matrix (nt last)'
        
        mask = load_image(mask_path)
        tmp_mask, self.mask_affine, tmp_shape = self.nifti.load_nifti(mask_path)
        mask = np.asarray(mask)
        self.raw_affine = self.mask_affine
            
        if verbose:
            print 'mask shape:', mask.shape
        self.mask_shape = mask.shape
            
        if reverse_transpose:
            mask = np.transpose(mask.astype(np.bool), [2, 1, 0])
        else:
            mask = mask.copy().astype(np.bool)
            
        self.original_mask = mask.copy()
        self.flat_mask = mask.copy()
        self.flat_mask.shape = np.product(mask.shape)
        
        if verbose:
            print 'flat mask shape:', self.flat_mask.shape
            
        nmask = np.not_equal(mask, 0).sum()
        
        if verbose:
            print 'mask shape', mask.shape
        
        self.trial_mask = np.zeros((ntrs, mask.shape[0], mask.shape[1], mask.shape[2]))
        
        if verbose:
            print 'trial mask shape', self.trial_mask.shape
        
        for t in range(ntrs):
            self.trial_mask[t,:,:,:] = mask
            
        self.trial_mask = self.trial_mask.astype(np.bool)   
        
        

    def prepare_greymatter_mask(self, mask_path, greymatter_prefix='greymatter_resamp',
                                afni_greymatter_dset='/Users/span/abin/TT_caez_gw_18+tlrc.',
                                penalty_weight=1., afni_index=0, reverse_transpose=True):
        
        # resample the gray matter mask to the user's mask:
        
        gm_resample_mask = os.path.join(os.path.split(mask_path)[0],greymatter_prefix)
        old_resamps = glob.glob(gm_resample_mask+'*')
        for oresamp in old_resamps:
            try:
                os.remove(oresamp)
            except:
                pass
        
        cmd = ['3dresample','-master', mask_path, '-prefix', gm_resample_mask,
               '-inset', afni_greymatter_dset+'['+str(afni_index)+']']
        
        subprocess.call(cmd)
        
        niicmd = ['3dAFNItoNIFTI', '-prefix', greymatter_prefix, greymatter_prefix+'+tlrc.']
        
        subprocess.call(niicmd)
        
        # for now need to have made the trial mask, etc...
        
        self.grey_matter = np.zeros(self.trial_mask.shape)
        
        gm_mask = load_image(gm_resample_mask+'.nii')
        gm_mask = np.asarray(gm_mask)
        
        self.grey_matter_flat = gm_mask.copy()
        self.grey_matter_flat.shape = np.product(self.grey_matter_flat.shape)
            
        if reverse_transpose:
            gm_mask = np.transpose(gm_mask, [2, 1, 0])
            for tr in range(len(self.trial_mask)):
                self.grey_matter[tr,:,:,:] = (gm_mask[:,:,:]*penalty_weight)+(1.-penalty_weight)
        
        

        
        
    def masked_data(self, nifti, trialsvec, selected_trs=[], mask_path=None, lag=2,
                    reverse_transpose=True, verbose=True):
        
        '''
        This function masks, transposes, and subselects the trials from the nifti
        data.
        --------
        nifti           :   a filepath to the nifti.
        trialsvec       :   numpy array denoting the response variable at the TR of the
                            trial onset.
        selected_trs    :   a list of the trs in the trial to be subselected
        mask_path       :   path to the mask (optional but recommended)
        lag             :   how many TRs to push out the trial (2 recommended)
        '''
        
        if verbose:
            if reverse_transpose:
                print 'using time-first reverse transposition of nifti matrix'
            else:
                print 'preserving dimensionality of nifti matrix (nt last)'
        
        image = load_image(nifti)
            
        if verbose:
            print 'nifti shape:', image.shape
            
        nmask = np.not_equal(self.original_mask, 0).sum()
        
        ntrs = len(selected_trs)
        
        p = np.prod(image.shape[:-1])
        
        trial_inds, response = self.parse_trialsvec(trialsvec)
        
        ntrials = len(trial_inds)
        
        if reverse_transpose:
            X = np.zeros((ntrials, ntrs, nmask))
        else:
            X = np.zeros((ntrials, nmask, ntrs))
        Y = np.zeros(ntrials)
        
        reselect_trs = [x-1 for x in selected_trs]
        
        if reverse_transpose:
            im = np.transpose(np.asarray(image), [3, 2, 1, 0])
        
            for i in range(ntrials):
                if len(im) > trial_inds[i]+reselect_trs[-1]+lag:
                    # OLD VERSION: could only do a continuous range
                    #row = im[trial_inds[i]+reselect_trs[0]+lag:trial_inds[i]+reselect_trs[-1]+1+lag].reshape((ntrs,p))
                    
                    # NEW VERSION: uses list comprehension for any index range
                    row_inds = [trial_inds[i]+lag+x for x in reselect_trs]
                    row = im[row_inds].reshape((ntrs,p))
                    
                    X[i] = row[:,self.flat_mask]
                    Y[i] = response[i]
            
        else:
            im = np.asarray(image)
            
            for i in range(ntrials):
                if im.shape[3] > trial_inds[i]+reselect_trs[-1]+lag:
                    # OLD VERSION: could only do a continuous range
                    #row = im[:,:,:,trial_inds[i]+reselect_trs[0]+lag:trial_inds[i]+reselect_trs[-1]+1+lag].reshape((p,ntrs))

                    # NEW VERSION: uses list comprehension for any index range
                    row_inds = [trial_inds[i]+lag+x for x in reselect_trs]
                    row = im[:,:,:,row_inds].reshape((p,ntrs))
                    
                    X[i] = row[self.flat_mask,:]
                    Y[i] = response[i]
            
        return X, Y
    
    
    
    def create_trial_mask(self, mask_path, ntrs, reverse_transpose=True):
        
        mask = load_image(mask_path)
        tmp_mask, self.mask_affine, tmp_shape = self.nifti.load_nifti(mask_path)
        mask = np.asarray(mask)
        
        if reverse_transpose:
            mask = np.transpose(mask.astype(np.bool), [2, 1, 0])
        else:
            mask = mask.copy().astype(np.bool)
        
        self.original_mask = mask.copy()
        
        print mask.shape
        
        self.trial_mask = np.zeros((ntrs, mask.shape[0], mask.shape[1], mask.shape[2]))
        
        for t in range(ntrs):
            self.trial_mask[t,:,:,:] = mask
            
        self.trial_mask = self.trial_mask.astype(np.bool)
        print self.trial_mask.shape
    
            

    def create_design(self, subject_dirs, nifti_name, respvec_name, selected_trs,
                      mask_path=None, lag=2, reverse_transpose=True):
        
        self.selected_trs = selected_trs
        
        self.load_niftis_fromdirs(subject_dirs, nifti_name, respvec_name)
        
        self.subject_design = {}
        for subject, [image, respvec] in self.subject_data_dict.items():
            sX, sY = self.masked_data(image, respvec, selected_trs=selected_trs,
                                      mask_path=mask_path, lag=lag, reverse_transpose=reverse_transpose)
            sX.shape = (sX.shape[0], np.prod(sX.shape[1:]))
            print 'subject X shape:', sX.shape
            self.subject_design[subject] = [np.array(sX), np.array(sY)]
            
        del(self.subject_data_dict)
            
            
    def create_design_logan_npy(self, subject_npys):
        
        self.subject_design = {}
        for npy in subject_npys:
            subject = npy.split('.')[0]
            cur_data = np.load(npy)
            sX, sY = [], []
            for ind in range(len(cur_data)):
                sY.append(cur_data[ind]['Y'])
                i_X = cur_data[ind]['X'].copy()
                i_X.shape = (i_X.shape[0]*i_X.shape[1])
                #print i_X.shape
                sX.append(i_X)
            self.subject_design[subject] = [np.array(sX), np.array(sY)]
        
        del(self.subject_data_dict)


    def normalize_data(self, data, verbose=True):

        '''
        a typical normalization function: subtracts the means and divides by
        the standard deviation, column-wise
        '''

        if verbose:
            print 'normalizing dataset column-wise:'
            print 'pre-normalization sum:', np.sum(data)

        stdev = np.std(data, axis=0)
        means = np.mean(data, axis=0)

        dnorm = np.zeros(data.shape)
        dnorm = data-means
        dnorm = dnorm/stdev
        
        dnorm = np.nan_to_num(dnorm)

        if verbose:
            print 'post-normalization sum:', np.sum(dnorm)
        
        return dnorm
        












