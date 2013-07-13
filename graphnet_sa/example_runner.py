
import os, sys, glob
from directorytools import subjects as get_subjects
from directorytools import subject_dirs as get_subject_dirs
from graphnet_hook import GraphnetInterface, Gridsearch
from datamanager import BrainData


if __name__ == "__main__":

	'''
	Examples of using the graphnet interface with nifti data.
	'''
	
	#-----------------------------------------------------------#
	# Load in subject data:
	#-----------------------------------------------------------#

	thisdir = os.getcwd()
	subject_top_dir = os.path.split(thisdir)[0]

	subject_folder_names = ['subject1','subject2','subject3','subject4','subject5']

	subject_directories = get_subject_dirs(topdir=subject_top_dir,
		prefixes=subject_folder_names)


	mask = os.path.join(subject_top_dir, 'mask.nii')

	functional_name = 'functional_warped.nii'
	trial_demarcation_vector_name = 'trial_markers.1D'

	lag = 2
	selected_trial_trs = [1,2,3,4,5]

	datamanager = BrainData()

	datamanager.make_masks(mask, len(selected_trial_trs))

	datamanager.create_design(subject_directories,
		functional_name,
		trial_demarcation_vector_name,
		selected_trial_trs,
		lag=lag)

	datamanager.create_XY_matrices(downsample_type='subject',
		with_replacement=True,
		replacement_ceiling=36,
		Ybinary=[1.,-1])

	datamanager.delete_subject_design()

	graphnet = GraphnetInterface(data_obj=datamanager)

	#-----------------------------------------------------------#
	# Basic usage and dumping coefficients:
	#-----------------------------------------------------------#

	graphnet.train_graphnet(datamanager.X, datamanager.Y, 
		trial_mask=datamanager.trial_mask,
		l1=10., l2=100., l3=1000., delta=0.8, adaptive=True)

	coefs = graphnet.coefficients[0].copy()

	unmasked_coefs = datamanager.unmask_Xcoefs(coefs, len(selected_trial_trs),
		slice_off_back=datamanager.X.shape[0])

	datamanager.save_unmasked_coefs(unmasked_coefs, 'graphnet_coef_map')


	#-----------------------------------------------------------#
	# Crossvalidation:
	#-----------------------------------------------------------#


	train_keyword_args = {'trial_mask':datamanager.trial_mask,
						  'l1':10., 'l2':100., 'l3':1000., 'delta':0.8,
						  'adaptive':True}

	graphnet.setup_crossvalidation(folds=5, leave_mod_in=True)

	graphnet.crossvalidate(train_keyword_args)


	#-----------------------------------------------------------#
	# Gridsearching:
	#-----------------------------------------------------------#

	gridsearch = Gridsearch()
	gridsearch.folds = 5
	gridsearch.initial_l1_min = 10.
	gridsearch.initial_l1_max = 60.
	gridsearch.l1_stepsizes = [5.,3.,1.]
	gridsearch.deltas = [.3,.5,.7]

	gridsearch.zoom_gridsearch(graphnet,
		name='graphnet_gridsearch',
		adaptive=True)










