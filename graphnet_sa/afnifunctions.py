
import os, sys
import glob
import subprocess
import shutil
from baseclasses import Process





class AfniFunction(object):
    
    
    def __init__(self):
        super(AfniFunction, self).__init__()
        
    
    def _clean_remove(self, files):
        print files
        for file in files:
            print file
            try:
                os.remove(file)
            except:
                pass
            
    def _check_afni_suffix(self, path, suffix='orig'):
        if suffix == 'orig':
            if not path.endswith('+orig.') and not path.endswith('+orig'):
                path = path+'+orig.'
        elif suffix == 'tlrc':
            if not path.endswith('+tlrc.') and not path.endswith('+tlrc'):
                path = path+'+tlrc.'
        return path
            
    
    def _clean(self, glob_prefix, clean_type=None, path=None):
        if path:
            prefix = os.path.join(path, glob_prefix)
        else:
            prefix = glob_prefix
            
        if clean_type is None:
            self._clean_remove(glob.glob(prefix))
            
        elif clean_type is 'orig':
            self._clean_remove(glob.glob(prefix+'+orig*'))

        elif clean_type is 'tlrc':
            print 'cleaning tlrc'
            print prefix
            print glob.glob(prefix+'+tlrc')
            self._clean_remove(glob.glob(prefix+'+tlrc*'))
            
        elif clean_type is 'afni':
            self._clean_remove(glob.glob(prefix+'+orig*'))
            self._clean_remove(glob.glob(prefix+'+tlrc*'))
            
            
    
    
class Copy3d(AfniFunction):
    
    def __init__(self):
        super(Copy3d, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dcopy', input_path, output_path_prefix]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, input, output_prefix):
        
        section = {'command':['3dcopy ${anatomical_nifti} ${anatomical_name}'],
                   'variables':{'anatomical_nifti':input,
                                'anatomical_name':output_prefix},
                   'clean':[['anatomical_name','orig']],
                   'header':'Convert Anatomical'}

        scriptwriter.add_section(section)        
        
        
class TcatBuffer(AfniFunction):
    
    def __init__(self):
        super(TcatBuffer, self).__init__()
        
        
    def __call__(self, functional_path, output_path_prefix,
                 abs_leadin, abs_leadout):
        
        self._clean(output_path_prefix, clean_type='orig')
        cut_dset = functional_path+'['+str(abs_leadin)+'..'+str(abs_leadout)+']'
        cmd = ['3dTcat', '-prefix', output_path_prefix, cut_dset]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, functional, epi_prefix, leadin,
              leadout, cmd_only=False):
        
        clean = {'afni':{'epi_prefix':epi_prefix}}
        header = 'Cut off lead-in and lead-out:'
        
        write_cmd = ['3dTcat -prefix ${epi_prefix} \'${functional_nifti}[${leadin}..${leadout}]\'']
        write_vars = {'functional_nifti':functional,
                      'leadin':leadin, 'leadout':leadout, 'epi_prefix':epi_prefix}
        
        
        if not cmd_only:
            section = {'header':header, 'command':write_cmd,
                       'variables':write_vars, 'clean':[['epi_prefix','orig']]}
        else:
            section = {'command':write_cmd, 'variables':write_vars}

        scriptwriter.add_section(section)        
        
        
        
class Refit(AfniFunction):
    
    def __init__(self):
        super(Refit, self).__init__()
        
        
    def __call__(self, input_path, tr_length):
        
        input_path = self._check_afni_suffix(input_path)
        cmd = ['3dRefit', '-TR', str(tr_length), input_path]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, input, tr_length, cmd_only=False):
        header = 'Refit to ensure correct TR length:'
        write_cmd = ['3drefit -TR ${tr_length} ${epi_prefix}+orig.']
        write_vars = {'epi_prefix':input, 'tr_length':tr_length}
        
        if not cmd_only:
            section = {'header':header, 'command':write_cmd,
                       'variables':write_vars}
        else:
            section = {'command':write_cmd, 'variables':write_vars}
            
        scriptwriter.add_section(section)
            
            
            
class RefittoParent(AfniFunction):
    
    def __init__(self):
        super(RefittoParent, self).__init__()
        
        
    def __call__(self, apar_path, dpar_path):
        
        input_path = self._check_afni_suffix(dpar_path)
        cmd = ['3dRefit', '-apar', apar_path, dpar_path]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, apar, dpar):
        header = 'Refit dataset to anatomical parent:'
        write_cmd = ['3drefit -apar ${refit_apar} ${refit_dpar}+orig.']
        write_vars = {'refit_apar':apar, 'refit_dpar':dpar}
        
        section = {'header':header, 'command':write_cmd, 'variables':write_vars}
        
        scriptwriter.add_section(section)
            
        
        
        
class Tshift(AfniFunction):
    
    def __init__(self):
        super(Tshift, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, tshift_slice, tpattern):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dTshift', '-slice', str(tshift_slice), '-tpattern',
               tpattern, '-prefix', output_path_prefix, input_path]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, input, output_prefix, tshift_slice, tpattern,
              cmd_only=False):
        
        header = 'Time slice correction:'
        write_cmd = ['3dTshift -slice ${tshift_slice} -tpattern ${tpattern} -prefix ${epits_prefix} ${epi_prefix}+orig.']
        write_vars = {'epi_prefix':input, 'tshift_slice':tshift_slice,
                       'tpattern':tpattern, 'epits_prefix':output_prefix}
        
        if not cmd_only:
            section = {'header':header, 'command':write_cmd, 'variables':write_vars,
                       'clean':[['epits_prefix', 'orig']]}
        else:
            section = {'command':write_cmd, 'variables':write_vars}
            
        scriptwriter.add_section(section)
            
            
            
        
        
        
class TcatDatasets(AfniFunction):
    
    def __init__(self):
        super(TcatDatasets, self).__init__()
        
        
    def __call__(self, input_paths, output_path_prefix, cleanup=True):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dTcat', '-prefix', output_path_prefix]+input_paths
        subprocess.call(cmd)
        
        if cleanup:
            for ip in input_paths:
                self._clean(ip, clean_type='orig')
                
                
    def write(self, scriptwriter, inputs, output_prefix):
        
        header = 'Concatenate epis into functional dataset:'
        write_cmd = ['3dTcat -prefix ${functional_name} ${epi_names}']
        if type(inputs) not in (list, tuple):
            inputs = [inputs]
            
        inputs = [x.rstrip('+orig.') for x in inputs if x.startswith('+orig')
                  or x.startswith('+orig.')]
        inputs = ' '.join([x+'+orig' for x in inputs])
        write_vars = {'epi_names':inputs, 'functional_name':output_prefix}
        
        section = {'header':header, 'command':write_cmd, 'variables':write_vars,
                   'clean':[['functional_name','orig']]}
        
        scriptwriter.add_section(section)
                
                

class Volreg(AfniFunction):
    
    def __init__(self):
        super(Volreg, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, motionfile, volreg_base):
        
        self._clean(output_path_prefix, clean_type='orig')
        self._clean(motionfile)
        cmd = ['3dvolreg','-Fourier','-twopass','-prefix', output_path_prefix,
               '-base', str(volreg_base), '-dfile', motionfile, input_path]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, input, output, motionfile, volreg_base):
        
        header = 'Motion correction:'
        write_cmd = ['3dvolreg -Fourier -twopass -prefix ${output} -base ${volreg_base} -dfile ${motionfile_name} ${input}+orig']
        write_vars = {'input':input,
                       'output':output, 'volreg_base':volreg_base,
                       'motionfile_name':motionfile}
        
        section = {'header':header, 'variables':write_vars, 'command':write_cmd,
                   'clean':[['output','orig'],['3dmotion.1D',None]]}
        
        scriptwriter.add_section(section)

        
        
        
    
class Smooth(AfniFunction):
    
    def __init__(self):
        super(Smooth, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, blur_kernel):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dmerge', '-prefix', output_path_prefix, '-1blur_fwhm',
               str(blur_kernel), '-doall', input_path]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, input, output, blur_kernel):
        
        header = 'Blur dataset:'
        write_cmd = ['3dmerge -prefix ${output} -1blur_fwhm ${blur_kernel} -doall ${input}+orig']
        write_vars = {'blur_kernel':blur_kernel, 'input':input, 'output':output}
        
        section = {'header':header, 'variables':write_vars, 'command':write_cmd,
                   'clean':[['output','orig']]}
        
        scriptwriter.add_section(section)
        
        
        

class NormalizePSC(AfniFunction):
    
    def __init__(self):
        super(NormalizePSC, self).__init__()
        
        
    def __call__(self, input_path, ave_path_prefix, output_path_prefix, trs, expression):
        
        self._clean(output_path_prefix, clean_type='orig')
        self._clean(ave_path_prefix, clean_type='orig')
        
        ave_cmd = ['3dTstat', '-prefix', ave_path_prefix,
                   input_path+'[0..'+str(trs)+']']
        refit_cmd = ['3drefit', '-abuc', ave_path_prefix+'+orig']
        calc_cmd = ['3dcalc', '-datum', 'float', '-a',
                    input_path+'[0..'+str(trs)+']', '-b',
                    ave_path_prefix+'+orig', '-expr', expression, '-prefix',
                    output_path_prefix]
        
        subprocess.call(ave_cmd)
        subprocess.call(refit_cmd)
        subprocess.call(calc_cmd)

            
            
    def write(self, scriptwriter, input, output, average, trs, expression):
        
        header = 'Normalize dataset:'
        
        tstat_cmd = ['3dTstat -prefix ${average} \'${input}+orig[0..${trs}]]\'']
        tstat_vars = {'average':average, 'input':input, 'trs':trs}
        
        average_d = {'header':header, 'command':tstat_cmd, 'variables':tstat_vars,
                   'clean':[['average','orig']]}
        
        scriptwriter.add_section(average_d)
        
        refit_cmd = ['3drefit -abuc ${average}+orig']
        refit_vars = {'average':average}
        
        refit = {'command':refit_cmd, 'variables':refit_vars}
        
        scriptwriter.add_section(refit)
        
        calc_cmd = ['3dcalc -datum float -a \'${input}+orig[0..${trs}]\' -b ${average} -expr \"${expression}\" -prefix ${output}']
        calc_vars = {'input':input, 'output':output,
                     'average':average, 'trs':trs,
                     'expression':expression}
        
        norm = {'command':calc_cmd, 'variables':calc_vars, 'clean':[['output','orig']]}
        
        scriptwriter.add_section(norm)
            
            
            
            
class HighpassFilter(AfniFunction):
    
    def __init__(self):
        super(HighpassFilter, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix, highpass_value):
        
        self._clean(output_path_prefix, clean_type='orig')
        cmd = ['3dFourier', '-prefix', output_path_prefix, '-highpass',
               str(highpass_value), input_path]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, input, output, highpass_value):
        
        header = 'Fourier highpass filter:'
        write_cmd = ['3dFourier -prefix ${output} -highpass ${highpass_value} ${input}+orig']
        write_vars = {'input':input, 'output':output,
                      'highpass_value':highpass_value}
        
        section = {'header':header, 'command':write_cmd, 'variables':write_vars,
                   'clean':[['output','orig']]}
        
        scriptwriter.add_section(section)
        
        
        
class TalairachWarp(AfniFunction):
    
    def __init__(self):
        super(TalairachWarp, self).__init__()
        
        
    def __call__(self, functional_path, anatomical_path, template_path):
        
        import os
        curdir = os.getcwd()
        os.chdir(os.path.split(anatomical_path)[0])
        anatomical_path = os.path.split(anatomical_path)[1]
        print anatomical_path
        self._clean(anatomical_path[:-5], clean_type='tlrc')
        cmd = ['@auto_tlrc', '-warp_orig_vol', '-suffix', 'NONE',
               '-base', template_path, '-input', anatomical_path]
        refit_cmd = ['3drefit', '-apar', anatomical_path[:-5]+'+tlrc',
                     functional_path]
        
        subprocess.call(cmd)
        subprocess.call(refit_cmd)
        os.chdir(curdir)
        
        
    def write(self, scriptwriter, functional, anatomical, template_path):
        
        header = 'Warp to talairach space:'
        tlrc_cmd = ['@auto_tlrc -warp_orig_vol -suffix NONE -base ${talairach_template} -input ${anatomical}+orig']
        tlrc_vars = {'anatomical':anatomical,
                     'talairach_template':template_path}
        
        tlrc_section = {'header':header, 'variables':tlrc_vars, 'command':tlrc_cmd,
                        'clean':[['anatomical','tlrc']]}
        
        scriptwriter.add_section(tlrc_section)
        
        refit_cmd = ['3drefit -apar ${anatomical}+orig ${functional}+orig']
        refit_vars = {'anatomical':anatomical,
                      'functional':functional}
        
        refit_section = {'command':refit_cmd, 'variables':refit_vars}
        scriptwriter.add_section(refit_section)


class Adwarp(AfniFunction):
    
    def __init__(self):
        super(Adwarp, self).__init__()
        
        
    def __call__(self, apar_path, dpar_path, output_path_prefix, dxyz=1.,
                 adwarp_type='tlrc', force=False):
        
        self._clean(output_path_prefix, clean_type=adwarp_type)
        if not force:
            cmd = ['adwarp', '-apar', apar_path, '-dpar', dpar_path,
                   '-dxyz', str(dxyz), '-prefix', output_path_prefix]
        else:
            cmd = ['adwarp', '-apar', apar_path, '-dpar', dpar_path,
                   '-dxyz', str(dxyz), '-force', '-prefix', output_path_prefix]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, apar, dpar, output_prefix, dxyz=1., force=False):
        
        header = 'Adwarp dataset:'
        if not force:
            adwarp_cmd = ['adwarp -apar ${apar_dset} -dpar ${dpar_dset} -dxyz ${adwarp_dxyz} -prefix ${adwarp_output}']
        else:
            adwarp_cmd = ['adwarp -force -apar ${adwarp_apar} -dpar ${adwarp_dpar} -dxyz ${adwarp_dxyz} -prefix ${adwarp_output}']
        adwarp_vars = {'adwarp_apar':apar, 'adwarp_dpar': dpar,
                       'adwarp_dxyz':str(dxyz), 'adwarp_output':output_prefix}
        
        
        section = {'header':header, 'command':adwarp_cmd, 'variables':adwarp_vars,
                   'clean':[['adwarp_output', 'tlrc']]}
        
        scriptwriter.add_section(section)

        
        
class Automask(AfniFunction):
    
    def __init__(self):
        super(Automask, self).__init__()
        
        
    def __call__(self, mask_dset_path, output_path_prefix, clfrac=.3,
                 dset_type='tlrc'):
        
        self._clean(output_path_prefix, clean_type=dset_type)
        cmd = ['3dAutomask','-prefix',output_path_prefix, '-clfrac',
               str(clfrac), mask_dset_path]
        subprocess.call(cmd)
        
        
    
'''BROKEN (scriptwriter needs to be updated)'''
class AfnitoNifti(AfniFunction):
    
    def __init__(self):
        super(AfnitoNifti, self).__init__()
        
        
    def __call__(self, input_path, output_path_prefix):
        
        self._clean(output_path_prefix+'.nii')
        cmd = ['3dAFNItoNIFTI', '-prefix', output_path_prefix, input_path]
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, input, output_prefix):
        
        header = 'Convert Afni file to Nifti:'
        clean = {'standard':{'nifti_output':output_prefix+'.nii'}}
        a2n_cmd = ['3dAFNItoNIFTI -prefix ${nifti_output} ${afni_input}']
        a2n_vars = {'nifti_output':output_prefix, 'afni_input':input}
        
        section = {'header':header, 'command':a2n_cmd, 'variables':a2n_vars,
                   'clean':[['nifti_output','nii']]}
        
        scriptwriter.add_section(section)


    
class MaskAve(AfniFunction):
    
    def __init__(self):
        super(MaskAve, self).__init__()
        
        
    def __call__(self, mask_path, dataset_path, mask_area_strs=['l','r','b'],
            mask_area_codes=[[1,1],[2,2],[1,2]], tmp_tc_dir='raw_tc'):
        
        subject_path = os.path.split(dataset_path)[0]
        subject_name = os.path.split(subject_path)[1]
        
        mask_name = os.path.split(mask_path)[1].split('+')[0]
        
        print 'maskave', subject_name, mask_name
        
        tmpdir = os.path.join(subject_path, tmp_tc_dir)
        try:
            os.makedirs(tmpdir)
        except:
            pass
        
        for area, codes in zip(mask_area_strs, mask_area_codes):
            
            raw_tc = '_'.join([subject_name, area, mask_name, 'raw.tc'])
            raw_tc = os.path.join(tmpdir, raw_tc)
            
            self._clean(raw_tc)
            
            cmd = ['3dmaskave', '-mask', mask_path, '-quiet', '-mrange',
                   str(codes[0]), str(codes[1]), dataset_path]
            
            fcontent = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            fcontent.wait()
            fcontent = fcontent.communicate()[0]
            
            fid = open(raw_tc, 'w')
            fid.write(fcontent)
            fid.close()
            
    
    
class Info(AfniFunction):
    
    def __init__(self):
        super(Info, self).__init__()
        
    
    def __call__(self, dataset_path, output_filepath=None):
        if output_filepath:
            fid = open(output_filepath, 'w')
        
        cmd = ['3dinfo', dataset_path]
        
        fcontent = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        fcontent.wait()
        fcontent = fcontent.communicate()[0]
        
        if output_filepath:
            fid.write(fcontent)
            fid.close()
            
        return fcontent
    
    
    
class FractionizeMask(AfniFunction):
    
    def __init__(self):
        super(FractionizeMask, self).__init__()
        
        
    def __call__(self, mask_path, dataset_path, anat_path, subject_mask_suffix='r'):
        
        subject_path = os.path.split(dataset_path)[0]
        print mask_path
        mask_name = os.path.split(mask_path)[1].split('+')[0]
        print mask_name
        
        subject_mask = os.path.join(subject_path, mask_name+subject_mask_suffix+'+orig')
        self._clean(subject_mask+'*')
        
        cmd = ['3dfractionize', '-template', dataset_path, '-input', mask_path,
           '-warp', anat_path, '-clip', '0.1', '-preserve', '-prefix',
           subject_mask]
        
        subprocess.call(cmd)
        return subject_mask
        
        
        
class MaskDump(AfniFunction):
    
    def __init__(self):
        super(MaskDump, self).__init__()
        self.fractionize = FractionizeMask()
        self.maskave = MaskAve()
        
        
    def run(self, dataset_path, anat_path, mask_paths, mask_area_strs=['l','r','b'],
            mask_area_codes=[[1,1],[2,2],[1,2]]):
        
        subject_path = os.path.split(dataset_path)[0]
        
        #if clean_type(mask_paths) in (list, tuple):
        #    for mask in mask_paths:
        #        subject_mask = self.fractionize.run(mask, dataset_path, anat_path)
        #        self.maskave.run(subject_mask, dataset_path, mask_area_strs=mask_area_strs,
        #                         mask_area_codes=mask_area_codes)
                
        
        for maskp in mask_paths:
            subject_mask = self.fractionize(maskp, dataset_path, anat_path)
            self.maskave(subject_mask, dataset_path, mask_area_strs=mask_area_strs,
                         mask_area_codes=mask_area_codes)
        
        
    def run_over_subjects(self, subject_dirs=None, functional_name=None, anatomical_name=None,
                          mask_names=None, mask_dir=None, mask_area_strs=['l','r','b'],
                          mask_area_codes=[[1,1],[2,2],[1,2]]):
        
        self.mask_names = mask_names
        self.functional_name = functional_name
        self.anatomical_name = anatomical_name
        self.mask_dir = mask_dir
        self.subject_dirs = subject_dirs
        
        for i, mask in enumerate(self.mask_names):
            if not mask.endswith('+tlrc'):
                self.mask_names[i] = mask+'+tlrc'
                
        mask_paths = [os.path.join(self.mask_dir, name) for name in self.mask_names]

        for dir in self.subject_dirs:
            dataset_path = os.path.join(dir, self.functional_name)
            anat_path = os.path.join(dir, self.anatomical_name)
            
            self.run(dataset_path, anat_path, mask_paths, mask_area_strs=mask_area_strs,
                     mask_area_codes=mask_area_codes)
            
            

class RegAna(AfniFunction):
    
    def __init__(self):
        super(AfniFunction, self).__init__()
        
        
    def __call__(self, subject_dirs, dataset_name, output_path, Xrows, modelX=[], null=[0],
                 rmsmin=0):
        
        dataset_paths = [os.path.join(sdir, dataset_name) for sdir in subject_dirs]
        
        self._clean(output_path)
        
        if not modelX:
            modelX = [x+1 for x in range(len(Xrows[0]))]
            
        cols = len(Xrows[0])
        rows = len(Xrows)
        
        cmd = ['3dRegAna', '-rows', str(rows), '-cols', str(cols)]
        
        for dset, sX in zip(dataset_paths, Xrows):
            subrow = ['-xydata']+[str(x) for x in sX]+[dset]
            cmd.extend(subrow)
            
        cmd.extend(['-rmsmin', str(rmsmin)])
        cmd.extend(['-model']+[str(x) for x in modelX]+[':']+[str(x) for x in null])
        cmd.extend(['-bucket','0',output_path])
        
        subprocess.call(cmd)
        
        
    def write(self, scriptwriter, subjects, dataset_name, dataset_ind, output_name, Xrows,
              modelX=[], null=[0], rmsmin=0):
        
        header = '3dRegAna auto-script:'
        
        if not (dataset_name.endswith('+tlrc')) or not (dataset_name.endswith('+tlrc.')):
            dataset_name = dataset_name+'+tlrc'
            
        dataset_with_ind = dataset_name+'['+str(dataset_ind)+']'
        
        dataset_paths = [os.path.join('..', sub, '${regana_dataset}') for sub in subjects]
        
        if not modelX:
            modelX = [x+1 for x in range(len(Xrows[0]))]
        
        cols = len(Xrows[0])
        rows = len(Xrows)
        
        cmd = ['3dRegAna -rows '+str(rows)+' -cols '+str(cols)+' \\']
        subcmd = []
        for dset, sX in zip(dataset_paths, Xrows):
            subrow = '-xydata '+' '.join([str(x) for x in sX])+' '+dset+' \\'
            subcmd.append(subrow)
                
        
        subcmd.append('-rmsmin '+str(rmsmin))
        subcmd.append('-model '+' '.join([str(x) for x in modelX])+' : '+' '.join([str(x) for x in null]))
        subcmd.append('-bucket  0 '+output_name)
        
        cmd.append(subcmd)
        
        regana_vars = {'regana_dataset':dataset_with_ind}
        
        section = {'header':header, 'command':cmd, 'variables':regana_vars}
        
        scriptwriter.add_section(section)

        
        
        
        
    
class AfniWrapper(object):
    
    def __init__(self):
        super(AfniWrapper, self).__init__()
        
        self.maskave = MaskAve()
        self.fractionize = FractionizeMask()
        self.maskdump = MaskDump()
        self.talairachwarp = TalairachWarp()
        self.highpassfilter = HighpassFilter()
        self.normalize = NormalizePSC()
        self.smooth = Smooth()
        self.volreg = Volreg()
        self.tcatdsets = TcatDatasets()
        self.tcatbuffer = TcatBuffer()
        self.tshift = Tshift()
        self.refit = Refit()
        self.refit_apar = RefittoParent()
        self.copy3d = Copy3d()
        self.adwarp = Adwarp()
        self.automask = Automask()
        self.afnitonifti = AfnitoNifti()
        self.info = Info()
        self.regana = RegAna()
        
        
        
        
        
        
        
        
    