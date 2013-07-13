
import re, os, glob
import shutil



def dirs(topdir=os.getcwd(), prefixes=[], exclude=[], regexp=None, initial_glob='*'):
    
    files = [f for f in glob.glob(os.path.join(topdir,initial_glob)) if os.path.isdir(f)]
    if regexp:
        files = [f for f in files if re.search(regexp, os.path.split(f)[1])]
    files = [f for f in files if not any([os.path.split(f)[1].startswith(ex) for ex in exclude])]
    if prefixes:
        files = [f for f in files if any([os.path.split(f)[1].startswith(pr) for pr in prefixes])]
        
    return sorted(files)



def subject_dirs(topdir=os.getcwd(), prefixes=[], exclude=[], initial_glob='*'):
    return dirs(topdir=topdir, prefixes=prefixes, exclude=exclude,
                initial_glob=initial_glob,regexp=r'[a-zA-Z]\d\d\d\d\d\d')
    
    


def subjects(max_length=None, topdir=os.getcwd(), prefixes=[], exclude=[],
             initial_glob='*'):
    subjdirs=subject_dirs(topdir=topdir, prefixes=prefixes, exclude=exclude,
                          initial_glob=initial_glob)
    if not max_length:
        return [os.path.split(x)[1] for x in subjdirs]
    else:
        return [os.path.split(x)[1][0:min(max_length,len(os.path.split(x)[1]))] for x in subjdirs]
        


def consprint(input_list, python=True, bash=True):
    
    if python:
        print '[\''+'\',\''.join([str(x) for x in input_list])+'\']'
        
    if bash:
        print '( '+' '.join([str(x) for x in input_list])+' )'




def glob_remove(file_prefix, suffix='*'):
    candidates = glob.glob(file_prefix+suffix)
    for c in candidates:
        try:
            os.remove(c)
        except:
            pass
        
        
        
class DirectoryCleaner(object):
    
    def __init__(self, prefixes=[], exclude=[], topdir=None):
        super(DirectoryCleaner, self).__init__()
        self.topdir = topdir or os.getcwd()
        if prefixes:
            self.dirs = parse_dirs(topdir=self.topdir, prefixes=prefixes,
                                   exclude=exclude)
        else:
            self.dirs = subject_dirs(topdir=topdir, exclude=exclude)
        self.types, self.files = [], []
        
    def walk_directories(self, function):
        for dir in self.dirs:
            os.chdir()
            self.files = glob.glob('./*')
            function()
            os.chdir('..')
        self.files = []
        
    def action_flag(self, action):
        if action == 'remove':
            for file in self.files:
                for suffix in self.types:
                    if file.endswith(suffix): os.remove(file)
        elif action == 'move':
            for flag in self.types:
                if not flag.endswith('HEAD') or not flag.endswith('BRIK'):
                    dir_name = 'old_'+flag.strip('.')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    for file in self.files:
                        if file.endswith(flag): shutil.move(file, dir_name)
                else:
                    dir_name = 'old_afni'
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    for file in self.files:
                        if file.endswith(flag): shutil.move(file, dir_name)
        
    def remove(self, *args):
        print os.getcwd()
        if args:
            self.types = args
            print self.types
        if not self.files:
            self.walk_directories(self.remove)
        else:
            self.action_flag('rm')
            
    def move(self, *args):
        print os.getcwd()
        if args:
            self.types = args
            print self.types
        if not self.files:
            self.walk_directories(self.move)
        else:
            self.action_flag('mv')