#!/usr/bin/env python

# Imports
import os, sys, string, numpy
from setuptools import setup, find_packages

from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Compiler import Main

#-----------------------------------------------------------------------------#
# Functions

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#-----------------------------------------------------------------------------#
# setup call

#ext_modules = [Extension("optimization.cwpath.lasso", ["./optimization/cwpath/lasso.pyx"],include_dirs=[numpy.get_include()])]
ext_modules = [Extension("optimization.cwpath.graphnet", ["./optimization/cwpath/graphnet.pyx"],include_dirs=[numpy.get_include()])]
ext_modules += [Extension("cwpath.graphnet", ["./optimization/cwpath/graphnet.pyx"],include_dirs=[numpy.get_include()])]
ext_modules += [Extension("optimization.cwpath.regression", ["./optimization/cwpath/regression.pyx"],include_dirs=[numpy.get_include()])]
ext_modules += [Extension("optimization.cwpath.cwpath", ["./optimization/cwpath/cwpath.pyx"],include_dirs=[numpy.get_include()])]

setup(
    name = "Neuroparser",
    version = "0.1",
    packages = find_packages(),
    py_modules = ['optimization','gui','examples','optimization.cwpath','optimization.cwpath.graphnet'],
    cmdclass = {'build_ext' : build_ext },
    ext_modules = ext_modules,
    # Project uses Numpy, Scipy, Matplotlib, h5py, multiprocessing
    install_requires = ['numpy>=1.3', 'scipy>=0.7', 'matplotlib>=0.99', 'h5py>=1.3', 'multiprocessing>=0.7'],

    # metadata for upload to PyPI
    author = "Logan Grosenick, Brad Klingenberg, Jonathan Taylor",
    author_email = "logang@gmail.com",
    description = "Neuroparser is a package for applying supervised and unsupervised statistical learning methods to large neuroimaging data.", 
    long_description=read('README'),
    license = "PSF",
    keywords = ["fmri", "sparse", "structured", "multivariate", "calcium imaging", "neuronal dynamics"],
    url = "https://github.com/logang/neuroparser", 

    classifiers=[
        "Development Status :: Alpha",
    ],
)



# import os, sys
# import string

# from Cython.Compiler import Main
# from distutils.extension import Extension
# from Cython.Distutils import build_ext

# def cython_extension(srcfile):
#     options = Main.CompilationOptions(include_path=[os.path.join(os.path.abspath(os.path.dirname(__file__)), 'include')])
#     Main.compile(srcfile, options=options)

# def configuration(parent_package='',top_path=None):
#     from numpy.distutils.misc_util import Configuration
#     config = Configuration(None,parent_package,top_path)
#     config.add_subpackage('optimization/cwpath')
#     return config

# if __name__ == '__main__':

# #    ext_modules = [Extension("optimization.cwpath.lasso", ["./optimization/cwpath/lasso.pyx"])]
#     ext_modules = [Extension("optimization.cwpath.graphnet", ["./optimization/cwpath/graphnet.pyx"])]
#     ext_modules += [Extension("optimization.cwpath.regression", ["./optimization/cwpath/regression.pyx"])]
#     ext_modules += [Extension("optimization.cwpath.cwpath", ["./optimization/cwpath/cwpath.pyx"])]

#     #cython_extension("optimization/cwpath/lasso.pyx")
# #    cython_extension("optimization/cwpath/graphnet.pyx")
# #    cython_extension("optimization/cwpath/regression.pyx")
# #    cython_extension("optimization/cwpath/cwpath.pyx")
    
#     from numpy.distutils.core import setup

#     c = configuration(top_path='',
#                       ext_modules=ext_modules,
#                       cmdclass = {'build_ext': build_ext}
#                       ).todict()
#     setup(**c)
