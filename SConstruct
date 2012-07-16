import numpy, os

# Get env and make Python extensions look up compiler flags and include paths etc. from distutils)
env = Environment(ENV=os.environ, tools = ["default"], toolpath = '.', PYEXT_USE_DISTUTILS=True)

# Set tools and numpy path
env.Tool("pyext")
env.Tool("cython")
env.Append(PYEXTINCPATH=[numpy.get_include()])

# Override location of Cython 
#env.Replace(CYTHON="python /Library/Frameworks/EPD64.framework/Versions/7.0/lib/python2.7/site-packages/cython.py")

# Specify extensions to be compiled
#env.PythonExtension('optimization.cwpath.cwpath', ['./optimization/cwpath/cwpath.pyx'])
#env.PythonExtension('optimization.cwpath.graphnet', ['./optimization/cwpath/graphnet.pyx'])
#env.PythonExtension('optimization.cwpath.regression', ['./optimization/cwpath/regression.pyx'])
#env.PythonExtension('./optimization/cwpath/lasso', ['./optimization/cwpath/lasso.pyx'])

Export('env')
SConscript('optimization/cwpath/SConscript')
