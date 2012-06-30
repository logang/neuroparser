
""" Builders for Cython.

This module will add the following builders to the environment :
* Cython : compile .pyx files to .c files (e.g. env.Cython("foo.pyx"))
* CythonModule : compile .pyx files to a shared library, loadable from
Python (e.g. env.CythonModule("my_module", ["foo.pyx", "bar.pyx"]))
"""

import Cython.Compiler.Main
from SCons.Builder import Builder

def module_builder(env, module_name, source) :
    """ Pseudo-builder for a cython module.
    """
    c_source = env.Cython(source)
    env.PythonModule(module_name, c_source)

    def exists(env):
        return env.Detect("cython")

    def generate(env):
        env["BUILDERS"]["Cython"] = Builder(action="cython $SOURCE",
                                            suffix = ".c", src_suffix = ".pyx")
        env.AddMethod(module_builder, "CythonModule") 
