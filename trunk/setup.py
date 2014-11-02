#! /usr/bin/env python
# System imports
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np                           # <---- New line

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

# creflect extension module
_creflect = Extension(
             name = 'pyplatypus.analysis._creflect',
             sources=['src/_creflect.pyx', 'src/refcalc.cpp'],
             include_dirs = [numpy_include],
             language = 'c',
             extra_link_args = ['-lpthread']
                 # libraries=
                 # extra_compile_args = "...".split(),
             )

# pyplatypus setup
setup(  name        = "pyplatypus",
        cmdclass = {'build_ext': build_ext},
        description = "Processing and Analysing Reflectometry data",
        author      = "Andrew Nelson",
        version     = "1.0",
        ext_modules = [_creflect],
        packages = ['pyplatypus', 'pyplatypus.reduce', 'pyplatypus.analysis',
                    'pyplatypus.dataset', 'pyplatypus.util'],
        requires = ['numpy', 'scipy', 'numdifftools']
     )
