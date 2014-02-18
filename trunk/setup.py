#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# creflect extension module
_creflect = Extension("pyplatypus.analysis.__creflect",
                      ["src/reflect.i","src/reflect.c", "src/refcalc.cpp"],
                      include_dirs = [numpy_include],
                      undef_macros=['NDEBUG'])

# pyplatypus setup
setup(  name        = "pyplatypus",
        description = "Processing and Analysing Reflectometry data",
        author      = "Andrew Nelson",
        version     = "1.0",
        ext_modules = [_creflect],
        packages = ['pyplatypus', 'pyplatypus.reduce', 'pyplatypus.analysis',
                    'pyplatypus.dataset', 'pyplatypus.util'],
        requires = ['numpy', 'scipy', 'numdifftools']
     )