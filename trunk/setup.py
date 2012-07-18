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
_creflect = Extension("pyplatypus._creflect",
                   ["src/reflect.i","src/reflect.c", "src/refcalc.cpp"],
                   include_dirs = [numpy_include],
                   )

# creflect setup
setup(  name        = "Calculates reflectivity",
        description = "",
        author      = "Andrew Nelson",
        version     = "1.0",
        packages = ['pyplatypus', 'pyplatypus.reduce', 'pyplatypus.reflect', 'pyplatypus.creflect'],
        ext_modules = [_creflect]
        requires ['numpy', 'scipy']
        )