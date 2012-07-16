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

# ezrange extension module
_creflect = Extension("_creflect",
                   ["reflect.i","reflect.c", "refcalc.cpp"],
                   include_dirs = [numpy_include],
                   )

# creflect setup
setup(  name        = "Calculates reflectivity",
        description = "",
        author      = "Andrew Nelson",
        version     = "1.0",
        ext_modules = [_creflect]
        )