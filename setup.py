#! /usr/bin/env python
# System imports
import numpy as np
from setuptools import setup, Extension, find_packages
try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

packages = find_packages()

ext_modules = []

if USE_CYTHON:
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
    ext_modules.append(_creflect)


# pyplatypus setup
setup(  name        = 'pyplatypus',
        ext_modules = ext_modules,

        cmdclass = {'build_ext': build_ext},

        description = 'Neutron and X-ray Reflectometry Analysis',
        author      = 'Andrew Nelson',
        author_email = 'andrew.nelson@ansto.gov.au',
        version     = '0.0.1',
        license     = 'BSD',
        url         = 'https://github.com/andyfaff/pyplatypus',
        platforms = ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        classifiers =[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        ],
        packages = packages,
        install_requires = ['numpy', 'scipy', 'lmfit'],
     )
