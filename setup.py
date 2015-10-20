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

# refnx setup
info = {
        'name': 'refnx',
        'description': 'Neutron and X-ray Reflectometry Analysis',
        'author': 'Andrew Nelson',
        'author_email': 'andrew.nelson@ansto.gov.au',
        'version': '0.0.1',
        'license': 'BSD',
        'url': 'https://github.com/andyfaff/refnx',
        'platforms': ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        'classifiers': [
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
        'packages': packages,
        'install_requires': ['numpy', 'scipy', 'lmfit', 'uncertainties']
        }

if USE_CYTHON:
    ext_modules = []

    _creflect = Extension(
                          name='refnx.analysis._creflect',
                          sources=['src/_creflect.pyx', 'src/refcalc.cpp'],
                          include_dirs=[numpy_include],
                          language='c',
                          extra_compile_args=[''],
                          extra_link_args=['']
                          #extra_link_args = ['-lpthread']
                          )

    ext_modules.append(_creflect)
    info['cmdclass'] = {'build_ext': build_ext}
    info['ext_modules'] = ext_modules

try:
    setup(**info)
except ValueError:
    # there probably wasn't a C-compiler (windows). Try removing extension
    # compilation
    print("")
    print("*****WARNING*****")
    print("You didn't try to build the Reflectivity calculation extension."
          " Calculation will be slow, falling back to pure python."
          " To compile extension install cython. If installing in windows you"
          " should then install from Visual Studio command prompt (this makes"
          " C compiler available")
    print("*****************")
    print("")
    info.pop('cmdclass')
    info.pop('ext_modules')
    setup(**info)