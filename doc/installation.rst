.. _installation_chapter:

====================================
Installation
====================================

.. _Visual Studio compiler: https://wiki.python.org/moin/WindowsCompilers
.. _miniconda: https://conda.io/miniconda.html
.. _github: https://github.com/refnx/refnx
.. _homebrew: https://brew.sh/

*refnx* has been tested on Python 3.8, 3.9, 3.10, and 3.11. It requires the
*numpy, scipy, cython* packages to work. Additional features
require the *pytest, h5py, xlrd, uncertainties, attrs, matplotlib, Jupyter,*
*ipywidgets, traitlets, tqdm, pandas, qtpy, pyqt6, periodictable, pymc,
pytensor*. To build the bleeding edge code you will need to have access to a
C-compiler to build a couple of Python extensions. C-compilers should be
installed on Linux. On OSX you will need to install Xcode and the command line
tools. On Windows you will need to install the correct
`Visual Studio compiler`_ for your Python version.


Installation into a *conda* environment
=======================================

Perhaps the easiest way to create a scientific computing environment is to use
the `miniconda`_ package manager. Once *conda* has been installed the first
step is to create a *conda* environment.

Creating a conda environment
============================

1. In a shell window create a conda environment and install the dependencies.
   Note that not all of these dependencies are essential, but they are required
   to run the full refnx test suite. The **-n** flag indicates that the
   environment is called *refnx*.

    ::

     conda create -n refnx python=3.9

2. Activate the environment that we're going to be working in:

    ::

     # on OSX
     conda activate refnx

     # on windows
     activate refnx

3. Install the remaining dependencies:

    ::

     python -m pip install refnx[all]

Installing with pip
===================

There are refnx wheels available for macOS/Windows/Linux on PyPI. Using the
[all] modifier means that all refnx's optional dependencies will also be
installed.

    ::

     # install refnx and all optional dependencies
     python -m pip install refnx[all]

     # alternatively just install refnx itself
     python -m pip install refnx

Installing into a conda environment from a released version
===========================================================

1. There are pre-built versions on *conda-forge*:

   ::

     conda install -c conda-forge refnx

2. Start up a Python interpreter and make sure the tests run:

    ::

     >>> import refnx
     >>> refnx.test()

Installing from source
=======================

The latest source code can be obtained from `github`_. You can build the
package from within the refnx git repository.

1. [macOS only] If you wish to enable the parallelised calculation of
   reflectivity with OpenMP, then you will need to install *libomp*. This is
   easily achieved via `homebrew`_, and the setting of environment variables.
   The refnx kernel that uses OpenMP is not normally any better than the
   default C version.

    ::

     brew install libomp
     export CC=clang
     export CXX=clang++
     export CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp"
     export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
     export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
     export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
     export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib

2. In a shell window navigate into the source directory and build the package.
   If you are on Windows you'll need to start a Visual Studio command window.

    ::

     pip install .

3. Run the tests, they should all work.

    ::

     python setup.py test
