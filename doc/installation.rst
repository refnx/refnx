.. _installation_chapter:

====================================
Installation
====================================

.. _Visual Studio compiler: https://wiki.python.org/moin/WindowsCompilers
.. _miniconda: https://conda.io/miniconda.html
.. _github: https://github.com/refnx/refnx

*refnx* has been tested on Python 2.7, 3.4, 3.5 and 3.6. It requires the
*numpy, scipy, cython, pandas, emcee* packages to work. Additional features
require the *pytest, h5py, xlrd, uncertainties, ptemcee, matplotlib*
packages. To build the bleeding edge code you will need to have access to a
C-compiler to build a couple of Python extensions. C-compilers should be
installed on Linux. On OSX you will need to install Xcode and the command
line tools. On Windows you will need to install the correct
`Visual Studio compiler`_ for your Python version.


Installation into a *conda* environment
=======================================

Perhaps the easiest way to create a scientific computing environment is to use
the `miniconda`_ package manager. Once *conda* has been installed the first
step is to create a *conda* environment.

Creating a conda environment
============================

1. In a shell window create a conda environment and install the
   dependencies. The **-n** flag indicates that the environment is called
   *refnx*.

    ::

     conda create -n refnx python=3.6 numpy scipy cython pandas h5py xlrd pytest

2. Activate the environment that we're going to be working in:

    ::

     # on OSX
     source activate refnx

     # on windows
     activate refnx

3. Install the remaining dependencies:

    ::

     pip install emcee uncertainties ptemcee

Installing from source
=======================

The latest source code can be obtained from `github`_. You can also build the
package from within the refnx git repository (see later in this document).

1. In a shell window navigate into the source directory and build the package.
   If you are on Windows you'll need to start a Visual Studio command window.

    ::

     python setup.py build
     python setup.py install

2. Run the tests, they should all work.

    ::

     python setup.py test

Installing into a conda environment from a released version
===========================================================

1. There are pre-built versions on *conda-forge*, but they're not necessarily at the bleeding edge:

   ::

     conda install -c conda-forge refnx

2. Start up a Python interpreter and make sure the tests run:

    ::

     >>> import refnx
     >>> refnx.test()
