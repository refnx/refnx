"""Helpers for OpenMP support during the build."""

# This code is adapted for a large part from the astropy openmp helpers, which
# can be found at: https://github.com/astropy/astropy-helpers/blob/master/astropy_helpers/openmp_helpers.py  # noqa
# Further adapted from sklearn for refnx.


import os
import sys
import glob
import tempfile
import textwrap
import subprocess

from numpy.distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler
from distutils.errors import CompileError, LinkError


CCODE = textwrap.dedent(
    """\
    #include <omp.h>
    #include <stdio.h>
    int main(void) {
    #pragma omp parallel
    printf("nthreads=%d\\n", omp_get_num_threads());
    return 0;
    }
    """)


def get_openmp_flag(compiler):
    if hasattr(compiler, 'compiler'):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ('icc' in compiler or 'icl' in compiler):
        return ['/Qopenmp']
    elif sys.platform == "win32":
        return ['/openmp']
    elif sys.platform == "darwin" and ('icc' in compiler or 'icl' in compiler):
        return ['-openmp']
    elif sys.platform == "darwin":
        # default for macOS, assuming Apple-clang
        # -fopenmp can't be passed as compile flag when using Apple-clang.
        # OpenMP support has to be enabled during preprocessing.
        #
        # it may be possible that someone builds with a different/updated
        # compiler (don't know how to check for that).
        #
        # set the following environment variables, assumes that llvm openmp
        # has been built and installed by the user.
        #
        # export CFLAGS="-Xpreprocessor -fopenmp $CFLAGS"
        # export CPPFLAGS="-Xpreprocessor -fopenmp $CPPFLAGS"
        # export CFLAGS="$CFLAGS -I/usr/local/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/include"
        # export LDFLAGS="$LDFLAGS -lomp"
        return []
    # Default flag for GCC and clang:
    return ['-fopenmp']


def check_openmp_support():
    """Check whether OpenMP test code can be compiled and run"""
    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    start_dir = os.path.abspath('.')

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)

            # Write test program
            with open('test_openmp.c', 'w') as f:
                f.write(CCODE)

            os.mkdir('objects')

            # Compile, test program
            openmp_flags = get_openmp_flag(ccompiler)
            ccompiler.compile(['test_openmp.c'], output_dir='objects',
                              extra_postargs=openmp_flags)

            # Link test program
            extra_preargs = os.getenv('LDFLAGS', None)
            if extra_preargs is not None:
                extra_preargs = extra_preargs.split(" ")
            else:
                extra_preargs = []

            objects = glob.glob(
                os.path.join('objects', '*' + ccompiler.obj_extension))
            ccompiler.link_executable(objects, 'test_openmp',
                                      extra_preargs=extra_preargs,
                                      extra_postargs=openmp_flags)

            # Run test program
            output = subprocess.check_output('./test_openmp')
            output = output.decode(sys.stdout.encoding or 'utf-8').splitlines()

            # Check test program output
            if 'nthreads=' in output[0]:
                nthreads = int(output[0].strip().split('=')[1])
                openmp_supported = (len(output) == nthreads)
            else:
                openmp_supported = False

        except (CompileError, LinkError, subprocess.CalledProcessError):
            openmp_supported = False

        finally:
            os.chdir(start_dir)

    return openmp_supported
