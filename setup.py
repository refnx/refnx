#! /usr/bin/env python
# System imports
from setuptools import setup, Extension, find_packages
from setuptools.command.test import test as TestCommand
import os
import subprocess
import platform
import sys
import warnings
import glob
import tempfile
import textwrap
import subprocess

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
    warnings.warn("Cython was not found. Slow reflectivity calculations will be used.")
else:
    USE_CYTHON = True


###############################################################################
"""
Is openMP usable?
"""
CCODE = textwrap.dedent(
    """\
    #include <omp.h>
    #include <stdio.h>
    int main(void) {
    #pragma omp parallel
    printf("nthreads=%d\\n", omp_get_num_threads());
    return 0;
    }
    """
)


def get_openmp_flag(compiler):
    if hasattr(compiler, "compiler"):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == "win32" and ("icc" in compiler or "icl" in compiler):
        return ["/Qopenmp"]
    elif sys.platform == "win32":
        return ["/openmp"]
    elif sys.platform == "darwin" and ("icc" in compiler or "icl" in compiler):
        return ["-openmp"]
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
        # brew install libomp
        # export CC=clang
        # export CXX =clang++
        # export CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
        # export DYLD_LIBRARY_PATH =/usr/local/opt/libomp/lib
        return []
    # Default flag for GCC and clang:
    return ["-fopenmp"]


def check_openmp_support():
    """Check whether OpenMP test code can be compiled and run"""

    try:
        from setuptools._distutils.ccompiler import new_compiler
        from setuptools._distutils.sysconfig import customize_compiler
        # from numpy.distutils.ccompiler import new_compiler
        # from distutils.sysconfig import customize_compiler
        from distutils.errors import CompileError, LinkError
    except ImportError:
        return False

    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    start_dir = os.path.abspath(".")

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)

            # Write test program
            with open("test_openmp.c", "w") as f:
                f.write(CCODE)

            os.mkdir("objects")

            # Compile, test program
            openmp_flags = get_openmp_flag(ccompiler)
            ccompiler.compile(
                ["test_openmp.c"], output_dir="objects", extra_postargs=openmp_flags
            )

            # Link test program
            extra_preargs = os.getenv("LDFLAGS", None)
            if extra_preargs is not None:
                extra_preargs = extra_preargs.split(" ")
            else:
                extra_preargs = []

            objects = glob.glob(os.path.join("objects", "*" + ccompiler.obj_extension))
            ccompiler.link_executable(
                objects,
                "test_openmp",
                extra_preargs=extra_preargs,
                extra_postargs=openmp_flags,
            )

            # Run test program
            output = subprocess.check_output("./test_openmp")
            output = output.decode(sys.stdout.encoding or "utf-8").splitlines()

            # Check test program output
            if "nthreads=" in output[0]:
                nthreads = int(output[0].strip().split("=")[1])
                openmp_supported = len(output) == nthreads
            else:
                openmp_supported = False

        except (CompileError, LinkError, subprocess.CalledProcessError):
            openmp_supported = False

        finally:
            os.chdir(start_dir)

    return openmp_supported


# do you want to parallelise things with openmp?
HAS_OPENMP = check_openmp_support()
# HAS_OPENMP = False

###############################################################################


# versioning
MAJOR = 0
MINOR = 1
MICRO = 29
ISRELEASED = True
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"


# are we on windows, darwin, etc?
platform = sys.platform
packages = find_packages()
try:
    idx = packages.index("benchmarks")
    if idx >= 0:
        packages.pop(idx)
    idx = packages.index("benchmarks.benchmarks")
    if idx >= 0:
        packages.pop(idx)
    idx = packages.index("motofit")
    if idx >= 0:
        packages.pop(idx)
except ValueError:
    pass


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ["SYSTEMROOT", "PATH"]:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env["LANGUAGE"] = "C"
        env["LANG"] = "C"
        env["LC_ALL"] = "C"
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        GIT_REVISION = out.strip().decode("ascii")
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of refnx.version messes
    # up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists(".git"):
        GIT_REVISION = git_version()
    elif os.path.exists("refnx/version.py"):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load refnx/__init__.py
        import imp

        version = imp.load_source("refnx.version", "refnx/version.py")
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += ".dev0+" + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename="refnx/version.py"):
    cnt = """
# THIS FILE IS GENERATED FROM REFNX SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
"""
    FULLVERSION, GIT_REVISION = get_version_info()

    a = open(filename, "w")
    try:
        a.write(
            cnt
            % {
                "version": VERSION,
                "full_version": FULLVERSION,
                "git_revision": GIT_REVISION,
                "isrelease": str(ISRELEASED),
            }
        )
    finally:
        a.close()


class PyTest(TestCommand):
    user_options = [("pytest-args=", "a", "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = "refnx"

    def run_tests(self):
        import shlex
        import pytest

        print("Running tests with pytest")
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


# refnx setup
info = {
    "packages": packages,
    "include_package_data": True,
    "cmdclass": {"test": PyTest},
}

####################################################################
# this is where setup starts
####################################################################
def setup_package():

    # Rewrite the version file every time
    write_version_py()
    info["version"] = get_version_info()[0]
    print(info["version"])

    if USE_CYTHON:
        # Obtain the numpy include directory.  This logic works across numpy
        # versions.
        ext_modules = []
        HAS_NUMPY = True

        try:
            import numpy as np
        except:
            info["setup_requires"] = ["numpy"]
            HAS_NUMPY = False

        if HAS_NUMPY:
            try:
                numpy_include = np.get_include()
            except AttributeError:
                numpy_include = np.get_numpy_include()

            _cevent = Extension(
                name="refnx.reduce._cevent",
                sources=["src/_cevent.pyx"],
                include_dirs=[numpy_include],
                language="c++",
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                # libraries=
                # extra_compile_args = "...".split(),
            )
            ext_modules.append(_cevent)

            _cutil = Extension(
                name="refnx._lib._cutil",
                sources=["src/_cutil.pyx"],
                include_dirs=[numpy_include],
                language="c",
                # libraries=
                # extra_compile_args = "...".split(),
            )
            ext_modules.append(_cutil)

            # creflect extension module
            # Compile reflectivity calculator to object with C compiler
            # first.
            # It's not possible to do this in an Extension object because
            # the `-std=c++11` compile arg and C99 C code are incompatible
            # (at least on Darwin).
            from setuptools._distutils.ccompiler import new_compiler
            from setuptools._distutils.sysconfig import customize_compiler
            # from numpy.distutils.ccompiler import new_compiler
            # from distutils.sysconfig import customize_compiler

            ccompiler = new_compiler()
            customize_compiler(ccompiler)
            ccompiler.verbose = True
            extra_preargs = [
                "-O2",
            ]

            if sys.platform == "win32":
                # use the C++ code on Windows. The C++ code uses the
                # std::complex<double> object for its arithmetic.
                f = ["src/refcalc.cpp"]
            else:
                # and C code on other machines. The C code uses C99 complex
                # arithmetic which is 10-20% faster.
                # the CMPLX macro was only standardised in C11
                extra_preargs.extend(
                    [
                        "-std=c11",
                        "-funsafe-math-optimizations",
                        "-ffinite-math-only",
                    ]
                )
                f = ["src/refcalc.c"]
            refcalc_obj = ccompiler.compile(f, extra_preargs=extra_preargs)
            # print(refcalc_obj)

            _creflect = Extension(
                name="refnx.reflect._creflect",
                sources=["src/_creflect.pyx", "src/refcaller.cpp"],
                include_dirs=[numpy_include],
                language="c++",
                extra_compile_args=["-std=c++11"],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                extra_objects=refcalc_obj,
            )
            ext_modules.append(_creflect)

            # if we have openmp use pure cython version
            # openmp should be present on windows, linux
            #
            # However, it's not present in Apple Clang. Therefore one has to
            # jump through hoops to enable it.
            # It's probably easier to install OpenMP on macOS via homebrew.
            # However, it's fairly simple to build the OpenMP library, and
            # installing it into PREFIX=/usr/local
            #
            # https://gist.github.com/andyfaff/084005bee32aee83d6b59e843278ab3e
            #
            # Instructions for macOS:
            #
            # brew install libomp
            # export CC=clang
            # export CXX=clang++
            # export CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp"
            # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
            # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
            # export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
            # export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib

            if HAS_OPENMP:
                # cyreflect extension module
                _cyreflect = Extension(
                    name="refnx.reflect._cyreflect",
                    sources=["src/_cyreflect.pyx"],
                    include_dirs=[numpy_include],
                    language="c++",
                    extra_compile_args=[],
                    extra_link_args=[]
                    # libraries=
                    # extra_compile_args = "...".split(),
                )
                openmp_flags = get_openmp_flag(ccompiler)
                _cyreflect.extra_compile_args += openmp_flags
                _cyreflect.extra_link_args += openmp_flags

                ext_modules.append(_cyreflect)

            # specify min deployment version for macOS
            if platform == "darwin":
                for mod in ext_modules:
                    mod.extra_compile_args.append("-mmacosx-version-min=10.9")

            info["ext_modules"] = cythonize(ext_modules)
            info["zip_safe"] = False

    try:
        setup(**info)
    except ValueError:
        # there probably wasn't a C-compiler (windows). Try removing extension
        # compilation
        print("")
        print("*****WARNING*****")
        print(
            "You didn't try to build the Reflectivity calculation extension."
            " Calculation will be slow, falling back to pure python."
            " To compile extension install cython. If installing in windows you"
            " should then install from Visual Studio command prompt (this makes"
            " C compiler available"
        )
        print("*****************")
        print("")
        info.pop("cmdclass")
        info.pop("ext_modules")
        setup(**info)


if __name__ == "__main__":
    setup_package()
