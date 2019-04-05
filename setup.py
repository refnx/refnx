#! /usr/bin/env python
# System imports
from setuptools import setup, Extension, find_packages
from setuptools.command.test import test as TestCommand
import os
import subprocess
import sys
import warnings

try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
    warnings.warn(
        "Cython was not found. Slow reflectivity calculations will be used.")
else:
    USE_CYTHON = True

packages = find_packages()
try:
    idx = packages.index('motofit')
    if idx >= 0:
        packages.pop(idx)
except ValueError:
    pass

# versioning
MAJOR = 0
MINOR = 1
MICRO = 5
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


# are we on windows, darwin, etc?
platform = sys.platform


# Return the git revision as a string
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"

    return GIT_REVISION


def get_version_info():
    # Adding the git rev number needs to be done inside
    # write_version_py(), otherwise the import of refnx.version messes
    # up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    elif os.path.exists('refnx/version.py'):
        # must be a source distribution, use existing version file
        # load it as a separate module to not load refnx/__init__.py
        import imp
        version = imp.load_source('refnx.version', 'refnx/version.py')
        GIT_REVISION = version.git_revision
    else:
        GIT_REVISION = "Unknown"

    if not ISRELEASED:
        FULLVERSION += '.dev0+' + GIT_REVISION[:7]

    return FULLVERSION, GIT_REVISION


def write_version_py(filename='refnx/version.py'):
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

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = 'refnx'

    def run_tests(self):
        import shlex
        import pytest
        print("Running tests with pytest")
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


# refnx setup
info = {
        'name': 'refnx',
        'description': 'Neutron and X-ray Reflectometry Analysis',
        'author': 'Andrew Nelson',
        'author_email': 'andyfaff+refnx@gmail.com',
        'license': 'BSD',
        'url': 'https://github.com/refnx/refnx',
        'platforms': ["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
        'classifiers': [
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Operating System :: OS Independent',
        # 'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        ],
        'packages': packages,
        'include_package_data': True,
        'setup_requires': ['numpy'],
        'python_requires': '>=3.5',
        'install_requires': ['numpy', 'scipy'],
        'extras_require': {'all': ['IPython', 'ipywidgets', 'traitlets',
                                   'matplotlib', 'xlrd', 'h5py', 'tqdm',
                                   'pymc3', 'theano', 'ptemcee', 'pandas',
                                   'pyparsing', 'periodictable', 'pyqt']},
        'tests_require': ['pytest', 'uncertainties'],
        'cmdclass': {'test': PyTest},
        'entry_points': {"gui_scripts" : ['refnx = refnx.reflect:main',
                                          'slim = refnx.reduce:main']}
        }

####################################################################
# this is where setup starts
####################################################################
def setup_package():

    # Rewrite the version file every time
    write_version_py()
    info['version'] = get_version_info()[0]
    print(info['version'])

    if USE_CYTHON:
        # Obtain the numpy include directory.  This logic works across numpy
        # versions.
        ext_modules = []
        HAS_NUMPY = True

        try:
            import numpy as np
        except:
            info['setup_requires'] = ['numpy']
            HAS_NUMPY = False

        if HAS_NUMPY:
            try:
                numpy_include = np.get_include()
            except AttributeError:
                numpy_include = np.get_numpy_include()

            # creflect extension module
            _creflect = Extension(
                                  name='refnx.reflect._creflect',
                                  sources=['src/_creflect.pyx',
                                           'src/refcalc.cpp'],
                                  include_dirs=[numpy_include],
                                  language='c++',
                                  extra_compile_args=[],
                                  extra_link_args=['-lpthread']
                                  # libraries=
                                  # extra_compile_args = "...".split(),
                                  )
            ext_modules.append(_creflect)

            _cevent = Extension(
                                name='refnx.reduce._cevent',
                                sources=['src/_cevent.pyx'],
                                include_dirs=[numpy_include],
                                language='c++',
                                # libraries=
                                # extra_compile_args = "...".split(),
                                )
            ext_modules.append(_cevent)

            _cutil = Extension(
                               name='refnx._lib._cutil',
                               sources=['src/_cutil.pyx'],
                               include_dirs=[numpy_include],
                               language='c++',
                               # libraries=
                               # extra_compile_args = "...".split(),
                               )
            ext_modules.append(_cutil)

            # specify min deployment version for macOS
            if platform == 'darwin':
                for mod in ext_modules:
                    mod.extra_compile_args.append('-mmacosx-version-min=10.9')

            info['cmdclass'].update({'build_ext': build_ext})
            info['ext_modules'] = ext_modules
            info['zip_safe'] = False

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


if __name__ == '__main__':
    setup_package()
