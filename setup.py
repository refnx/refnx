#! /usr/bin/env python
# System imports
from setuptools import setup, Extension, find_packages
import os
import subprocess


try:
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

packages = find_packages()


# versioning
MAJOR = 0
MINOR = 0
MICRO = 3
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)


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


# refnx setup
info = {
        'name': 'refnx',
        'description': 'Neutron and X-ray Reflectometry Analysis',
        'author': 'Andrew Nelson',
        'author_email': 'andrew.nelson@ansto.gov.au',
        'license': 'BSD',
        'url': 'https://github.com/refnx/refnx',
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
        'include_package_data': True,
        'setup_requires': ['numpy', 'scipy', 'lmfit', 'uncertainties'],
        'install_requires': ['numpy', 'scipy', 'lmfit', 'uncertainties']
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
        # Obtain the numpy include directory.  This logic works across numpy versions.
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
                                  name='refnx.analysis._creflect',
                                  sources=['src/_creflect.pyx', 'src/refcalc.cpp'],
                                  include_dirs=[numpy_include],
                                  language='c',
                                  extra_link_args=['-lpthread']
                                  # libraries=
                                  # extra_compile_args = "...".split(),
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


if __name__ == '__main__':
    setup_package()
