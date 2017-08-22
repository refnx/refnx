#!/usr/bin/env python
"""
runtests.py [OPTIONS] [-- ARGS]

Run tests, building the project first.

Examples::

    $ python runtests.py
    $ python runtests.py -s {SAMPLE_SUBMODULE}

"""

#
# This is a generic test runner script for projects using Numpy's test
# framework. Change the following values to adapt to your project:
#

PROJECT_MODULE = "refnx"
PROJECT_ROOT_FILES = ['refnx', 'LICENSE', 'setup.py']
SAMPLE_SUBMODULE = "analysis"
EXTRA_PATH = ['']

# ---------------------------------------------------------------------


if __doc__ is None:
    __doc__ = "Run without -OO if you want usage info"
else:
    __doc__ = __doc__.format(**globals())


import sys
import os
import site

# In case we are run from the source directory, we don't want to import the
# project from there:
sys.path.pop(0)

import shutil
import subprocess
import time
import imp
from argparse import ArgumentParser, REMAINDER


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

def main(argv):
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("--verbose", "-v", action="count", default=1,
                        help="more verbosity")
    parser.add_argument("--no-build", "-n", action="store_true", default=False,
                        help="do not build the project (use system installed version)")
    parser.add_argument("--build-only", "-b", action="store_true", default=False,
                        help="just build, do not run any tests")
    parser.add_argument("--doctests", action="store_true", default=False,
                        help="Run doctests in module")
    parser.add_argument("--mode", "-m", default="fast",
                        help="'fast', 'full', or something that could be "
                             "passed to nosetests -A [default: fast]")
    parser.add_argument("--submodule", "-s", default=None,
                        help="Submodule whose tests to run (analysis, reduce, ...)")
    parser.add_argument("--pythonpath", "-p", default=None,
                        help="Paths to prepend to PYTHONPATH")
    parser.add_argument("--tests", "-t", action='append',
                        help="Specify tests to run")
    parser.add_argument("--python", action="store_true",
                        help="Start a Python shell with PYTHONPATH set")
    parser.add_argument("--shell", action="store_true",
                        help="Start Unix shell with PYTHONPATH set")
    parser.add_argument("--debug", "-g", action="store_true",
                        help="Debug build")
    parser.add_argument("--show-build-log", action="store_true",
                        help="Show build output rather than using a log file")
    parser.add_argument("args", metavar="ARGS", default=[], nargs=REMAINDER,
                        help="Arguments to pass to Nose, Python or shell")
    args = parser.parse_args(argv)

    test_dir = os.path.join(ROOT_DIR, 'build', 'test')
    cwd = os.getcwd()

    if args.pythonpath:
        for p in reversed(args.pythonpath.split(os.pathsep)):
            sys.path.insert(0, p)

    if not args.no_build:
        cwd = os.getcwd()
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)
        os.chdir(test_dir)

        dst_dir, site_dir = build_project(args)
        sys.path.insert(0, site_dir)
        os.environ['PYTHONPATH'] = site_dir
        site.addsitedir(site_dir)

    extra_argv = args.args[:]
    if extra_argv and extra_argv[0] == '--':
        extra_argv = extra_argv[1:]

    if args.python:
        if extra_argv:
            # Don't use subprocess, since we don't want to include the
            # current path in PYTHONPATH.
            sys.argv = extra_argv
            with open(extra_argv[0], 'r') as f:
                script = f.read()
            sys.modules['__main__'] = imp.new_module('__main__')
            ns = dict(__name__='__main__',
                      __file__=extra_argv[0])
            exec_(script, ns)
            sys.exit(0)
        else:
            import code
            code.interact()
            sys.exit(0)

    if args.shell:
        shell = os.environ.get('SHELL', 'sh')
        print("Spawning a Unix shell...")
        os.execv(shell, [shell] + extra_argv)
        sys.exit(1)

    if args.build_only:
        sys.exit(0)
    else:
        __import__(PROJECT_MODULE)
        test = sys.modules[PROJECT_MODULE].test

    if args.submodule:
        tests = [PROJECT_MODULE + "." + args.submodule]
    elif args.tests:
        tests = args.tests
    else:
        tests = None

    # Run the tests under build/test
    try:
        shutil.rmtree(test_dir)
    except OSError:
        pass
    try:
        os.makedirs(test_dir)
    except OSError:
        pass

    try:
        os.chdir(test_dir)
        result = test(args.mode,
                      verbose=args.verbose,
                      extra_argv=extra_argv)
    finally:
        os.chdir(cwd)

    if isinstance(result, bool):
        sys.exit(0 if result else 1)
    elif result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)


def build_project(args):
    """
    Build a dev version of the project.

    Returns
    -------
    dst_dir, site_dir: tuple
        directory for the test environment
        site-packages directory where it was installed

    """

    root_ok = [os.path.exists(os.path.join(ROOT_DIR, fn))
               for fn in PROJECT_ROOT_FILES]
    if not all(root_ok):
        print("To build the project, run runtests.py in "
              "git checkout or unpacked source")
        sys.exit(1)

    dst_dir = os.path.join(ROOT_DIR, 'build', 'testenv')

    env = dict(os.environ)
    cmd = [sys.executable, 'setup.py']

    env['PATH'] = os.pathsep.join(env.get('PATH', '').split(os.pathsep))

    cmd += ['build']
    cmd += ['install', '--prefix=' + dst_dir]
    cmd += ['--single-version-externally-managed', '--record=t']

    from distutils.sysconfig import get_python_lib
    site_dir = get_python_lib(prefix=dst_dir, plat_specific=True)

    # easy_install won't install to a path that Python by default cannot see
    # and isn't on the PYTHONPATH.  Plus, it has to exist.
    if not os.path.exists(site_dir):
        os.makedirs(site_dir)
    env['PYTHONPATH'] = site_dir

    log_filename = os.path.join(ROOT_DIR, 'build.log')

    if args.show_build_log:
        ret = subprocess.call(cmd, env=env, cwd=ROOT_DIR)
    else:
        log_filename = os.path.join(ROOT_DIR, 'build.log')
        print("Building, see build.log...")
        with open(log_filename, 'w') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                 cwd=ROOT_DIR)

        # Wait for it to finish, and print something to indicate the
        # process is alive, but only if the log file has grown (to
        # allow continuous integration environments kill a hanging
        # process accurately if it produces no output)
        last_blip = time.time()
        last_log_size = os.stat(log_filename).st_size
        while p.poll() is None:
            time.sleep(0.5)
            if time.time() - last_blip > 60:
                log_size = os.stat(log_filename).st_size
                if log_size > last_log_size:
                    print("    ... build in progress")
                    last_blip = time.time()
                    last_log_size = log_size

        ret = p.wait()

    if ret == 0:
        print("Build OK")
    else:
        if not args.show_build_log:
            with open(log_filename, 'r') as f:
                print(f.read())
            print("Build failed!")
        sys.exit(1)

    return dst_dir, site_dir


#
# Python 3 support
#

if sys.version_info[0] >= 3:
    import builtins
    exec_ = getattr(builtins, "exec")
else:
    def exec_(code, globs=None, locs=None):
        """Execute code in a namespace."""
        if globs is None:
            frame = sys._getframe(1)
            globs = frame.f_globals
            if locs is None:
                locs = frame.f_locals
            del frame
        elif locs is None:
            locs = globs
        exec("""exec code in globs, locs""")

if __name__ == "__main__":
    main(argv=sys.argv[1:])
