#!/usr/bin/env python
"""
runtests.py [OPTIONS] [-- ARGS]

Run tests, building the project first.

Examples::

    $ python runtests.py
    $ python runtests.py -s {SAMPLE_SUBMODULE}
    $ python runtests.py -t {SAMPLE_TEST}
"""

#
# This is a generic test runner script for projects using Numpy's test
# framework. Change the following values to adapt to your project:
#

PROJECT_MODULE = "refnx"
PROJECT_ROOT_FILES = ['refnx', 'LICENSE', 'setup.py']
SAMPLE_TEST = "refnx.analysis.test.:test_objective::TestObjective"
SAMPLE_SUBMODULE = "analysis"

EXTRA_PATH = []

# ---------------------------------------------------------------------


if __doc__ is None:
    __doc__ = "Run without -OO if you want usage info"
else:
    __doc__ = __doc__.format(**globals())


import sys
import os

# In case we are run from the source directory, we don't want to import the
# project from there:
sys.path.pop(0)

import shutil
import subprocess
import time
import datetime
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
                        help="Submodule whose tests to run (cluster, constants, ...)")
    parser.add_argument("--pythonpath", "-p", default=None,
                        help="Paths to prepend to PYTHONPATH")
    parser.add_argument("--tests", "-t", action='append',
                        help="Specify tests to run")
    parser.add_argument("--python", action="store_true",
                        help="Start a Python shell with PYTHONPATH set")
    parser.add_argument("--ipython", "-i", action="store_true",
                        help="Start IPython shell with PYTHONPATH set")
    parser.add_argument("--shell", action="store_true",
                        help="Start Unix shell with PYTHONPATH set")
    parser.add_argument("--parallel", "-j", type=int, default=1,
                        help="Number of parallel jobs during build (requires "
                             "Numpy 1.10 or greater).")
    parser.add_argument("--show-build-log", action="store_true",
                        help="Show build output rather than using a log file")
    parser.add_argument("--bench", action="store_true",
                        help="Run benchmark suite instead of test suite")
    parser.add_argument("--bench-compare", action="append", metavar="BEFORE",
                        help=("Compare benchmark results of current HEAD to BEFORE. "
                              "Use an additional --bench-compare=COMMIT to override HEAD with COMMIT. "
                              "Note that you need to commit your changes first!"
                             ))
    parser.add_argument("args", metavar="ARGS", default=[], nargs=REMAINDER,
                        help="Arguments to pass to Nose, Python or shell")
    args = parser.parse_args(argv)

    if args.bench_compare:
        args.bench = True
        args.no_build = True # ASV does the building

    if args.pythonpath:
        for p in reversed(args.pythonpath.split(os.pathsep)):
            sys.path.insert(0, p)

    if not args.no_build:
        site_dir = build_project(args)
        sys.path.insert(0, site_dir)
        os.environ['PYTHONPATH'] = site_dir

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

    if args.ipython:
        import IPython
        IPython.embed(user_ns={})
        sys.exit(0)

    if args.shell:
        shell = os.environ.get('SHELL', 'sh')
        print("Spawning a Unix shell...")
        os.execv(shell, [shell] + extra_argv)
        sys.exit(1)

    if args.bench:
        # Run ASV
        items = extra_argv
        if args.tests:
            items += args.tests
        if args.submodule:
            items += [args.submodule]

        bench_args = []
        for a in items:
            bench_args.extend(['--bench', a])

        if not args.bench_compare:
            cmd = [os.path.join(ROOT_DIR, 'benchmarks', 'run.py'),
                   'run', '-n', '-e', '--python=same'] + bench_args
            os.execv(sys.executable, [sys.executable] + cmd)
            sys.exit(1)
        else:
            if len(args.bench_compare) == 1:
                commit_a = args.bench_compare[0]
                commit_b = 'HEAD'
            elif len(args.bench_compare) == 2:
                commit_a, commit_b = args.bench_compare
            else:
                p.error("Too many commits to compare benchmarks for")

            # Check for uncommitted files
            if commit_b == 'HEAD':
                r1 = subprocess.call(['git', 'diff-index', '--quiet', '--cached', 'HEAD'])
                r2 = subprocess.call(['git', 'diff-files', '--quiet'])
                if r1 != 0 or r2 != 0:
                    print("*"*80)
                    print("WARNING: you have uncommitted changes --- these will NOT be benchmarked!")
                    print("*"*80)

            # Fix commit ids (HEAD is local to current repo)
            p = subprocess.Popen(['git', 'rev-parse', commit_b], stdout=subprocess.PIPE)
            out, err = p.communicate()
            commit_b = out.strip()

            p = subprocess.Popen(['git', 'rev-parse', commit_a], stdout=subprocess.PIPE)
            out, err = p.communicate()
            commit_a = out.strip()

            cmd = [os.path.join(ROOT_DIR, 'benchmarks', 'run.py'),
                   'continuous', '-e', '-f', '1.05',
                   commit_a, commit_b] + bench_args
            os.execv(sys.executable, [sys.executable] + cmd)
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

    # Run the tests

    if not args.no_build:
        test_dir = site_dir
    else:
        test_dir = os.path.join(ROOT_DIR, 'build', 'test')
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

    cwd = os.getcwd()
    try:
        os.chdir(test_dir)
        result = test(args.mode,
                      verbose=args.verbose,
                      extra_argv=extra_argv,
                      doctests=args.doctests,
                      tests=tests)
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
    site_dir
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

    # Always use ccache, if installed
    env['PATH'] = os.pathsep.join(EXTRA_PATH + env.get('PATH', '').split(os.pathsep))

    cmd += ['build']
    if args.parallel > 1:
        cmd += ['-j', str(args.parallel)]
    # Install; avoid producing eggs so refnx can be imported from dst_dir.
    cmd += ['install', '--prefix=' + dst_dir,
            '--single-version-externally-managed',
            '--record=' + dst_dir + 'tmp_install_log.txt']

    from distutils.sysconfig import get_python_lib
    site_dir = get_python_lib(prefix=dst_dir, plat_specific=True)
    # easy_install won't install to a path that Python by default cannot see
    # and isn't on the PYTHONPATH.  Plus, it has to exist.
    if not os.path.exists(site_dir):
        os.makedirs(site_dir)
    env['PYTHONPATH'] = site_dir

    log_filename = os.path.join(ROOT_DIR, 'build.log')
    start_time = datetime.datetime.now()

    if args.show_build_log:
        ret = subprocess.call(cmd, env=env, cwd=ROOT_DIR)
    else:
        log_filename = os.path.join(ROOT_DIR, 'build.log')
        print("Building, see build.log...")
        with open(log_filename, 'w') as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log,
                                 cwd=ROOT_DIR)

        try:
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
                        elapsed = datetime.datetime.now() - start_time
                        print("    ... build in progress ({0} elapsed)".format(elapsed))
                        last_blip = time.time()
                        last_log_size = log_size

            ret = p.wait()
        except:
            p.terminate()
            raise

    elapsed = datetime.datetime.now() - start_time

    if ret == 0:
        print("Build OK ({0} elapsed)".format(elapsed))
    else:
        if not args.show_build_log:
            with open(log_filename, 'r') as f:
                print(f.read())
            print("Build failed! ({0} elapsed)".format(elapsed))
        sys.exit(1)

    return site_dir


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
