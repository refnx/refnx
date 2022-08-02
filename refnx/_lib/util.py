from multiprocessing import Pool, get_context
import warnings as _warnings
import os as _os
import sys as _sys
import functools
from tempfile import mkdtemp
from contextlib import contextmanager
from inspect import getfullargspec as _getargspecf


def preserve_cwd(function):
    """
    Ensures that the original working directory is kept when exiting a function

    Parameters
    ----------
    function : callable

    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        cwd = _os.getcwd()
        try:
            return function(*args, **kwargs)
        finally:
            _os.chdir(cwd)

    return decorator


class TemporaryDirectory:
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, suffix="", prefix="tmp", dir=None):
        self._closed = False
        # Handle mkdtemp raising an exception
        self.name = None
        self.name = mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        if self.name and not self._closed:
            try:
                self._rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                # Issue #10188: Emit a warning on stderr
                # if the directory could not be cleaned
                # up due to missing globals
                if "None" not in str(ex):
                    raise
                print(
                    "ERROR: {!r} while cleaning up {!r}".format(ex, self),
                    file=_sys.stderr,
                )
                return
            self._closed = True
            if _warn:
                # ResourceWarning
                self._warn(
                    "ResourceWarning: Implicitly cleaning"
                    " up {!r}".format(self)
                )

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

    # XXX (ncoghlan): The following code attempts to make
    # this class tolerant of the module nulling out process
    # that happens during CPython interpreter shutdown
    # Alas, it doesn't actually manage it. See issue #10188
    _listdir = staticmethod(_os.listdir)
    _path_join = staticmethod(_os.path.join)
    _isdir = staticmethod(_os.path.isdir)
    _islink = staticmethod(_os.path.islink)
    _remove = staticmethod(_os.remove)
    _rmdir = staticmethod(_os.rmdir)
    _warn = _warnings.warn

    def _rmtree(self, path):
        # Essentially a stripped down version of shutil.rmtree.  We can't
        # use globals because they may be None'ed out at shutdown.
        for name in self._listdir(path):
            fullname = self._path_join(path, name)
            try:
                isdir = self._isdir(fullname) and not self._islink(fullname)
            except OSError:
                isdir = False
            if isdir:
                self._rmtree(fullname)
            else:
                try:
                    self._remove(fullname)
                except OSError:
                    pass
        try:
            self._rmdir(path)
        except OSError:
            pass


def flatten(seq):
    """
    Flatten a nested sequence.

    Parameters
    ----------
    seq : sequence
        The sequence to flatten

    Returns
    -------
    el : generator
        yields flattened sequences from seq
    """
    for el in seq:
        try:
            iter(el)
            if isinstance(el, (str, bytes)):
                raise TypeError
            yield from flatten(el)
        except TypeError:
            yield el


def unique(seq, idfun=id):
    """
    List of unique values in sequence (by object id, not by value).
    Ordering is preserved.

    Parameters
    ----------
    seq : sequence
    idfun : callable

    Returns
    -------
    p : generator
        yields unique values from l

    Notes
    -----
    Because this function works on object id (by default), it won't work
    for looking at the number of unique values in a numpy array - all
    the entries have different object id's.
    """
    seen = {}
    for item in seq:
        marker = idfun(item)
        if marker not in seen:
            seen[marker] = 1
            yield item


@contextmanager
def possibly_open_file(f, mode="r"):
    """
    Context manager for files.

    Parameters
    ----------
    f : {file-like, Path, str}
        If `f` is a file, then yield the file. If `f` is a str or Path then
        open the file and yield the newly opened file.
        On leaving this context manager the file is closed, if it was opened
        by this context manager (i.e. `f` was a str or Path).
    mode : str, optional
        mode is an optional string that specifies the mode in which the file
        is opened.

    Yields
    ------
    g : file-like
        On leaving the context manager the file is closed, if it was opened by
        this context manager.
    """
    close_file = False
    if (hasattr(f, "read") and hasattr(f, "write")) or f is None:
        g = f
    else:
        g = open(f, mode)
        close_file = True
    yield g
    if close_file:
        g.close()


class MapWrapper:
    """
    Parallelisation wrapper for working with map-like callables, such as
    `multiprocessing.Pool.map`.

    Parameters
    ----------
    pool : int or map-like callable
        If `pool` is an integer, then it specifies the number of threads to
        use for parallelization. If ``int(pool) == 1``, then no parallel
        processing is used and the map builtin is used.
        If ``pool == -1``, then the pool will utilise all available CPUs.
        If `pool` is a map-like callable that follows the same
        calling sequence as the built-in map function, then this callable is
        used for parallelisation.
    context : None, {'spawn', 'fork', 'forkserver'}

    """

    def __init__(self, pool=-1, context=None):
        self.pool = None
        self._mapfunc = map
        self._own_pool = False

        ctx = get_context(context)

        if callable(pool):
            self.pool = pool
            self._mapfunc = self.pool
        else:
            # user supplies a number
            if int(pool) == -1:
                # use as many processors as possible
                self.pool = ctx.Pool()
                self._mapfunc = self.pool.map
                self._own_pool = True
            elif int(pool) in [0, 1]:
                pass
            elif int(pool) > 1:
                # use the number of processors requested
                self.pool = ctx.Pool(processes=int(pool))
                self._mapfunc = self.pool.map
                self._own_pool = True

    def __enter__(self):
        return self

    def __del__(self):
        self.close()
        self.terminate()

    def terminate(self):
        if self._own_pool:
            self.pool.terminate()

    def join(self):
        if self._own_pool:
            self.pool.join()

    def close(self):
        if self._own_pool:
            self.pool.close()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._own_pool:
            self.pool.close()
            self.pool.terminate()

    def __call__(self, func, iterable):
        # only accept one iterable because that's all Pool.map accepts
        try:
            return self._mapfunc(func, iterable)
        except TypeError:
            # wrong number of arguments
            raise TypeError(
                "The map-like callable must be of the"
                " form f(func, iterable)"
            )

    def map(self, func, iterable):
        return self(func, iterable)


def getargspec(f):
    return _getargspecf(f)
