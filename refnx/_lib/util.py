from __future__ import print_function

import warnings as _warnings
import os as _os
import sys as _sys
import functools
from tempfile import mkdtemp
import collections
from contextlib import contextmanager
try:
    from inspect import getfullargspec as _getargspecf
except ImportError:
    # on 2.7
    from inspect import getargspec as _getargspecf


from refnx._lib.emcee.interruptible_pool import InterruptiblePool


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


class TemporaryDirectory(object):
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
                print("ERROR: {!r} while cleaning up {!r}".format(ex, self,),
                      file=_sys.stderr)
                return
            self._closed = True
            if _warn:
                # ResourceWarning
                self._warn("ResourceWarning: Implicitly cleaning"
                           " up {!r}".format(self))

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


def flatten(l):
    """
    Flatten a nested sequence.

    Parameters
    ----------
    l : sequence
        The sequence to flatten

    Returns
    -------
    el : generator
        yields flattened sequences from l
    """
    for el in l:
        if (isinstance(el, collections.Iterable) and
                not isinstance(el, (str, bytes))):
            # 2.7 has no yield from
            # yield from flatten(el)
            for elel in flatten(el):
                yield elel
        else:
            yield el


def unique(seq, idfun=id):
    """
    List of unique values in sequence (by object id). Ordering is preserved

    Parameters
    ----------
    seq : sequence
    idfun : callable

    Returns
    -------
    p : generator
        yields unique values from l
    """
    seen = {}
    for item in seq:
        marker = idfun(item)
        if marker not in seen:
            seen[marker] = 1
            yield item


@contextmanager
def possibly_open_file(f, mode='wb'):
    """
    Context manager for files.

    Parameters
    ----------
    f : file-like or str
        If `f` is a file, then yield the file. If `f` is a str then open the
        file and yield the newly opened file.
        On leaving this context manager the file is closed, if it was opened
        by this context manager (i.e. `f` was a string).
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
    if (hasattr(f, 'read') and hasattr(f, 'write')) or f is None:
        g = f
    else:
        g = open(f, mode)
        close_file = True
    yield g
    if close_file:
        g.close()


class possibly_create_pool(object):
    """
     Context manager for multiprocessing.

     Parameters
     ----------
     pool : int or map-like object
         If `pool` is an `int` then it specifies the number of threads to
         use for parallelization. If `pool == 1`, then no parallel
         processing is used.
         If pool is an object with a map method that follows the same
         calling sequence as the built-in map function, then this pool is
         used for parallelisation.

     Yields
     ------
     g : pool-like object
         On leaving the context manager the pool is closed, if it was opened by
         this context manager.
     """
    def __init__(self, pool):
        """
        Context manager for multiprocessing.

        Parameters
        ----------
        pool : int or map-like object
            If `pool` is an `int` then it specifies the number of threads to
            use for parallelization. If `pool == 1`, then no parallel
            processing is used.
            If pool is an object with a map method that follows the same
            calling sequence as the built-in map function, then this pool is
            used for parallelisation.

        Yields
        ------
        g : pool-like object
            On leaving the context manager the pool is closed, if it was opened
            by this context manager.
        """
        self.pool = pool
        self._created_pool = None

    def __enter__(self):
        if hasattr(self.pool, 'map'):
            return self.pool
        else:
            # user supplies a number
            if self.pool == 0:
                # use as many processors as possible
                g = InterruptiblePool()
            elif self.pool == 1:
                return map
            elif self.pool > 1:
                # only use the number of processors requested
                g = InterruptiblePool(processes=int(self.pool))
            else:
                raise ValueError("you need to supply an integer for creating a"
                                 " pool")

            self._created_pool = g
            return g

    def __exit__(self, *args):
        if self._created_pool is not None:
            self._created_pool.close()


def getargspec(f):
    return _getargspecf(f)
