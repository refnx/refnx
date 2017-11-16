from __future__ import division, absolute_import
import collections


def c_flatten(seq):
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
        if (isinstance(el, collections.Iterable) and
                not isinstance(el, (str, bytes))):
            # 2.7 has no yield from
            # yield from c_flatten(el)
            for elel in c_flatten(el):
                yield elel
        else:
            yield el


def c_unique(seq, idfun=id):
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
