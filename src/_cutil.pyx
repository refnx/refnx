# cython: language_level=3


def c_flatten(seq):
    """
    Flatten a nested sequence. Ordering is preserved

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
            yield from c_flatten(el)
        except TypeError:
            yield el


def c_unique(seq, idfun=id):
    """
    List of unique values in sequence (by object id).

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
