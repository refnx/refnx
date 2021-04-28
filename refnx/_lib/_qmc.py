"""Quasi-Monte Carlo engines and helpers.
vendored from scipy, will be in scipy 1.17
"""
import copy
import numbers
from abc import ABC, abstractmethod
import math
import warnings

import numpy as np

import scipy.stats as stats


__all__ = [
    "scale",
    "discrepancy",
    "update_discrepancy",
    "QMCEngine",
    "Halton",
    "LatinHypercube",
]


# Based on scipy._lib._util.check_random_state
# Based on scipy._lib._util.check_random_state
def check_random_state(seed=None):
    """Turn `seed` into a `numpy.random.Generator` instance.
    Parameters
    ----------
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    Returns
    -------
    seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
        Random number generator.
    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        if not hasattr(np.random, "Generator"):
            # This can be removed once numpy 1.16 is dropped
            msg = (
                "NumPy 1.16 doesn't have Generator, use either "
                "NumPy >= 1.17 or `seed=np.random.RandomState(seed)`"
            )
            raise ValueError(msg)
        return np.random.default_rng(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    elif isinstance(seed, np.random.Generator):
        # The two checks can be merged once numpy 1.16 is dropped
        return seed
    else:
        raise ValueError(
            "%r cannot be used to seed a numpy.random.Generator"
            " instance" % seed
        )


def scale(sample, l_bounds, u_bounds, reverse=False):
    r"""Sample scaling from unit hypercube to different bounds.
    To convert a sample from :math:`[0, 1)` to :math:`[a, b), b>a`,
    with :math:`a` the lower bounds and :math:`b` the upper bounds.
    The following transformation is used:
    .. math::
        (b - a) \cdot \text{sample} + a
    Parameters
    ----------
    sample : array_like (n, d)
        Sample to scale.
    l_bounds, u_bounds : array_like (d,)
        Lower and upper bounds (resp. :math:`a`, :math:`b`) of transformed
        data. If `reverse` is True, range of the original data to transform
        to the unit hypercube.
    reverse : bool, optional
        Reverse the transformation from different bounds to the unit hypercube.
        Default is False.
    Returns
    -------
    sample : array_like (n, d)
        Scaled sample.
    Examples
    --------
    Transform 3 samples in the unit hypercube to bounds:
    >>> from scipy.stats import qmc
    >>> l_bounds = [-2, 0]
    >>> u_bounds = [6, 5]
    >>> sample = [[0.5 , 0.75],
    ...           [0.5 , 0.5],
    ...           [0.75, 0.25]]
    >>> sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    >>> sample_scaled
    array([[2.  , 3.75],
           [2.  , 2.5 ],
           [4.  , 1.25]])
    And convert back to the unit hypercube:
    >>> sample_ = qmc.scale(sample_scaled, l_bounds, u_bounds, reverse=True)
    >>> sample_
    array([[0.5 , 0.75],
           [0.5 , 0.5 ],
           [0.75, 0.25]])
    """
    sample = np.asarray(sample)
    lower = np.atleast_1d(l_bounds)
    upper = np.atleast_1d(u_bounds)

    # Checking bounds and sample
    if not sample.ndim == 2:
        raise ValueError("Sample is not a 2D array")

    lower, upper = np.broadcast_arrays(lower, upper)

    if not np.all(lower < upper):
        raise ValueError("Bounds are not consistent a < b")

    if len(lower) != sample.shape[1]:
        raise ValueError("Sample dimension is different than bounds dimension")

    if not reverse:
        # Checking that sample is within the hypercube
        if not (np.all(sample >= 0) and np.all(sample <= 1)):
            raise ValueError("Sample is not in unit hypercube")

        return sample * (upper - lower) + lower
    else:
        # Checking that sample is within the bounds
        if not (np.all(sample >= lower) and np.all(sample <= upper)):
            raise ValueError("Sample is out of bounds")

        return (sample - lower) / (upper - lower)


def discrepancy(sample, iterative=False, method="CD"):
    """Discrepancy of a given sample.

    Parameters
    ----------
    sample : array_like (n, d)
        The sample to compute the discrepancy from.
    iterative : bool, optional
        Must be False if not using it for updating the discrepancy.
        Default is False. Refer to the notes for more details.
    method : str, optional
        Type of discrepancy, can be ``CD``, ``WD``, ``MD`` or ``L2-star``.
        Refer to the notes for more details. Default is ``CD``.

    Returns
    -------
    discrepancy : float
        Discrepancy.

    Notes
    -----
    The discrepancy is a uniformity criterion used to assess the space filling
    of a number of samples in a hypercube. A discrepancy quantifies the
    distance between the continuous uniform distribution on a hypercube and the
    discrete uniform distribution on :math:`n` distinct sample points.

    The lower the value is, the better the coverage of the parameter space is.

    For a collection of subsets of the hypercube, the discrepancy is the
    difference between the fraction of sample points in one of those
    subsets and the volume of that subset. There are different definitions of
    discrepancy corresponding to different collections of subsets. Some
    versions take a root mean square difference over subsets instead of
    a maximum.

    A measure of uniformity is reasonable if it satisfies the following
    criteria [1]_:

    1. It is invariant under permuting factors and/or runs.
    2. It is invariant under rotation of the coordinates.
    3. It can measure not only uniformity of the sample over the hypercube,
       but also the projection uniformity of the sample over non-empty
       subset of lower dimension hypercubes.
    4. There is some reasonable geometric meaning.
    5. It is easy to compute.
    6. It satisfies the Koksma-Hlawka-like inequality.
    7. It is consistent with other criteria in experimental design.

    Four methods are available:

    * ``CD``: Centered Discrepancy - subspace involves a corner of the
      hypercube
    * ``WD``: Wrap-around Discrepancy - subspace can wrap around bounds
    * ``MD``: Mixture Discrepancy - mix between CD/WD covering more criteria
    * ``L2-star``: L2-star discrepancy - like CD BUT variant to rotation

    See [2]_ for precise definitions of each method.

    Lastly, using ``iterative=True``, it is possible to compute the
    discrepancy as if we had :math:`n+1` samples. This is useful if we want
    to add a point to a sampling and check the candidate which would give the
    lowest discrepancy. Then you could just update the discrepancy with
    each candidate using `update_discrepancy`. This method is faster than
    computing the discrepancy for a large number of candidates.

    References
    ----------
    .. [1] Fang et al. "Design and modeling for computer experiments".
       Computer Science and Data Analysis Series, 2006.
    .. [2] Zhou Y.-D. et al. Mixture discrepancy for quasi-random point sets.
       Journal of Complexity, 29 (3-4) , pp. 283-301, 2013.
    .. [3] T. T. Warnock. "Computational investigations of low discrepancy
       point sets". Applications of Number Theory to Numerical
       Analysis, Academic Press, pp. 319-343, 1972.

    Examples
    --------
    Calculate the quality of the sample using the discrepancy:

    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> space
    array([[0.08333333, 0.41666667],
           [0.25      , 0.91666667],
           [0.41666667, 0.25      ],
           [0.58333333, 0.75      ],
           [0.75      , 0.08333333],
           [0.91666667, 0.58333333]])
    >>> qmc.discrepancy(space)
    0.008142039609053464

    We can also compute iteratively the ``CD`` discrepancy by using
    ``iterative=True``.

    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample)

    # Checking that sample is within the hypercube and 2D
    if not sample.ndim == 2:
        raise ValueError("Sample is not a 2D array")

    if not (np.all(sample >= 0) and np.all(sample <= 1)):
        raise ValueError("Sample is not in unit hypercube")

    n, d = sample.shape

    if iterative:
        n += 1

    if method == "CD":
        # reference [1], page 71 Eq (3.7)
        abs_ = abs(sample - 0.5)
        disc1 = np.sum(np.prod(1 + 0.5 * abs_ - 0.5 * abs_ ** 2, axis=1))

        prod_arr = 1
        for i in range(d):
            s0 = sample[:, i]
            prod_arr *= (
                1
                + 0.5 * abs(s0[:, None] - 0.5)
                + 0.5 * abs(s0 - 0.5)
                - 0.5 * abs(s0[:, None] - s0)
            )
        disc2 = prod_arr.sum()

        return (13.0 / 12.0) ** d - 2.0 / n * disc1 + 1.0 / (n ** 2) * disc2
    elif method == "WD":
        # reference [1], page 73 Eq (3.8)
        prod_arr = 1
        for i in range(d):
            s0 = sample[:, i]
            x_kikj = abs(s0[:, None] - s0)
            prod_arr *= 3.0 / 2.0 - x_kikj + x_kikj ** 2

        # typo in the book sign missing: - (4.0 / 3.0) ** d
        return -((4.0 / 3.0) ** d) + 1.0 / (n ** 2) * prod_arr.sum()
    elif method == "MD":
        # reference [2], page 290 Eq (18)
        abs_ = abs(sample - 0.5)
        disc1 = np.sum(
            np.prod(5.0 / 3.0 - 0.25 * abs_ - 0.25 * abs_ ** 2, axis=1)
        )

        prod_arr = 1
        for i in range(d):
            s0 = sample[:, i]
            prod_arr *= (
                15.0 / 8.0
                - 0.25 * abs(s0[:, None] - 0.5)
                - 0.25 * abs(s0 - 0.5)
                - 3.0 / 4.0 * abs(s0[:, None] - s0)
                + 0.5 * abs(s0[:, None] - s0) ** 2
            )
        disc2 = prod_arr.sum()

        disc = (19.0 / 12.0) ** d
        disc1 = 2.0 / n * disc1
        disc2 = 1.0 / (n ** 2) * disc2

        return disc - disc1 + disc2
    elif method == "L2-star":
        # reference [1], page 69 Eq (3.5)
        disc1 = np.sum(np.prod(1 - sample ** 2, axis=1))

        xik = sample[None, :, :]
        xjk = sample[:, None, :]
        disc2 = np.sum(
            np.sum(np.prod(1 - np.maximum(xik, xjk), axis=2), axis=1)
        )

        return np.sqrt(
            3 ** (-d) - 1 / n * 2 ** (1 - d) * disc1 + 1 / (n ** 2) * disc2
        )
    else:
        raise ValueError(
            "{} is not a valid method. Options are "
            "CD, WD, MD, L2-star.".format(method)
        )


def update_discrepancy(x_new, sample, initial_disc):
    """Update the centered discrepancy with a new sample.

    Parameters
    ----------
    x_new : array_like (1, d)
        The new sample to add in `sample`.
    sample : array_like (n, d)
        The initial sample.
    initial_disc : float
        Centered discrepancy of the `sample`.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the sample composed of `x_new` and `sample`.

    Examples
    --------
    We can also compute iteratively the discrepancy by using
    ``iterative=True``.

    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample)
    x_new = np.asarray(x_new)

    # Checking that sample is within the hypercube and 2D
    if not sample.ndim == 2:
        raise ValueError("Sample is not a 2D array")

    if not (np.all(sample >= 0) and np.all(sample <= 1)):
        raise ValueError("Sample is not in unit hypercube")

    # Checking that x_new is within the hypercube and 1D
    if not x_new.ndim == 1:
        raise ValueError("x_new is not a 1D array")

    if not (np.all(x_new >= 0) and np.all(x_new <= 1)):
        raise ValueError("x_new is not in unit hypercube")

    n = len(sample) + 1
    abs_ = abs(x_new - 0.5)

    # derivation from P.T. Roy (@tupui)
    disc1 = -2 / n * np.prod(1 + 1 / 2 * abs_ - 1 / 2 * abs_ ** 2)
    disc2 = (
        2
        / (n ** 2)
        * np.sum(
            np.prod(
                1
                + 1 / 2 * abs_
                + 1 / 2 * abs(sample - 0.5)
                - 1 / 2 * abs(x_new - sample),
                axis=1,
            )
        )
    )
    disc3 = 1 / (n ** 2) * np.prod(1 + abs_)

    return initial_disc + disc1 + disc2 + disc3


def primes_from_2_to(n):
    """Prime numbers from 2 to *n*.
    Parameters
    ----------
    n : int
        Sup bound with ``n >= 6``.
    Returns
    -------
    primes : list(int)
        Primes in ``2 <= p < n``.
    Notes
    -----
    Taken from [1]_ by P.T. Roy, written consent given on 23.04.2021
    by the original author, Bruno Astrolino, for free use in SciPy under
    the 3-clause BSD.
    References
    ----------
    .. [1] `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        k = 3 * i + 1 | 1
        sieve[k * k // 3 :: 2 * k] = False
        sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def n_primes(n):
    """List of the n-first prime numbers.
    Parameters
    ----------
    n : int
        Number of prime numbers wanted.
    Returns
    -------
    primes : list(int)
        List of primes.
    """
    primes = [
        2,
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
        101,
        103,
        107,
        109,
        113,
        127,
        131,
        137,
        139,
        149,
        151,
        157,
        163,
        167,
        173,
        179,
        181,
        191,
        193,
        197,
        199,
        211,
        223,
        227,
        229,
        233,
        239,
        241,
        251,
        257,
        263,
        269,
        271,
        277,
        281,
        283,
        293,
        307,
        311,
        313,
        317,
        331,
        337,
        347,
        349,
        353,
        359,
        367,
        373,
        379,
        383,
        389,
        397,
        401,
        409,
        419,
        421,
        431,
        433,
        439,
        443,
        449,
        457,
        461,
        463,
        467,
        479,
        487,
        491,
        499,
        503,
        509,
        521,
        523,
        541,
        547,
        557,
        563,
        569,
        571,
        577,
        587,
        593,
        599,
        601,
        607,
        613,
        617,
        619,
        631,
        641,
        643,
        647,
        653,
        659,
        661,
        673,
        677,
        683,
        691,
        701,
        709,
        719,
        727,
        733,
        739,
        743,
        751,
        757,
        761,
        769,
        773,
        787,
        797,
        809,
        811,
        821,
        823,
        827,
        829,
        839,
        853,
        857,
        859,
        863,
        877,
        881,
        883,
        887,
        907,
        911,
        919,
        929,
        937,
        941,
        947,
        953,
        967,
        971,
        977,
        983,
        991,
        997,
    ][:n]

    if len(primes) < n:
        big_number = 2000
        while "Not enough primes":
            primes = primes_from_2_to(big_number)[:n]
            if len(primes) == n:
                break
            big_number += 1000

    return primes


def van_der_corput(n, base=2, start_index=0, scramble=False, seed=None):
    """Van der Corput sequence.
    Pseudo-random number generator based on a b-adic expansion.
    Scrambling uses permutations of the remainders (see [1]_). Multiple
    permutations are applied to construct a point. The sequence of
    permutations has to be the same for all points of the sequence.
    Parameters
    ----------
    n : int
        Number of element of the sequence.
    base : int, optional
        Base of the sequence. Default is 2.
    start_index : int, optional
        Index to start the sequence from. Default is 0.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
    Returns
    -------
    sequence : list (n,)
        Sequence of Van der Corput.
    References
    ----------
    .. [1] A. B. Owen. "A randomized Halton algorithm in R",
       arXiv:1706.02808, 2017.
    """
    rng = check_random_state(seed)
    sequence = np.zeros(n)

    quotient = np.arange(start_index, start_index + n)
    b2r = 1 / base

    while (1 - b2r) < 1:
        remainder = quotient % base

        if scramble:
            # permutation must be the same for all points of the sequence
            perm = rng.permutation(base)
            remainder = perm[np.array(remainder).astype(int)]

        sequence += remainder * b2r
        b2r /= base
        quotient = (quotient - remainder) / base

    return sequence


class QMCEngine(ABC):
    """A generic Quasi-Monte Carlo sampler class meant for subclassing.
    QMCEngine is a base class to construct a specific Quasi-Monte Carlo
    sampler. It cannot be used directly as a sampler.
    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
    Notes
    -----
    By convention samples are distributed over the half-open interval
    ``[0, 1)``. Instances of the class can access the attributes: ``d`` for
    the dimension; and ``rng`` for the random number generator (used for the
    ``seed``).
    **Subclassing**
    When subclassing `QMCEngine` to create a new sampler,  ``__init__`` and
    ``random`` must be redefined.
    * ``__init__(d, seed=None)``: at least fix the dimension. If the sampler
      does not take advantage of a ``seed`` (deterministic methods like
      Halton), this parameter can be omitted.
    * ``random(n)``: draw ``n`` from the engine and increase the counter
      ``num_generated`` by ``n``.
    Optionally, two other methods can be overwritten by subclasses:
    * ``reset``: Reset the engine to it's original state.
    * ``fast_forward``: If the sequence is deterministic (like Halton
      sequence), then ``fast_forward(n)`` is skipping the ``n`` first draw.
    Examples
    --------
    To create a random sampler based on ``np.random.random``, we would do the
    following:
    >>> from scipy.stats import qmc
    >>> class RandomEngine(qmc.QMCEngine):
    ...     def __init__(self, d, seed=None):
    ...         super().__init__(d=d, seed=seed)
    ...
    ...
    ...     def random(self, n=1):
    ...         self.num_generated += n
    ...         return self.rng.random((n, self.d))
    ...
    ...
    ...     def reset(self):
    ...         super().__init__(d=self.d, seed=self.rng_seed)
    ...         return self
    ...
    ...
    ...     def fast_forward(self, n):
    ...         self.random(n)
    ...         return self
    After subclassing `QMCEngine` to define the sampling strategy we want to
    use, we can create an instance to sample from.
    >>> engine = RandomEngine(2)
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])
    We can also reset the state of the generator and resample again.
    >>> _ = engine.reset()
    >>> engine.random(5)
    array([[0.22733602, 0.31675834],  # random
           [0.79736546, 0.67625467],
           [0.39110955, 0.33281393],
           [0.59830875, 0.18673419],
           [0.67275604, 0.94180287]])
    """

    @abstractmethod
    def __init__(self, d, seed=None):
        if not np.issubdtype(type(d), np.integer):
            raise ValueError("d must be an integer value")

        self.d = d
        self.rng = check_random_state(seed)
        self.rng_seed = copy.deepcopy(seed)
        self.num_generated = 0

    @abstractmethod
    def random(self, n=1):
        """Draw `n` in the half-open interval ``[0, 1)``.
        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space.
            Default is 1.
        Returns
        -------
        sample : array_like (n, d)
            QMC sample.
        """
        # self.num_generated += n

    def reset(self):
        """Reset the engine to base state.
        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.
        """
        seed = copy.deepcopy(self.rng_seed)
        self.rng = check_random_state(seed)
        self.num_generated = 0
        return self

    def fast_forward(self, n):
        """Fast-forward the sequence by `n` positions.
        Parameters
        ----------
        n : int
            Number of points to skip in the sequence.
        Returns
        -------
        engine : QMCEngine
            Engine reset to its base state.
        """
        self.random(n=n)
        return self


class Halton(QMCEngine):
    """Halton sequence.
    Pseudo-random number generator that generalize the Van der Corput sequence
    for multiple dimensions. The Halton sequence uses the base-two Van der
    Corput sequence for the first dimension, base-three for its second and
    base-:math:`n` for its n-dimension.
    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    scramble : bool, optional
        If True, use Owen scrambling. Otherwise no scrambling is done.
        Default is True.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
    Notes
    -----
    The Halton sequence has severe striping artifacts for even modestly
    large dimensions. These can be ameliorated by scrambling. Scrambling
    also supports replication-based error estimates and extends
    applicabiltiy to unbounded integrands.
    References
    ----------
    .. [1] Halton, "On the efficiency of certain quasi-random sequences of
       points in evaluating multi-dimensional integrals", Numerische
       Mathematik, 1960.
    .. [2] A. B. Owen. "A randomized Halton algorithm in R",
       arXiv:1706.02808, 2017.
    Examples
    --------
    Generate samples from a low discrepancy sequence of Halton.
    >>> from scipy.stats import qmc
    >>> sampler = qmc.Halton(d=2, scramble=False)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.        , 0.        ],
           [0.5       , 0.33333333],
           [0.25      , 0.66666667],
           [0.75      , 0.11111111],
           [0.125     , 0.44444444]])
    Compute the quality of the sample using the discrepancy criterion.
    >>> qmc.discrepancy(sample)
    0.088893711419753
    If some wants to continue an existing design, extra points can be obtained
    by calling again `random`. Alternatively, you can skip some points like:
    >>> _ = sampler.fast_forward(5)
    >>> sample_continued = sampler.random(n=5)
    >>> sample_continued
    array([[0.3125    , 0.37037037],
           [0.8125    , 0.7037037 ],
           [0.1875    , 0.14814815],
           [0.6875    , 0.48148148],
           [0.4375    , 0.81481481]])
    Finally, samples can be scaled to bounds.
    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample_continued, l_bounds, u_bounds)
    array([[3.125     , 3.11111111],
           [8.125     , 4.11111111],
           [1.875     , 2.44444444],
           [6.875     , 3.44444444],
           [4.375     , 4.44444444]])
    """

    def __init__(self, d, scramble=True, seed=None):
        super().__init__(d=d, seed=seed)
        self.seed = seed
        self.base = n_primes(d)
        self.scramble = scramble

    def random(self, n=1):
        """Draw `n` in the half-open interval ``[0, 1)``.
        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        Returns
        -------
        sample : array_like (n, d)
            QMC sample.
        """
        # Generate a sample using a Van der Corput sequence per dimension.
        # important to have ``type(bdim) == int`` for performance reason
        sample = [
            van_der_corput(
                n,
                int(bdim),
                self.num_generated,
                scramble=self.scramble,
                seed=copy.deepcopy(self.seed),
            )
            for bdim in self.base
        ]

        self.num_generated += n
        return np.array(sample).T.reshape(n, self.d)


class LatinHypercube(QMCEngine):
    """Latin hypercube sampling (LHS).
    A Latin hypercube sample [1]_ generates :math:`n` points in
    :math:`[0,1)^{d}`. Each univariate marginal distribution is stratified,
    placing exactly one point in :math:`[j/n, (j+1)/n)` for
    :math:`j=0,1,...,n-1`. They are still applicable when :math:`n << d`.
    LHS is extremely effective on integrands that are nearly additive [2]_.
    LHS on :math:`n` points never has more variance than plain MC on
    :math:`n-1` points [3]_. There is a central limit theorem for LHS [4]_,
    but not necessarily for optimized LHS.
    Parameters
    ----------
    d : int
        Dimension of the parameter space.
    centered : bool, optional
        Center the point within the multi-dimensional grid. Default is False.
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` singleton is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used.
    References
    ----------
    .. [1] Mckay et al., "A Comparison of Three Methods for Selecting Values
       of Input Variables in the Analysis of Output from a Computer Code",
       Technometrics, 1979.
    .. [2] M. Stein, "Large sample properties of simulations using Latin
       hypercube sampling." Technometrics 29, no. 2: 143-151, 1987.
    .. [3] A. B. Owen, "Monte Carlo variance of scrambled net quadrature."
       SIAM Journal on Numerical Analysis 34, no. 5: 1884-1910, 1997
    .. [4]  Loh, W.-L. "On Latin hypercube sampling." The annals of statistics
       24, no. 5: 2058-2080, 1996.
    Examples
    --------
    Generate samples from a Latin hypercube generator.
    >>> from scipy.stats import qmc
    >>> sampler = qmc.LatinHypercube(d=2)
    >>> sample = sampler.random(n=5)
    >>> sample
    array([[0.1545328 , 0.53664833],  # random
           [0.84052691, 0.06474907],
           [0.52177809, 0.93343721],
           [0.68033825, 0.36265316],
           [0.26544879, 0.61163943]])
    Compute the quality of the sample using the discrepancy criterion.
    >>> qmc.discrepancy(sample)
    0.019558034794794565  # random
    Finally, samples can be scaled to bounds.
    >>> l_bounds = [0, 2]
    >>> u_bounds = [10, 5]
    >>> qmc.scale(sample, l_bounds, u_bounds)
    array([[1.54532796, 3.609945  ],  # random
           [8.40526909, 2.1942472 ],
           [5.2177809 , 4.80031164],
           [6.80338249, 3.08795949],
           [2.65448791, 3.83491828]])
    """

    def __init__(self, d, centered=False, seed=None):
        super().__init__(d=d, seed=seed)
        self.centered = centered

    def random(self, n=1):
        """Draw `n` in the half-open interval ``[0, 1)``.
        Parameters
        ----------
        n : int, optional
            Number of samples to generate in the parameter space. Default is 1.
        Returns
        -------
        sample : array_like (n, d)
            LHS sample.
        """
        if self.centered:
            samples = 0.5
        else:
            samples = self.rng.uniform(size=(n, self.d))

        perms = np.tile(np.arange(1, n + 1), (self.d, 1))
        for i in range(self.d):
            self.rng.shuffle(perms[i, :])
        perms = perms.T

        samples = (perms - samples) / n
        self.num_generated += n
        return samples

    def reset(self):
        """Reset the engine to base state.
        Returns
        -------
        engine : LatinHypercube
            Engine reset to its base state.
        """
        self.__init__(d=self.d, centered=self.centered, seed=self.rng_seed)
        self.num_generated = 0
        return self
