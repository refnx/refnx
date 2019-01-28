from scipy.stats import rv_continuous, uniform
from scipy.stats._distn_infrastructure import rv_frozen
from scipy._lib._util import check_random_state
import numpy as np


class Bounds(object):
    """
    A base class that describes the probability distribution for a parameter

    """
    def __init__(self, seed=None):
        self._random_state = check_random_state(seed)

    def logp(self, value):
        """
        Calculate the log-prior probability of a value with the probability
        distribution.

        Parameters
        ----------
        val : float or np.ndarray
            variates to calculate the log-probability for

        Returns
        -------
        arr : float or np.ndarray
            The log-probabilty corresponding to each of the variates

        """
        raise NotImplementedError

    def valid(self, val):
        """
        Checks whether a parameter value is within the support of the
        distribution. If it isn't then it returns a value that is within the
        support.

        Parameters
        ----------
        val : array-like
            values to examine

        Returns
        -------
        valid : array-like
            valid values within the support

        """
        raise NotImplementedError

    def rvs(self, size=1, random_state=None):
        """
        Generate random variates from the probability distribution.

        Parameters
        ----------
        size : int or tuple
            Specifies the number, or array shape, of random variates to return.
        random_state : None, int, float or np.random.RandomState
            For reproducible sampling

        Returns
        -------
        arr : array-like
            Random variates from within the probability distribution.

        """
        raise NotImplementedError


class PDF(Bounds):
    """
    A class that describes the probability distribution for a parameter.

    Parameters
    ----------
    rv : :class:`scipy.stats.rv_continuous` or Object
        A continuous probability distribution. If `rv` is not an
        `rv_continuous`, then it must implement the `logpdf` and `rvs`
        methods.
    seed : int, float or np.random.RandomState
        Seed for random variates

    Examples
    --------

    >>> import scipy.stats as stats
    >>> from refnx.analysis import Parameter, PDF
    >>> p = Parameter(0.5)
    >>> # use a normal distribution for prior, mean=5 and sd=1.
    >>> p.bounds = PDF(stats.norm(5, 1))
    >>> p.logp(), stats.norm.logpdf(0.5, 5, 1)
    (-11.043938533204672, -11.043938533204672)

    """
    def __init__(self, rv, seed=None):
        super(PDF, self).__init__(seed=seed)
        # we'll accept any object so long as it has logpdf and rvs methods
        if hasattr(rv, 'logpdf') and hasattr(rv, 'rvs'):
            self.rv = rv
        else:
            raise ValueError("You must initialise PDF with an object that has"
                             " logpdf and rvs methods")

    def __repr__(self):
        return "PDF({rv!r})".format(**self.__dict__)

    def logp(self, val):
        """
        Calculate the log-prior probability of a value with the probability
        distribution.

        Parameters
        ----------
        val : float or np.ndarray
            variate to calculate the log-probability for

        Returns
        -------
        arr : float or np.ndarray
            The log-probabilty corresponding to each of the variates

        """
        return self.rv.logpdf(val)

    def valid(self, val):
        """
        Checks whether a parameter value is within the support of the
        distribution. If it isn't then it returns a *random* value that is
        within the support.

        Parameters
        ----------
        val : array-like
            values to examine

        Returns
        -------
        valid : array-like
            valid values within the support
        """
        _val = np.asfarray(val)
        valid = np.where(np.isfinite(self.logp(_val)),
                         _val,
                         self.rv.rvs(size=_val.shape))

        return valid

    def rvs(self, size=1, random_state=None):
        """
        Generate random variates from the probability distribution.

        Parameters
        ----------
        size : int or tuple
            Specifies the number, or array shape, of random variates to return.
        random_state : None, int, float or np.random.RandomState
            For reproducible sampling

        Returns
        -------
        arr : array-like
            Random variates from within the probability distribution.

        """
        if random_state is None:
            random_state = self._random_state
        return self.rv.rvs(size=size, random_state=random_state)


class Interval(Bounds):
    r"""
    Describes a uniform probability distribution. May be open, semi-open,
    or closed.

    Parameters
    ----------
    lb : float
        The lower bound
    ub : float
        The upper bound
    seed : None, int, float or np.random.RandomState
        For reproducible sampling

    Examples
    --------

    >>> from refnx.analysis import Parameter, Interval
    >>> p = Parameter(1)
    >>> # closed interval
    >>> p.bounds = Interval(0, 10)
    >>> p.logp([5, -1])
    array([-2.30258509,        -inf])

    A semi-closed interval will still prevent the fitter from accessing
    impossible locations.

    >>> p.bounds = Interval(lb=-10)
    >>> p.logp([5, -1])
    array([0., 0.])

    """
    def __init__(self, lb=-np.inf, ub=np.inf, seed=None):
        super(Interval, self).__init__(seed=seed)
        if lb is None:
            lb = -np.inf
        if ub is None:
            ub = np.inf

        self._lb = lb
        self._ub = ub
        self.rv = None
        self._closed_bounds = False
        self._set_bounds(self._lb, self._ub)

    def _set_bounds(self, lb, ub):
        t_lb, t_ub = lb, ub
        self._lb = min(t_lb, t_ub)
        self._ub = max(t_lb, t_ub)
        self._closed_bounds = False

        if np.isnan([self._lb, self._ub]).any():
            raise ValueError("Can't set Interval with NaN")

        if np.isfinite([self._lb, self._ub]).all():
            self._closed_bounds = True
            self.rv = uniform(self._lb, self._ub - self._lb)
        else:
            self._closed_bounds = False

    def __repr__(self):
        lb, ub = self.lb, self.ub
        if np.isneginf(self.lb):
            lb = '-np.inf'
        if np.isinf(self.ub):
            ub = 'np.inf'
        return "Interval(lb={}, ub={})".format(lb, ub)

    def __str__(self):
        return '[{0}, {1}]'.format(self.lb, self.ub)

    @property
    def lb(self):
        """
        Lower bound of uniform distribution
        """
        return self._lb

    @lb.setter
    def lb(self, val):
        if val is None:
            val = -np.inf
        self._set_bounds(val, self._ub)

    @property
    def ub(self):
        """
        Upper bound of uniform distribution
        """
        return self._ub

    @ub.setter
    def ub(self, val):
        if val is None:
            val = np.inf
        self._set_bounds(self._lb, val)

    def logp(self, val):
        _val = np.asfarray(val)
        valid = np.logical_and(self._lb <= _val, _val <= self._ub)

        if self._closed_bounds:
            # TODO special case where lb==ub==val?
            prob = np.where(valid, np.log(1 / (self._ub - self._lb)), -np.inf)
        else:
            prob = np.where(valid, 0, -np.inf)
        return prob

    def valid(self, val):
        """
        Checks whether a parameter value is within the support of the
        Interval. If it isn't then it returns a value that is within the
        support.
        If the Interval is closed (i.e. lower and upper bounds are both
        specified) then invalid values will be corrected by random samples
        taken between the lower and upper bounds.
        If the interval is semi-open, only one of the bounds being
        specified, then invalid values will be corrected by the value being
        reflected by the same distance from the relevant limit.

        Parameters
        ----------
        val : array-like
            values to examine

        Returns
        -------
        valid : array-like
            values within the support

        Examples
        --------
        >>> b = Interval(0, 10)
        >>> b.valid(11.5)
        8.5
        """
        _val = np.asfarray(val)
        if self._closed_bounds:
            valid = np.where(np.isfinite(self.logp(_val)),
                             _val,
                             self.rvs(size=_val.shape))
        else:
            valid = np.where(_val < self._ub, _val, 2 * self._ub - _val)
            valid = np.where(valid > self._lb, valid, 2 * self._lb - valid)

        return valid

    def rvs(self, size=1, random_state=None):
        if self._closed_bounds:
            if random_state is None:
                random_state = self._random_state
            return self.rv.rvs(size=size, random_state=random_state)
        else:
            raise RuntimeError("Can't ask for a random variate from a"
                               " semi-closed interval")
