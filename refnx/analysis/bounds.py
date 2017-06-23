from scipy.stats import rv_continuous, uniform
from scipy.stats._distn_infrastructure import rv_frozen
from scipy._lib._util import check_random_state
import numpy as np


class Bounds(object):
    def __init__(self, seed=None):
        self._random_state = check_random_state(seed)

    def lnprob(self, value):
        """
        Calculate the log-likelihood probability of a value with the bounds
        """
        raise NotImplementedError

    def valid(self, value):
        """
        Checks whether a parameter value is within the support of the
        distribution. If it isn't then it returns a value that is within the
        support.
        """
        raise NotImplementedError

    def rvs(self, size=1):
        raise NotImplementedError


class PDF(Bounds):
    def __init__(self, rv, seed=None):
        super(PDF, self).__init__(seed=seed)
        if isinstance(rv, rv_continuous) or isinstance(rv, rv_frozen):
            self.rv = rv
        else:
            raise ValueError("You must give PDF a scipy.stats.rv_continuous"
                             " instance")

    def __repr__(self):
        return repr(self.rv)

    def lnprob(self, val):
        """
        Calculate the log-likelihood probability of a value with the bounds
        """
        return self.rv.logpdf(val)

    def valid(self, val):
        """
        Checks whether a parameter value is within the support of the
        distribution. If it isn't then it returns a *random* value that is
        within the support.
        """
        retval = val
        if not np.isfinite(self.lnprob(val)):
            retval = self.rv.rvs(random_state=self._random_state)
        return retval

    def rvs(self, size=1):
        return self.rv.rvs(size=size)


class Interval(Bounds):
    def __init__(self, lb=-np.inf, ub=np.inf):
        super(Interval, self).__init__()
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
        return '[{0}, {1}]'.format(self.lb, self.ub)

    @property
    def lb(self):
        return self._lb

    @lb.setter
    def lb(self, val):
        self._set_bounds(val, self._ub)

    @property
    def ub(self):
        return self._ub

    @ub.setter
    def ub(self, val):
        self._set_bounds(self._lb, val)

    def lnprob(self, val):
        """
        Calculate the log-likelihood probability of a value with the bounds
        """
        if self._lb < val < self._ub:
            if self._closed_bounds:
                return np.log(1./(self._ub - self._lb))
            else:
                return 0
        else:
            return -np.inf

    def valid(self, val):
        """
        Checks whether a parameter value is within the support of the
        distribution. If it isn't then it returns a value that is within the
        support.
        """
        if val < self._lb:
            return self._lb
        elif val > self._ub:
            return self._ub
        else:
            return val

    def rvs(self, size=1):
        if self._closed_bounds:
            self.rv.rvs(size=size)
        else:
            raise RuntimeError("Can't ask for a random variate from a"
                               " semi-closed interval")
