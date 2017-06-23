import unittest
import pickle

from refnx.analysis import Interval, PDF

import numpy as np
from numpy.testing import (assert_equal)
from scipy.stats import norm, truncnorm


class TestBounds(unittest.TestCase):

    def setUp(self):
        pass

    def test_interval(self):
        # open interval should have lnprob of 0
        interval = Interval()
        assert_equal(interval.lnprob(0), 0)

        # semi closed interval
        interval.ub = 1000
        assert_equal(interval.lnprob(0), 0)
        assert_equal(interval.lnprob(1001), -np.inf)

        interval.lb = -1000
        assert_equal(interval.lnprob(-1001), -np.inf)
        assert_equal(interval.lnprob(0), np.log(1 / 2000.))

        # try and set lb higher than ub
        interval.lb = 1002
        assert_equal(interval.lb, 1000)
        assert_equal(interval.ub, 1002)

        # should get clipping if value is outside bounds
        assert_equal(interval.valid(1003), 1002)
        assert_equal(interval.valid(999), 1000)
        assert_equal(interval.valid(1001), 1001)

    def test_pdf(self):
        pdf = PDF(norm)

        # even if it's really far out it's still a valid value
        assert_equal(pdf.valid(1003), 1003)
        # logp
        assert_equal(pdf.lnprob(0), norm.logpdf(0))

        # examine dist with finite support
        pdf = PDF(truncnorm(-1, 1), seed=1)
        assert_equal(pdf.lnprob(-2), -np.inf)
        # extract a valid value from that distribution
        assert_equal(pdf.valid(-2), -0.14247656484294183)
        assert_equal(pdf.lnprob(-0.5), truncnorm.logpdf(-0.5, -1, 1))

        # obtain a random value of a bounds instance
        pdf.rvs(size=1)

    def test_pickle(self):
        bounds = PDF(norm(1., 2.))
        pkl = pickle.dumps(bounds)
        pickle.loads(pkl)

        bounds = Interval()
        pkl = pickle.dumps(bounds)
        pickle.loads(pkl)
