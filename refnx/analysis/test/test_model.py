import pickle

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose, assert_raises)

from refnx.analysis import Parameter, Model


def line(x, params, *args, **kwds):
    p_arr = np.array(params)
    return p_arr[0] + x * p_arr[1]


def line2(x, p):
    return p['c'].value + p['m'].value * x


def line3(x, params, x_err=None):
    pass


class TestModel(object):

    def setup_method(self):
        pass

    def test_evaluation(self):
        c = Parameter(1.0, name='c')
        m = Parameter(2.0, name='m')
        p = c | m

        fit_model = Model(p, fitfunc=line)
        x = np.linspace(0, 100., 20)
        y = 2. * x + 1.

        # different ways of getting the model instance to evaluate
        assert_equal(fit_model.model(x, p), y)
        assert_equal(fit_model(x, p), y)
        assert_equal(fit_model.model(x), y)
        assert_equal(fit_model(x), y)

        # can we pickle the model object
        pkl = pickle.dumps(fit_model)
        unpkl = pickle.loads(pkl)
        assert_equal(unpkl(x), y)

        # you should be able to use a lambda
        fit_model = Model(p, fitfunc=line2)
        assert_equal(fit_model(x, p), y)

        # and swap the order of parameters - retrieve by key
        p = m | c
        fit_model = Model(p, fitfunc=line2)
        assert_equal(fit_model(x, p), y)

    def test_xerr(self):
        c = Parameter(1.0, name='c')
        m = Parameter(2.0, name='m')
        p = c | m

        fit_model = Model(p, fitfunc=line3)
        assert_(fit_model._fitfunc_has_xerr is True)

        fit_model = Model(p, fitfunc=line2)
        assert_(fit_model._fitfunc_has_xerr is False)
