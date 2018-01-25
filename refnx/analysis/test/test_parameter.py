import pickle

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_,
                           assert_allclose)
from scipy.stats import norm, uniform

from refnx.analysis import (Interval, PDF, Parameter, Parameters,
                            is_parameters)


class TestParameter(object):

    def setup_method(self):
        pass

    def test_parameter(self):
        # simple test of constraint
        x = Parameter(5.)
        y = Parameter(1.)
        y.constraint = x * 2.
        assert_equal(y.value, x.value * 2.)

        # if you've constrained a parameter it shouldn't be varying
        assert_(y.vary is False)

        # you can't set a constraint on a parameter with an expression that
        # already involves the parameter
        from pytest import raises
        with raises(ValueError):
            x.constraint = y

        # try a negative value
        x.value = -1.
        assert_equal(y.value, -2.)

        # nested constraints
        z = Parameter(1.)
        z.constraint = x + y
        assert_equal(z.value, -3)
        # check that nested circular constraints aren't allowed
        with raises(ValueError):
            x.constraint = z

        # absolute value constraint
        y.constraint = abs(x)
        assert_equal(y.value, 1)

        # sin constraint
        y.constraint = np.sin(x) + 2.
        assert_equal(y.value, 2. + np.sin(x.value))

    def test_func_attribute(self):
        # a Parameter object should have math function attributes
        a = Parameter(1)
        assert_(hasattr(a, 'sin'))

    def test_remove_constraint(self):
        x = Parameter(5.)
        y = Parameter(1.)
        y.constraint = x * 2.
        y.constraint = None
        assert_(y.vary is False)
        assert_equal(y.value, 10)
        assert_(y._constraint is None)

    def test_parameter_bounds(self):
        x = Parameter(4, bounds=Interval(-4, 4))
        assert_equal(x.lnprob(), uniform.logpdf(0, -4, 8))

        x.bounds = None
        assert_(isinstance(x._bounds, Interval))
        assert_equal(x.bounds.lb, -np.inf)
        assert_equal(x.bounds.ub, np.inf)
        assert_equal(x.lnprob(), 0)

        x.setp(bounds=norm(0, 1))
        assert_almost_equal(x.lnprob(1), norm.logpdf(1, 0, 1))

    def test_range(self):
        x = Parameter(0.)
        x.range(-1, 1.)
        assert_equal(x.bounds.lb, -1)
        assert_equal(x.bounds.ub, 1.)

        vals = x.valid(np.linspace(-100, 100, 10000))
        assert_(np.min(vals) >= -1)
        assert_(np.max(vals) <= 1)

    def test_parameter_attrib(self):
        # each parameter should have bound math methods
        a = Parameter(1.)
        assert_(hasattr(a, 'sin'))

    def test_pickle(self):
        # a parameter and a constrained parameter should be pickleable
        bounds = PDF(norm(1., 2.))
        x = Parameter(1, bounds=bounds)
        pkl = pickle.dumps(x)
        unpkl = pickle.loads(pkl)

        # test pickling on a constrained parameter system
        a = Parameter(1.)
        b = Parameter(2.)
        b.constraint = np.sin(a)

        assert_(hasattr(a, 'sin'))
        c = [a, b]
        pkl = pickle.dumps(c)
        unpkl = pickle.loads(pkl)
        d, e = unpkl
        d.value = 2.
        assert_equal(e.value, np.sin(2.))
        # should still have all math functions
        assert_(hasattr(d, 'sin'))

    def test_or(self):
        # concatenation of Parameter instances
        a = Parameter(1, name='a')
        b = Parameter(2, name='b')
        c = Parameters(name='c')
        c.append(a)
        c.append(b)

        # concatenate Parameter instances
        d = a | b
        assert_(is_parameters(d))

        # concatenate Parameter with Parameters
        d = a | c
        assert_(is_parameters(d))
        assert_equal(len(d), 2)
        # a, a, b
        assert_equal(len(d.flattened()), 3)


class TestParameters(object):

    def setup_method(self):
        self.a = Parameter(1, name='a')
        self.b = Parameter(2, name='b')
        self.m = Parameters()
        self.m.append(self.a)
        self.m.append(self.b)

    def test_retrieve_by_name(self):
        p = self.m['a']
        assert_(p is self.a)

        # or by index
        p = self.m[0]
        assert_(p is self.a)

    def test_set_by_name(self):
        c = Parameter(3.)
        self.m['a'] = c
        assert_(self.m[0] is c)

        # can't set an entry by name, if there isn't an existing name in this
        # Parameters instance.
        from pytest import raises
        with raises(ValueError):
            self.m['abc'] = c

    def test_parameters(self):
        # we've added two parameters
        self.a.vary = True
        self.b.vary = True

        assert_equal(len(self.m.flattened()), 2)

        # the two entries should just be the objects
        assert_(self.m.varying_parameters()[0] is self.a)
        assert_(self.m.varying_parameters()[1] is self.b)

    def test_varying_parameters(self):
        # even though we've added a twice we should still only see 2
        # varying parameters
        self.a.vary = True
        self.b.vary = True
        p = self.a | self.b | self.a
        assert_equal(len(p.varying_parameters()), 2)

    def test_pickle_parameters(self):
        # need to check that Parameters can be pickled/unpickle
        pkl = pickle.dumps(self.m)
        pickle.loads(pkl)

    def test_or(self):
        # concatenation of Parameters
        # Parameters with Parameter
        c = self.m | self.b
        assert_equal(len(c), 3)
        assert_equal(len(c.flattened()), 3)
        assert_(c.flattened()[1] is self.b)
        assert_(c.flattened()[2] is self.b)

        # Parameters with Parameters
        c = Parameters(name='c')
        d = c | self.m
        assert_(d.name == 'c')

    def test_ior(self):
        # concatenation of Parameters
        # Parameters with Parameter
        c = Parameters(name='c')
        c |= self.b
        assert_equal(len(c), 1)
        assert_equal(len(c.flattened()), 1)
        assert_(c.flattened()[0] is self.b)

        # Parameters with Parameters
        c = Parameters(name='c')
        c |= self.m
        assert_(c.name == 'c')
        assert_equal(len(c), 1)
        assert_equal(len(c.flattened()), 2)
        assert_(c.flattened()[1] is self.b)
