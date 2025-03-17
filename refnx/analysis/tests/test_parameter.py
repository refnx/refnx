import pickle

import pytest
import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_,
    assert_allclose,
)
from scipy.stats import norm, uniform

from refnx.analysis import Interval, PDF, Parameter, Parameters, is_parameters
from refnx.analysis.parameter import (
    constraint_tree,
    build_constraint_from_tree,
    possibly_create_parameter,
    is_parameter,
    _BinaryOp,
    sequence_to_parameters,
)


class TestParameter:
    def setup_method(self):
        pass

    def test_parameter(self):
        # simple test of constraint
        x = Parameter(5.0)
        y = Parameter(1.0)

        y.constraint = x
        assert x in y.dependencies()

        y.constraint = x * 2.0
        assert_equal(y.value, x.value * 2.0)

        # parameter should be in y's dependencies
        assert x in y._deps
        assert x in y.dependencies()

        # if you've constrained a parameter it shouldn't be varying
        assert_(y.vary is False)

        # you can't set a constraint on a parameter with an expression that
        # already involves the parameter
        from pytest import raises

        with raises(ValueError):
            x.constraint = y

        # try a negative value
        x.value = -1.0
        assert_equal(y.value, -2.0)

        # nested constraints
        z = Parameter(1.0)
        z.constraint = x + y
        assert_equal(z.value, -3)
        # check that nested circular constraints aren't allowed
        with raises(ValueError):
            x.constraint = z

        # z = x + y --> z = x + 2*x
        # therefore y shouldn't be in z's dependencies, but x should be.
        assert x in z.dependencies()
        assert y not in z.dependencies()

        # absolute value constraint
        y.constraint = abs(x)
        assert_equal(y.value, 1)

        # sin constraint
        y.constraint = np.sin(x) + 2.0
        assert_equal(y.value, 2.0 + np.sin(x.value))

        x.value = 10
        assert_equal(y.value, 2.0 + np.sin(10))

    def test_is_parameter(self):
        p = Parameter(1.0)
        assert is_parameter(p)

        assert isinstance(0.5 * p, _BinaryOp)
        assert is_parameter(0.5 * p)

    def test_print(self):
        print(Parameter(0, vary=True, bounds=(1, 2), name="as"))

    def test_repr(self):
        p = Parameter(value=5, name="pop", vary=True)
        q = eval(repr(p))
        assert q.name == "pop"
        assert_allclose(q.value, p.value)

        p.bounds.lb = -5
        q = eval(repr(p))
        assert_allclose(q.bounds.lb, -5)
        assert_allclose(q.bounds.ub, np.inf)

        p = Parameter(value=5, vary=True)
        q = eval(repr(p))
        assert_allclose(q.value, p.value)
        assert_allclose(q.vary, p.vary)

    def test_func_attribute(self):
        # a Parameter object should have math function attributes
        a = Parameter(1)
        assert_(hasattr(a, "sin"))

    def test_remove_constraint(self):
        x = Parameter(5.0)
        y = Parameter(1.0)
        y.constraint = x * 2.0
        y.constraint = None
        assert y.vary is False
        assert_equal(y.value, 10)
        assert y._constraint is None
        assert y._constraint_args is None

    def test_parameter_bounds(self):
        x = Parameter(4, bounds=Interval(-4, 4))
        assert_equal(x.logp(), uniform.logpdf(0, -4, 8))

        x.bounds = None
        assert_(isinstance(x._bounds, Interval))
        assert_equal(x.bounds.lb, -np.inf)
        assert_equal(x.bounds.ub, np.inf)
        assert_equal(x.logp(), 0)

        x.setp(bounds=norm(0, 1))
        assert_almost_equal(x.logp(1), norm.logpdf(1, 0, 1))

        # all created parameters were mistakenly being given the same
        # default bounds instance!
        x = Parameter(4)
        y = Parameter(5)
        assert_(id(x.bounds) != id(y.bounds))

    def test_range(self):
        x = Parameter(0.0)
        x.range(-1, 1.0)
        assert_equal(x.bounds.lb, -1)
        assert_equal(x.bounds.ub, 1.0)

        vals = x.valid(np.linspace(-100, 100, 10000))
        assert_(np.min(vals) >= -1)
        assert_(np.max(vals) <= 1)

    def test_parameter_attrib(self):
        # each parameter should have bound math methods
        a = Parameter(1.0)
        assert_(hasattr(a, "sin"))

    def test_pickle(self):
        # a parameter and a constrained parameter should be pickleable
        bounds = PDF(norm(1.0, 2.0))
        x = Parameter(1, bounds=bounds)
        pkl = pickle.dumps(x)
        unpkl = pickle.loads(pkl)

        # test pickling on a constrained parameter system
        a = Parameter(1.0)
        b = Parameter(2.0)
        b.constraint = np.sin(a)

        assert_(hasattr(a, "sin"))
        c = [a, b]
        pkl = pickle.dumps(c)
        unpkl = pickle.loads(pkl)
        d, e = unpkl
        d.value = 2.0

        assert_equal(e.value, np.sin(2.0))
        # should still have all math functions
        assert_(hasattr(d, "sin"))
        assert_(hasattr(a, "sin"))

    def test_or(self):
        # concatenation of Parameter instances
        a = Parameter(1, name="a")
        b = Parameter(2, name="b")
        c = Parameters(name="c")
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

    def test_constraint_analyser(self):
        a = Parameter(1)
        b = Parameter(2, constraint=a)
        c = Parameter(2.0)

        d = Parameter(3, constraint=np.cos(b + np.sin(a) + 2 * (a + b + c)))
        val = d.value

        tree = constraint_tree(d.constraint)
        new_constraint = build_constraint_from_tree(tree)
        assert_allclose(new_constraint.value, val)

        a.value = 10
        assert_allclose(new_constraint.value, d.value)

        # inject constraint into parameter
        e = Parameter(1)
        e.constraint = new_constraint
        a.value = 11
        assert_allclose(e.value, d.value)

        # check that it's possible to build a constraint tree from a single
        # param
        tree = constraint_tree(b.constraint)
        new_constraint = build_constraint_from_tree(tree)
        e = Parameter(1)
        e.constraint = new_constraint
        a.value = 0.1234
        assert_allclose(e.value, a.value)

        # check that it's possible to build a constraint tree from a single
        # param
        e = Parameter(1)
        e.constraint = 2
        assert_allclose(e.value, 2)

    def test_function_constraint(self):
        a = Parameter(1)
        b = Parameter(2)
        c = Parameter(3)
        a.set_constraint(np.sin(2 * b))
        assert_allclose(a.value, -0.7568024953079282)

        def f(*args):
            return np.sin(args[0] * args[1][0])

        a.set_constraint(f, args=(2, [b, c]))
        assert_allclose(a.value, -0.7568024953079282)
        assert b in a._deps
        assert b in a.dependencies()
        assert c in a._deps
        assert a.constraint is f

        # check that a subsequently derived constraint works
        p = Parameter(np.nan)
        p.constraint = 1 - a
        assert_equal(float(p), 1.0 - float(a))

        # this should be a problem as the parameter is in the list of args
        with pytest.raises(ValueError):
            a.set_constraint(f, args=(2, b, a))

    def test_possibly_create_parameter(self):
        p = Parameter(10, bounds=(1.0, 2.0))
        q = possibly_create_parameter(p, vary=True, bounds=(-1.0, 2.0))
        assert q is p
        assert_allclose(p.bounds.lb, 1)
        assert_allclose(p.bounds.ub, 2)

        q = possibly_create_parameter(10, vary=True, bounds=(-1.0, 2.0))
        assert_allclose(q.value, 10)
        assert_allclose(q.bounds.lb, -1)
        assert_allclose(q.bounds.ub, 2.0)
        assert q.vary

        q = possibly_create_parameter(0.5 * p)
        assert isinstance(q, _BinaryOp)

    def test_dependencies(self):
        p1 = Parameter(1, "p1", vary=True)
        p2 = Parameter(2, "p2", vary=False)

        p_dep = p1 + p2
        p_dep2 = p1 + p_dep

        assert p_dep2.dependencies() == [p1, p2]


class TestParameters:
    def setup_method(self):
        self.a = Parameter(1, name="a")
        self.b = Parameter(2, name="b")
        self.m = Parameters()
        self.m.append(self.a)
        self.m.append(self.b)

    def test_retrieve_by_name(self):
        p = self.m["a"]
        assert_(p is self.a)

        # or by index
        p = self.m[0]
        assert_(p is self.a)

    def test_repr(self):
        p = Parameter(value=5, vary=False, name="test")
        g = Parameters(name="name")
        f = Parameters()
        f.append(p)
        f.append(g)

        q = eval(repr(f))
        assert q.name is None
        assert_equal(q[0].value, 5)
        assert q[0].vary is False
        assert isinstance(q[1], Parameters)

    def test_set_by_name(self):
        c = Parameter(3.0)
        self.m["a"] = c
        assert_(self.m[0] is c)

        # can't set an entry by name, if there isn't an existing name in this
        # Parameters instance.
        from pytest import raises

        with raises(ValueError):
            self.m["abc"] = c

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

    def test_varying_parameters_with_dependencies(self):
        p1 = Parameter(1, "p1", vary=True)
        p2 = Parameter(2, "p2", vary=False)
        p3 = Parameter(3, "p3", vary=False)
        p4 = Parameter(4, "p4", vary=True)

        p_dep = p1 + p2
        pars = Parameters([p2, p3, p4, p_dep])
        assert pars.varying_parameters() == [p4, p1]

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
        c = Parameters(name="c")
        d = c | self.m
        assert_(d.name == "c")

        # c and d should not be the same object
        assert c is not d

    def test_ior(self):
        # concatenation of Parameters
        # Parameters with Parameter
        c = Parameters(name="c")
        c |= self.b
        assert_equal(len(c), 1)
        assert_equal(len(c.flattened()), 1)
        assert_(c.flattened()[0] is self.b)

        # Parameters with Parameters
        c = Parameters(name="c")
        c |= self.m
        assert_(c.name == "c")
        assert_equal(len(c), 1)
        assert_equal(len(c.flattened()), 2)
        assert_(c.flattened()[1] is self.b)

    def test_sequence_conversion(self):
        a = Parameter(1)
        b = Parameter(2)
        p = sequence_to_parameters([a, [1, 2, 3, [b]]])
        assert isinstance(p, Parameters)
        assert len(p) == 5
