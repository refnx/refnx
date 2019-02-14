import operator
from types import MethodType
from collections import UserList


import numpy as np
from refnx._lib import flatten, unique as f_unique
from refnx.analysis import Interval, PDF, Bounds


# Functions for making Functors
def MAKE_BINARY(opfn):
    return lambda self, other: (_BinaryOp(self, asMagicNumber(other), opfn))


def MAKE_RBINARY(opfn):
    return lambda self, other: (_BinaryOp(asMagicNumber(other), self, opfn))


def MAKE_UNARY(opfn):
    return lambda val: _UnaryOp(val, opfn)


def asMagicNumber(x):
    return x if isinstance(x, BaseParameter) else Constant(x)


# a function that takes a function that returns a function
# MAKE_BINARY = lambda opfn: lambda self, other: (
#     _BinaryOp(self, asMagicNumber(other), opfn))
# MAKE_RBINARY = lambda opfn: lambda self, other: (
#     _BinaryOp(asMagicNumber(other), self, opfn))
# MAKE_UNARY = lambda opfn: lambda val: _UnaryOp(val, opfn)
# asMagicNumber = lambda x: x if isinstance(x, BaseParameter) else Constant(x)

# mathematical operations that we might want to use in constraints
ops = {'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'arcsin': np.arcsin,
       'arccos': np.arccos, 'arctan': np.arctan, 'log': np.log,
       'log10': np.log10, 'exp': np.exp, 'sqrt': np.sqrt, 'sum': np.sum}
math_ops = {k: MAKE_UNARY(v) for k, v in ops.items()}

binary = [operator.add, operator.sub, operator.mul, operator.truediv,
          operator.floordiv, np.power, operator.pow,
          operator.mod]

unary = [operator.neg, operator.abs, np.sin, np.tan, np.cos, np.arcsin,
         np.arctan, np.arccos, np.log10, np.log, np.sqrt, np.exp]


class Parameters(UserList):
    """
    A sequence of Parameters

    Parameters
    ----------
    data : sequence
        A sequence of :class:`Parameter` or :class:`Parameters`
    name : str
        Name of this :class:`Parameters` instance
    """
    def __init__(self, data=(), name=None):
        super(Parameters, self).__init__()
        self.name = name
        self.data.extend(data)

    def __getitem__(self, i):
        if type(i) is str:
            try:
                _names = [param.name for param in self.data]
                idx = _names.index(i)
            except ValueError:
                return None
            finally:
                return self.data[idx]
        else:
            return self.data[i]

    def __setitem__(self, i, v):
        # i can be index or string
        # v has to be Parameters or Parameter
        if type(i) is str:
            _names = [param.name for param in self.data]
            # this will raise a KeyError if the key isn't in there.
            idx = _names.index(i)
            self.data[idx] = v
        else:
            self.data[i] = v

    def __repr__(self):
        return ("Parameters(data={data!r},"
                " name={name!r})".format(**self.__dict__))

    def __str__(self):
        s = list()
        s.append("{:_>80}".format(''))
        s.append("Parameters: {0: ^15}".format(repr(self.name)))

        for el in self._pprint():
            s.append(el)

        return '\n'.join(list(flatten(s)))

    def _pprint(self):
        for el in self.data:
            if is_parameters(el):
                yield str(el)
            else:
                yield str(el)

    def __contains__(self, item):
        """
        Does this instance contain a given :class:`Parameter`
        """
        return id(item) in [id(p) for p in f_unique(flatten(self.data))]

    def __ior__(self, other):
        """
        Concatenate Parameter(s). You can concatenate :class:`Parameters` with
        :class:`Parameter` or :class:`Parameters` instances.
        """
        # self |= other
        if not (is_parameter(other) or is_parameters(other)):
            raise ValueError("Can only concatenate a Parameter with another"
                             " Parameter or Parameters instance")
        self.append(other)
        return self

    def __or__(self, other):
        """
        concatenate Parameter(s). You can concatenate :class:`Parameters` with
        :class:`Parameter` or :class:`Parameters` instances.
        """
        # c = self | other
        if not (is_parameter(other) or is_parameters(other)):
            raise ValueError("Can only concatenate a Parameter with another"
                             " Parameter or Parameters instance")

        self.append(other)
        return self

    def logp(self):
        """
        Calculates logp for all the parameters

        Returns
        -------
        logp : float
            Log probability for all the parameters
        """
        # logp for all the parameters
        return np.sum([param.logp() for param in f_unique(flatten(self.data))
                       if param.vary])

    def __array__(self):
        """
        Convert Parameters to an array containing their values.
        """
        return np.array([float(p) for p
                         in flatten(self.data)])

    @property
    def pvals(self):
        """
        An array containing the values of all the :class:`Parameter` in this
        object.
        """
        return np.array(self)

    @pvals.setter
    def pvals(self, pvals):
        varying = [param for param in f_unique(flatten(self.data))
                   if param.vary]
        if np.size(pvals) == len(varying):
            [setattr(param, 'value', pvals[i]) for i, param
             in enumerate(varying)]
            return

        flattened_parameters = list(flatten(self.data))

        if np.size(pvals) == len(flattened_parameters):
            [setattr(param, 'value', pvals[i]) for i, param
             in enumerate(flattened_parameters)]
            return
        else:
            raise ValueError("You supplied the wrong number of values %d when "
                             "setting this Parameters.pvals attribute"
                             % len(pvals))

    @property
    def parameters(self):
        return self

    @property
    def nparam(self):
        return len(self.flattened())

    def flattened(self, unique=False):
        """
        A list of all the :class:`Parameter` contained in this object,
        including those contained within :class:`Parameters` at any depth.

        Parameters
        ----------
        unique : bool
            The list will only contain unique objects.

        Returns
        -------
        params : list
            A list of :class:`Parameter` contained in this object.

        """
        if unique:
            return list(f_unique(flatten(self.data)))
        else:
            return list(flatten(self.data))

    def names(self):
        """
        Returns
        -------
        names : list
            A list of all the names of all the :class:`Parameter` contained in
            this object.
        """
        return [param.name for param in flatten(self.data)]

    def nvary(self):
        """
        Returns
        -------
        nvary : int
            The number of :class:`Parameter` contained in this object that are
            allowed to vary.

        """
        return np.sum([1 for param in f_unique(flatten(self.data))
                       if param.vary])

    def constrained_parameters(self):
        """
        Returns
        -------
        constrained_parameters : list
            A list of unique :class:`Parameter` contained in this object that
            have constraints.
        """
        return [param for param in f_unique(flatten(self.data))
                if param.constraint is not None]

    def varying_parameters(self):
        """
        Unique list of varying parameters

        Returns
        -------
        p : list
            Unique list of varying parameters
        """
        p = [param for param in f_unique(flatten(self.data)) if param.vary]
        q = Parameters()
        q.data = p
        return q

    def pgen(self, ngen=1000, nburn=0, nthin=1):
        """
        Yield random parameter vectors from MCMC samples.

        Parameters
        ----------
        ngen : int, optional
            the number of samples to yield. The actual number of samples
            yielded is `min(ngen, chain.size)`
        nburn : int, optional
            discard this many steps from the start of the chain
        nthin : int, optional
            only accept every `nthin` samples from the chain

        Yields
        ------
        pvec : np.ndarray
            A randomly chosen parameter vector

        Notes
        -----
        The entire parameter vector is yielded, not only the varying
        parameters. The reason for this is that some parameters may possess a
        chain if they are not varying, because they are controlled by a
        constraint.
        """

        # it's still possible to have chains, even if there are no varying
        # parameters, if there are parameters that have constraints
        # generate for all params that have chains.
        chain_pars = [i for i, p in enumerate(self.flattened()) if
                      p.chain is not None]

        chains = np.array([np.ravel(param.chain[..., nburn::nthin]) for param
                           in self.flattened()
                           if param.chain is not None])

        if len(chains) == 0 or np.size(chains, 1) == 0:
            raise ValueError("There were no chains to sample from")

        samples = np.arange(np.size(chains, 1))

        choices = np.random.choice(samples,
                                   size=(min(ngen, samples.size),),
                                   replace=False)

        template_array = np.array(self.flattened())

        for choice in choices:
            template_array[chain_pars] = chains[..., choice]
            yield template_array


class BaseParameter(object):
    __add__ = MAKE_BINARY(operator.add)
    __sub__ = MAKE_BINARY(operator.sub)
    __mul__ = MAKE_BINARY(operator.mul)
    __truediv__ = MAKE_BINARY(operator.truediv)
    __pow__ = MAKE_BINARY(operator.pow)
    __radd__ = MAKE_RBINARY(operator.add)
    __rsub__ = MAKE_RBINARY(operator.sub)
    __rmul__ = MAKE_RBINARY(operator.mul)
    __rtruediv__ = MAKE_RBINARY(operator.truediv)
    __abs__ = MAKE_UNARY(operator.abs)
    __mod__ = MAKE_BINARY(operator.mod)
    __rmod__ = MAKE_RBINARY(operator.mod)
    __lt__ = MAKE_BINARY(operator.lt)
    __le__ = MAKE_BINARY(operator.le)
    __ge__ = MAKE_BINARY(operator.ge)
    __gt__ = MAKE_BINARY(operator.gt)
    __neg__ = MAKE_UNARY(operator.neg)

    def __init__(self):
        # add mathematical operations as methods to this object
        # math_ops = {k: MAKE_UNARY(v) for k, v in self.ops.items()}

        for k, v in math_ops.items():
            setattr(self, k, MethodType(v, self))

        self._vary = False
        self.name = None
        self._value = None
        self._bounds = None
        self._deps = []
        self.stderr = None
        self.chain = None

    def __getstate__(self):
        # the mathematical ops aren't unpickleable, so lets pop them. They'll
        # be reinstated when the constructor is called anyway.

        d = self.__dict__
        for k in ops:
            if k in d:
                d.pop(k)
        return d

    def __setstate__(self, state):
        # the mathematical ops aren't unpickleable, so lets pop them. They'll
        # be reinstated when the constructor is called anyway.
        for k, v in math_ops.items():
            setattr(self, k, MethodType(v, self))

        self.__dict__.update(state)

    def __or__(self, other):
        # concatenate parameter with parameter or parameters
        if not (is_parameter(other) or is_parameters(other)):
            raise ValueError("Can only concatenate a Parameter with another"
                             " Parameter or Parameters instance")

        p = Parameters()
        p.append(self)
        p.append(other)
        return p

    @property
    def vary(self):
        return self._vary

    def logp(self):
        raise NotImplementedError("Subclass of BaseParameter should override"
                                  " this method")

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

    def __numpy_ufunc__(self):
        return np.array(self.value)

    @property
    def value(self):
        return self._eval()

    @property
    def constraint(self):
        return None

    @property
    def bounds(self):
        return None

    @value.setter
    def value(self, v):
        self._value = v

    def dependencies(self):
        dep_list = []
        for _dep in self._deps:
            if isinstance(_dep, Parameter):
                dep_list.append(_dep)
            if isinstance(_dep, (_UnaryOp, _BinaryOp)):
                dep_list.append(_dep.dependencies())
        return dep_list

    def __str__(self):
        """Returns printable representation of a Parameter object."""
        s = ("<Parameter:{name:^15s}, value={value:g}{fixed: ^10}, {bounds}"
             "{constraint}>")

        d = {'name': repr(self.name),
             'value': self.value,
             'fixed': '',
             'constraint': ''}

        if not self.vary and self.constraint is None:
            d['fixed'] = '(fixed)'
        elif self.stderr is not None:
            d['fixed'] = " +/- {0:0.3g}".format(self.stderr)

        d['bounds'] = "bounds={0}".format(str(self.bounds))
        if self.constraint is not None:
            d['constraint'] = ', constraint={}'.format(self.constraint)
        return s.format(**d)


class Constant(BaseParameter):
    def __init__(self, value=0., name=None):
        super(Constant, self).__init__()
        self.name = name
        self.value = value
        self._vary = False

    def __repr__(self):
        return "Constant(value={value}, name={name!r})".format(**self.__dict__)

    def _eval(self):
        return self._value


def possibly_create_parameter(value, name=''):
    """
    If supplied with a Parameter return it. If supplied with float, wrap it in
    a Parameter instance.

    Parameters
    ----------
    value : float or refnx.analysis.Parameter

    Returns
    -------
    parameter : refnx.analysis.Parameter

    """
    if is_parameter(value):
        return value
    else:
        return Parameter(value, name=name)


class Parameter(BaseParameter):
    """
    Class for specifying a variable.

    Parameters
    ----------
    value : float, optional
        Numerical Parameter value.
    name : str, optional
        Name of the parameter.
    bounds: `refnx.analysis.Bounds`, tuple, optional
        Sets the bounds for the parameter. Either supply a
        `refnx.analysis.Bounds` object (or one of its subclasses),
        or a `(lower_bound, upper_bound)` tuple.
    vary : bool, optional
        Whether the Parameter is fixed during a fit.
    constraint : expression, optional
        Python expression used to constrain the value during the fit.
    """

    def __init__(self, value=0., name=None, bounds=None, vary=False,
                 constraint=None):
        """
        Class for specifying a variable.

        Parameters
        ----------
        value : float, optional
            Numerical Parameter value.
        name : str, optional
            Name of the parameter.
        bounds: `refnx.analysis.Bounds`, tuple, optional
            Sets the bounds for the parameter. Either supply a
            `refnx.analysis.Bounds` object (or one of its subclasses),
            or a `(lower_bound, upper_bound)` tuple.
        vary : bool, optional
            Whether the Parameter is fixed during a fit.
        constraint : expression, optional
            Python expression used to constrain the value during the fit.
        """
        super(Parameter, self).__init__()

        self.name = name

        # set bounds before setting value because the value may not be valid
        # for the bounds
        self._bounds = Interval()
        if bounds is not None:
            self.bounds = bounds

        self.value = value
        self._vary = vary

        self._constraint = None
        self.constraint = constraint

    def __repr__(self):
        # repr does not include stderr because that can't be used to
        # create a Parameter
        d = {'kls': self.__class__.__name__, 'value': float(self.value),
             'name': self.name, 'vary': self.vary, 'bounds': self._bounds,
             'constraint': self._constraint}
        return ("{kls}(value={value},"
                " name={name!r}, vary={vary!r},"
                " bounds={bounds!r},"
                " constraint={constraint!r})".format(**d))

    def logp(self, pval=None):
        """
        Calculate the log probability of the parameter

        Returns
        -------
        prob : float
            log probability of the parameter
        """
        if pval is not None:
            val = pval
        else:
            val = self.value

        if self.bounds is not None:
            return self.bounds.logp(val)
        else:
            return 0.

    @property
    def value(self):
        """
        The numeric value of the :class:`Parameter`
        """
        if self._constraint is not None:
            retval = self._constraint._eval()
        else:
            retval = self._value

        return retval

    @value.setter
    def value(self, v):
        value = float(v)

        self._value = value

    def _eval(self):
        if self._constraint is not None:
            return self._constraint._eval()
        else:
            return self._value

    @property
    def bounds(self):
        """
        The bounds placed on this :class:`Parameter`.
        """
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if isinstance(bounds, Bounds):
            self._bounds = bounds
        elif bounds is None:
            self._bounds = Interval()
        elif hasattr(bounds, '__len__') and len(bounds) == 2:
            self.range(*bounds)
        else:
            rv = PDF(bounds)
            self._bounds = rv

    def valid(self, val):
        """
        The truth of whether a value would satisfy the bounds for this
        parameter.

        Parameters
        ----------
        val : float
            A proposed value

        Returns
        -------
        valid : bool
            `np.isfinite(Parameter.logp(val))`
        """
        return self.bounds.valid(val)

    def range(self, lower, upper):
        """
        Sets the lower and upper limits on the Parameter

        Parameters
        ----------
        lower : float
            lower bound
        upper : float
            upper bound

        Returns
        -------
        None

        """
        self.bounds = Interval(lower, upper)

    @property
    def vary(self):
        """
        Whether this :class:`Parameter` is allowed to vary
        """
        return self._vary

    @vary.setter
    def vary(self, vary):
        if self.constraint is not None:
            raise RuntimeError("cannot vary a Parameter which is constrained")
        else:
            self._vary = vary

    @property
    def constraint(self):
        return self._constraint

    @constraint.setter
    def constraint(self, expr):
        if expr is None:
            value = self.value
            self._constraint = None
            self.value = value
            return
        _expr = asMagicNumber(expr)
        if id(self) in [id(dep) for dep in flatten(_expr.dependencies())]:
            raise ValueError("Your constraint contains a circular dependency")
        self._constraint = _expr
        for _dep in flatten(_expr.dependencies()):
            self._deps.append(_dep)
        self._vary = False

    def setp(self, value=None, vary=None, bounds=None, constraint=None):
        """
        Set several attributes of the parameter at once.

        Parameters
        ----------
        value : float, optional
            Numerical Parameter value.
        vary : bool, optional
            Whether the Parameter is fixed during a fit.
        bounds: `refnx.analysis.Bounds`, tuple, optional
            Sets the bounds for the parameter. Either supply a
            `refnx.analysis.Bounds` object (or one of its subclasses),
            or a `(lower_bound, upper_bound)` tuple.
        constraint : expression, optional
            Python expression used to constrain the value during the fit.
        """
        if value is not None:
            self.value = value
        if vary is not None:
            self.vary = vary
        if bounds is not None:
            self.bounds = bounds
        if constraint is not None:
            self.constraint = constraint


class _BinaryOp(BaseParameter):
    def __init__(self, op1, op2, operation):
        super(_BinaryOp, self).__init__()
        self.op1 = op1
        self.op2 = op2
        self.opn = operation
        self._deps = []
        self._deps.append(op1)
        self._deps.append(op2)

    def _eval(self):
        return self.opn(self.op1._eval(), self.op2._eval())


class _UnaryOp(BaseParameter):
    def __init__(self, op1, operation):
        super(_UnaryOp, self).__init__()
        self.op1 = op1
        self.opn = operation
        self._deps = []
        self._deps.append(op1)

    def _eval(self):
        return self.opn(self.op1._eval())


def is_parameter(x):
    """Test for Parameter-ness."""
    return isinstance(x, Parameter)


def is_parameters(x):
    """Test for Parameter-ness."""
    return isinstance(x, Parameters)


def _constraint_tree_helper(expr):
    t = []
    if isinstance(expr, Parameter):
        return expr
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, _BinaryOp):
        t.append([_constraint_tree_helper(expr.op1),
                  _constraint_tree_helper(expr.op2),
                  expr.opn])
    if isinstance(expr, _UnaryOp):
        t.append([_constraint_tree_helper(expr.op1),
                  expr.opn])
    return t


def constraint_tree(expr):
    """
    builds a mathematical tree of a constraint expression
    this can be fed into build_constraint_from_tree to
    reconstitute a constraint
    """
    if isinstance(expr, Parameter):
        return [expr]
    if isinstance(expr, Constant):
        return [expr]
    return list(flatten(_constraint_tree_helper(expr)))


def build_constraint_from_tree(tree):
    """
    A calculator for a constraint tree. It's essentially a reverse
    Polish notation calculator.
    """
    v = []
    for t in tree:
        if callable(t):
            if t in binary:
                o1 = v.pop()
                o2 = v.pop()
                v.append(t(o1, o2))
            elif t in unary:
                o1 = v.pop()
                v.append(t(o1))
        else:
            v.append(t)

    return v[0]
