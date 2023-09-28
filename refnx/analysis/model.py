from refnx._lib.util import getargspec


def fitfunc(f):
    """
    A decorator that can be used to say if something is a fitfunc.

    """
    f.fitfuncwraps = True
    return f


class Model:
    """
    Calculates a generative model (dependent variable), given parameters
    and independent variables.

    Parameters
    ----------
    parameters : array or refnx.analysis.Parameters
        Parameters to calculate the model with
    fitfunc : callable, optional
        A function that calculates the generative model. Should have the
        signature ``fitfunc(x, parameters, *fcn_args, **fcn__kwds)`` where `x`
        is an array-like specifying the independent variables, and
        `parameters` are the parameters required to calculate the model.
        `fcn_args` and `fcn_kwds` can be used to supply additional arguments to
        to `fitfunc`.
    fcn_args : sequence, optional
        Supplies extra arguments to `fitfunc`
    fcn_kwds : dict, optional
        Supplies keyword arguments to `fitfunc`

    Notes
    -----
    It is not necessary to supply `fitfunc` to create a `Model` *iff* you are
    inheriting `Model` and are also overriding `Model.model`.

    """

    def __init__(self, parameters, fitfunc=None, fcn_args=(), fcn_kwds=None):
        self._parameters = parameters

        self._fitfunc = None
        self._fitfunc_has_xerr = False
        self.fitfunc = fitfunc

        self.fcn_args = fcn_args
        self.fcn_kwds = {}
        if fcn_kwds is not None:
            self.fcn_kwds = fcn_kwds

    def __repr__(self):
        return (
            f"Model({self._parameters!r}, fitfunc={self._fitfunc!r},"
            f" fcn_args={self.fcn_args!r},"
            f" fcn_kwds={self.fcn_kwds!r})"
        )

    def __call__(self, x, p=None, x_err=None):
        """
        Calculates a generative model(dependent variable), given parameters and
        independent variables.

        Parameters
        ----------
        x : array-like
            Independent variable.
        p : array-like or refnx.analysis.Parameters
            Parameters to supply to the generative function.
        x_err : optional
            Uncertainty in `x`.

        Returns
        -------
        generative : array-like or float

        Notes
        -----
        The interpretation of `x`, `p`, and `x_err` is up to the `fitfunc`
        supplied during construction of this object (or the overridden `model`
        method of this object).

        """
        return self.model(x, p=p, x_err=x_err)

    def model(self, x, p=None, x_err=None):
        """
        Calculates a generative model(dependent variable), given parameters and
        independent variables.

        Parameters
        ----------
        x : array-like
            Independent variable.
        p : array-like or refnx.analysis.Parameters
            Parameters to supply to the generative function.
        x_err : optional
            Uncertainty in `x`.

        Returns
        -------
        generative : array-like or float

        Notes
        -----
        The interpretation of `x`, `p`, and `x_err` is up to the `fitfunc`
        supplied during construction of this object (or the overridden `model`
        method of this object).

        """
        # self.fitfunc or this method has to understand the structure of
        # self.params.
        if self.fitfunc is not None:
            kwds = {}
            kwds.update(self.fcn_kwds)

            if self._fitfunc_has_xerr:
                # fitfunc has resolution
                kwds["x_err"] = x_err

            _params = self._parameters
            if p is not None:
                _params = p

            return self.fitfunc(x, _params, *self.fcn_args, **kwds)
        else:
            raise RuntimeError(
                "Overide Model.model() or provide a fitfunc to"
                " the constructor"
            )

    def logp(self):
        r"""
        The model can add additional terms to it's log-probability. However,
        it should _not_ include logp from any of the parameters. That is
        calculated by `Objective.logp`.

        """
        return 0

    @property
    def fitfunc(self):
        """
        The fit-function associated with the model
        """
        return self._fitfunc

    @fitfunc.setter
    def fitfunc(self, fitfunc):
        self._fitfunc = fitfunc
        self._fitfunc_has_xerr = False
        if fitfunc is not None and "x_err" in getargspec(fitfunc).args:
            self._fitfunc_has_xerr = True

    @property
    def parameters(self):
        r"""
        The refnx.analysis.Parameters set associated with the model.
        """
        # override this if model adds more parameters than just
        # self._parameters
        return self._parameters
