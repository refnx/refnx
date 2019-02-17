from refnx.reflect.reflect_model import (ReflectModel, reflectivity,
                                         MixedReflectModel)
from refnx.reflect.structure import (Structure, SLD, Slab, Component,
                                     sld_profile, Stack)
from refnx.reflect.spline import Spline
from refnx.reflect._lipid import LipidLeaflet
from refnx._lib._testutils import PytestTester
from refnx.reflect._app import gui, main


try:
    import ipywidgets as _ipywidgets
    import traitlets as _traitlets
    import matplotlib as _matplotlib
    import IPython as _ipython
    from refnx.reflect._interactive_modeller import Motofit
except ImportError:

    class Motofit():
        def __init__(self):
            raise RuntimeError("To run Motofit you need to install"
                               " IPython, ipywidgets, traitlets,"
                               " matplotlib")

        def __call__(self, dummy):
            pass


test = PytestTester(__name__)
del PytestTester


__all__ = [s for s in dir() if not s.startswith('_')]
