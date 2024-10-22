import os

from refnx.reflect.reflect_model import (
    ReflectModel,
    ReflectModelTL,
    reflectivity,
    MixedReflectModel,
    FresnelTransform,
    choose_dq_type,
    use_reflect_backend,
    available_backends,
    abeles,
    SpinChannel,
)
from refnx.reflect.structure import (
    Structure,
    SLD,
    Slab,
    Component,
    sld_profile,
    Stack,
    MaterialSLD,
    MixedSlab,
    create_occupancy,
    possibly_create_scatterer,
)
from refnx.reflect.interface import (
    Erf,
    Interface,
    Linear,
    Exponential,
    Tanh,
    Sinusoidal,
    Step,
)
from refnx.reflect.spline import Spline
from refnx.reflect._lipid import LipidLeaflet, LipidLeafletGuest
from refnx.reflect._functional_form import FunctionalForm
from refnx._lib._testutils import PytestTester
from refnx.reflect._app import gui, main


# On OSX, we can get a runtime error due to multiple OpenMP libraries loaded
# simultaneously. This can happen for instance when calling BLAS inside a
# prange. Setting the following environment variable allows multiple OpenMP
# libraries to be loaded.
# OMP
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

try:
    import numba

    # set the threading layer before any parallel target compilation
    numba.config.THREADING_LAYER = "forksafe"
except ImportError:
    pass

try:
    from refnx.reflect._interactive_modeller import Motofit
except ImportError:

    class Motofit:
        def __init__(self):
            raise RuntimeError(
                "To run Motofit you need to install"
                " IPython, ipywidgets, traitlets,"
                " matplotlib"
            )

        def __call__(self, dummy):
            pass


test = PytestTester(__name__)
del PytestTester


__all__ = [s for s in dir() if not s.startswith("_")]
