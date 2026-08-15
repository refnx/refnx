import os

from refnx._lib._testutils import PytestTester
from refnx.reflect._app import gui, main
from refnx.reflect._functional_form import FunctionalForm
from refnx.reflect._lipid import LipidLeaflet, LipidLeafletGuest
from refnx.reflect._polarised_reflect_model import (
    PolarisedReflectModel,
    pnr_data_and_generative,
)
from refnx.reflect.interface import (
    Erf,
    Exponential,
    Interface,
    Linear,
    Sinusoidal,
    Step,
    Tanh,
)
from refnx.reflect.reflect_model import (
    Footprint,
    FresnelTransform,
    MixedReflectModel,
    ReflectModel,
    ReflectModelTL,
    SpinChannel,
    abeles,
    available_backends,
    choose_dq_type,
    reflectivity,
    use_reflect_backend,
)
from refnx.reflect.spline import Spline
from refnx.reflect.structure import (
    SLD,
    Component,
    MagneticSlab,
    MaterialSLD,
    MixedSlab,
    Slab,
    Stack,
    Structure,
    create_occupancy,
    possibly_create_scatterer,
    sld_profile,
)

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
                " IPython, ipywidgets, traitlets, ipympl, "
                " matplotlib"
            )

        def __call__(self, dummy):
            pass


test = PytestTester(__name__)
del PytestTester


__all__ = [s for s in dir() if not s.startswith("_")]
