"""
refnx: Neutron and X-ray reflectometry analysis in Python
=========================================================
Documentation is available in the docstrings and
online at https://readthedocs.org/projects/refnx/
"""

import importlib as _importlib

try:
    from refnx.version import version as __version__
except ImportError:
    __version__ = "version string not created yet"

from refnx._lib._testutils import PytestTester

test = PytestTester(__name__)
del PytestTester


submodules = [
    "analysis",
    "dataset",
    "reduce",
    "reflect",
    "util",
]

__all__ = submodules + [
    "test",
    "__version__",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f"refnx.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'refnx' has no attribute '{name}'")
