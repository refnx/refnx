"""
refnx: Neutron and X-ray reflectometry analysis in Python
=========================================================
Documentation is available in the docstrings and
online at https://readthedocs.org/projects/refnx/
"""

try:
    from refnx.version import version as __version__
except ImportError:
    __version__ = "version string not created yet"

from refnx._lib._testutils import PytestTester


test = PytestTester(__name__)
del PytestTester
