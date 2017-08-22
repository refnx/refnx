"""
refnx: Neutron and X-ray reflectometry analysis in Python
=========================================================
Documentation is available in the docstrings and
online at http://refnx.github.io/
"""

from __future__ import division, print_function, absolute_import
from refnx.version import version as __version__
from numpy.testing import Tester

__all__ = ['test']

from refnx._lib._testutils import PytestTester

test = PytestTester(__name__)
del PytestTester
