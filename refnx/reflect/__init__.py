from __future__ import division, absolute_import

from refnx.reflect.reflect_model import ReflectModel, reflectivity
from refnx.reflect.structure import Structure, SLD, Slab, Component

import numpy.testing
test = numpy.testing.Tester().test

__all__ = [s for s in dir() if not s.startswith('_')]
