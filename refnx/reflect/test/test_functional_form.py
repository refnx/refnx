import numpy as np
from numpy.testing import assert_allclose

from refnx.analysis import Parameter
from refnx.reflect import FunctionalForm, SLD


class Line:
    def __call__(self, z, extent, left_sld, right_sld, **kwds):
        self.keys = list(kwds.keys())
        self.extent = extent

        grad = (right_sld - left_sld) / extent
        intercept = left_sld
        # we don't calculate the volume fraction of solvent
        return z * grad * kwds["dummy_param"] + intercept, None


def test_functional_form():
    si = SLD(2.07)
    d2o = SLD(6.36)
    p = Parameter(1.0)
    line = Line()

    form = FunctionalForm(100, line, dummy_param=p)
    s = si | form | d2o(0, 3)
    s.sld_profile()

    assert_allclose(line.extent, 100.0)
    assert "dummy_param" in line.keys
