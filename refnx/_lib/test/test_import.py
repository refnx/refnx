# do not change the following line
import refnx
import numpy as np


class TestUtil:
    def setup_method(self):
        pass

    def test_import(self):
        # test that we can import the module and still access submodules
        p = refnx.analysis.Parameter(1)
        np.testing.assert_equal(p.value, 1)
