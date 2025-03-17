from importlib import resources
import refnx
import numpy as np


class TestUtil:
    def setup_method(self):
        pass

    def test_import(self):
        # test that we can import the module and still access submodules
        p = refnx.analysis.Parameter(1)
        np.testing.assert_equal(p.value, 1)

    def test_module(self):
        # smoke test to check that refnx has an attribute called analysis
        # and that it can be located using importlib
        resources.files(refnx.analysis)
