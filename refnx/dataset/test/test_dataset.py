import os.path

import pytest
import glob

from refnx.dataset import ReflectDataset, Data1D, load_data, OrsoDataset
import numpy as np
from numpy.testing import assert_equal


class TestReflectDataset:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        self.pth = os.path.dirname(os.path.abspath(__file__))
        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_ort_load(self):
        d = load_data(os.path.join(self.pth, "ORSO_data.ort"))
        assert len(d) == 2
        assert isinstance(d, OrsoDataset)
        d.refresh()

    def test_load_data(self):
        # test the load_data function by trying to load all the files in the
        # test directory
        fs = glob.glob("*.*")
        fs = [f for f in fs if not f.endswith(".py")]
        fs = [f for f in fs if not f.startswith("coef_")]

        for f in fs:
            load_data(f)
