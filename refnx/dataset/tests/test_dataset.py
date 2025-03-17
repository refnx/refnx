import io
from pathlib import Path
import os

from io import BytesIO, StringIO
import pytest

from refnx.dataset import ReflectDataset, Data1D, load_data, OrsoDataset
from refnx.dataset.data1d import _data1D_to_hdf, _hdf_to_data1d
from refnx.dataset.reflectdataset import load_orso
from refnx._lib import possibly_open_file
import numpy as np
from numpy.testing import assert_equal


class TestReflectDataset:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.pth = Path(__file__).absolute().parent

        data = ReflectDataset()

        x1 = np.linspace(0, 10, 115)
        y1 = 2 * x1
        e1 = np.ones_like(x1)
        dx1 = np.ones_like(x1)
        data.add_data((x1, y1, e1, dx1))
        self.data = data

        self.cwd = Path.cwd()
        self.tmp_path = tmp_path
        os.chdir(self.tmp_path)

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_load(self):
        # load dataset from XML, via file handle
        dataset = ReflectDataset()
        with open(self.pth / "c_PLP0000708.xml") as f:
            dataset.load(f)

        assert_equal(len(dataset), 90)
        assert_equal(90, np.size(dataset.x))

        # load dataset from XML, via string
        dataset = ReflectDataset()
        dataset.load(self.pth / "c_PLP0000708.xml")

        assert_equal(len(dataset), 90)
        assert_equal(90, np.size(dataset.x))

        # load dataset from XML, via string
        dataset = ReflectDataset(self.pth / "c_PLP0000708.xml")

        assert_equal(len(dataset), 90)
        assert_equal(90, np.size(dataset.x))

        # load dataset from .dat, via file handle
        dataset1 = ReflectDataset()
        with open(self.pth / "c_PLP0000708.dat") as f:
            dataset1.load(f)

        assert_equal(len(dataset1), 90)
        assert_equal(90, np.size(dataset1.x))

        # load dataset from .dat, via string
        dataset2 = ReflectDataset()
        dataset2.load(self.pth / "c_PLP0000708.dat")

        assert_equal(len(dataset2), 90)
        assert_equal(90, np.size(dataset2.x))

    def test_dot(self):
        # test with file formats from http://www.reflectometry.net/refdata.htm
        dataset1 = ReflectDataset()
        with open(self.pth / "dot.dat") as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 31)
        assert dataset1.y_err is not None
        assert dataset1.x_err is None

        with open(self.pth / "dot.aft") as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 36)
        assert dataset1.y_err is not None

        with open(self.pth / "dot.refl") as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 59)
        assert dataset1.y_err is not None

        with open(self.pth / "dot.out") as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 1122)
        assert dataset1.y_err is not None
        assert dataset1.x_err is not None

        with open(self.pth / "dot.mft") as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 213)
        assert dataset1.y_err is not None
        assert dataset1.x_err is not None

    def test_load_dat_with_header(self):
        # check that the file load works with a header
        a = ReflectDataset(self.pth / "c_PLP0000708.dat")
        b = ReflectDataset(self.pth / "c_PLP0000708_header.dat")
        c = ReflectDataset(self.pth / "c_PLP0000708_header2.dat")
        assert_equal(len(a), len(b))
        assert_equal(len(a), len(c))

    def test_GH236(self):
        assert (self.pth / "c_PLP0033831.txt").exists()
        a = ReflectDataset(self.pth / "c_PLP0033831.txt")
        assert_equal(len(a), 166)

    def test_loading_junk(self):
        # if you can't load anything from a datafile then you should get a
        # RuntimeError raised.
        from pytest import raises

        with raises(RuntimeError):
            ReflectDataset(self.pth / "../__init__.py")

    def test_bytes(self):
        with open(self.pth / "c_PLP0033831.txt", "rb") as f:
            data = f.read()
        s = data.decode().replace("\r", "\n")
        b = io.StringIO(s)
        d = Data1D()
        d.load(b)

    def test_construction(self):
        # test we can construct a dataset directly from a file.
        pth = self.pth / "c_PLP0000708.xml"

        ReflectDataset(pth)

        with open(self.pth / "c_PLP0000708.xml") as f:
            ReflectDataset(f)

        ReflectDataset(str(self.pth / "d_a.txt"))

    def test_init_with_data1d(self):
        # test we can construct a dataset from a dataset (i.e. a copy)
        dataset = Data1D(self.data)
        assert_equal(dataset.y, self.data.y)

    def test_add_data(self):
        # test we can add data to the dataset

        # 2 columns
        data = Data1D()

        x1 = np.linspace(0, 10, 5)
        y1 = 2 * x1
        data.add_data((x1, y1))

        x2 = np.linspace(7, 20, 10)
        y2 = 2 * x2 + 2
        data.add_data((x2, y2), requires_splice=True)

        assert len(data) == 13

        # 3 columns
        data = Data1D()

        x1 = np.linspace(0, 10, 5)
        y1 = 2 * x1
        e1 = np.ones_like(x1)
        data.add_data((x1, y1, e1))

        x2 = np.linspace(7, 20, 10)
        y2 = 2 * x2 + 2
        e2 = np.ones_like(x2)
        data.add_data((x2, y2, e2), requires_splice=True)

        assert len(data) == 13

        # 4 columns
        data = Data1D()

        x1 = np.linspace(0, 10, 5)
        y1 = 2 * x1
        e1 = np.ones_like(x1)
        dx1 = np.ones_like(x1)
        data.add_data((x1, y1, e1, dx1))

        x2 = np.linspace(7, 20, 10)
        y2 = 2 * x2 + 2
        e2 = np.ones_like(x2)
        dx2 = np.ones_like(x2)

        data.add_data((x2, y2, e2, dx2), requires_splice=True)

        assert len(data) == 13

        # test addition of datasets.
        x1 = np.linspace(0, 10, 5)
        y1 = 2 * x1
        e1 = np.ones_like(x1)
        dx1 = np.ones_like(x1)
        data = Data1D((x1, y1, e1, dx1))

        x2 = np.linspace(7, 20, 10)
        y2 = 2 * x2 + 2
        e2 = np.ones_like(x2)
        dx2 = np.ones_like(x2)
        data2 = Data1D((x2, y2, e2, dx2))

        data3 = data + data2
        assert len(data3) == 13

        # test iadd of datasets
        data += data2
        assert len(data) == 13

    def test_add_non_overlapping(self):
        # check that splicing of non-overlapping datasets raises ValueError
        x1 = np.linspace(0, 10, 11)
        y1 = 2 * x1
        data = Data1D((x1, y1))

        x2 = np.linspace(11, 20, 10)
        y2 = 2 * x2 + 2

        from pytest import raises

        with raises(ValueError):
            data.add_data((x2, y2), requires_splice=True)

    def test_save_xml(self):
        self.data.save_xml("test.xml")
        with open("test.xml", "wb") as f:
            self.data.save_xml(f)

    def test_data_setter(self):
        # check that setting a Data1D's data from a tuple works correctly
        # this is the approach used in Data1D.synthesise.
        new_dataset = Data1D()
        new_dataset.data = self.data.data
        assert_equal(new_dataset.y_err, self.data.y_err)
        assert_equal(new_dataset.x_err, self.data.x_err)
        assert_equal(new_dataset.y, self.data.y)
        assert_equal(new_dataset.x, self.data.x)
        assert_equal(new_dataset.weighted, self.data.weighted)

    def test_synthesise(self):
        # add gaussian noise to a Data1D.y
        new_dataset = self.data.synthesise(random_state=1)

        # y-array should not be the same, so don't test that
        assert_equal(new_dataset.y_err, self.data.y_err)
        assert_equal(new_dataset.x_err, self.data.x_err)
        assert_equal(new_dataset.x, self.data.x)
        assert_equal(new_dataset.weighted, self.data.weighted)

        # synthesis should be repeatable with provision of a seed
        new_dataset2 = self.data.synthesise(random_state=1)
        assert_equal(new_dataset.y, new_dataset2.y)

    def test_mask(self):
        # if you mask all points there should be none left
        self.data.mask = np.full_like(self.data.y, False, bool)
        assert_equal(len(self.data), 0)

        # try masking a random selection
        rando = np.random.randint(0, 2, self.data._y.size)
        self.data.mask = rando
        assert_equal(len(self.data), np.count_nonzero(rando))

        # now clear
        self.data.mask = None
        assert_equal(len(self.data), self.data._y.size)

    def test_repr(self):
        a = Data1D(self.pth / "c_PLP0033831.txt")
        b = eval(repr(a))
        assert_equal(len(a), 166)
        assert_equal(len(b), 166)

        # load dataset from XML, via string
        a = ReflectDataset(str(self.pth / "c_PLP0000708.xml"))
        b = eval(repr(a))
        assert_equal(len(b), len(a))
        assert_equal(len(b), len(a))


class TestData1D:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.pth = Path(__file__).absolute().parent
        self.cwd = Path.cwd()
        self.tmp_path = tmp_path
        os.chdir(self.tmp_path)

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_2D_data(self):
        # despite the class being called Data1D it should be possible to
        # store ND data in it

        # try with multidimensional x data
        x = np.linspace(0.01, 0.5, 100).reshape(50, 2)
        y = np.arange(50, dtype=float)

        data = Data1D((x, y))
        assert len(data) == 50

        assert_equal(data.x, x)
        assert_equal(data.y, y)

        mask = np.ones_like(y, bool)
        mask[1] = False
        data.mask = mask
        assert len(data) == 49
        assert_equal(data.y, np.r_[y[0], y[2:]])
        assert_equal(data.x, np.concatenate((x[0][np.newaxis, :], x[2:])))

        # try with storing multiple channels for y as well
        # this might happen with something like ellipsometry data where you'd
        # have (delta, xsi) as the y data, and (lambda, aoi) as the x data
        # NOTE THAT MASKING WON'T WORK, FOR ND YDATA. This is because the class
        # is using fancy indexing with booleans to remove entries that are
        # unwanted. Could mark as NaN instead?
        x = np.linspace(0.01, 0.5, 50).reshape(50)
        y = np.arange(100, dtype=float).reshape(50, 2)

        data = Data1D((x, y))
        assert len(data) == 100

        assert_equal(data.x, x)
        assert_equal(data.y, y)

        x = np.linspace(0.01, 0.5, 100).reshape(50, 2)

        data = Data1D((x, y))
        assert len(data) == 100

        assert_equal(data.x, x)
        assert_equal(data.y, y)

    def test_rudimentry_HDF(self):
        rng = np.random.default_rng()
        x = np.linspace(0, 100, 101)
        y = rng.random(101)
        dy = rng.random(101)
        dx = rng.random((101, 2, 40))

        data = Data1D((x, y, dy, dx))

        f = BytesIO()
        _data1D_to_hdf(f, data)
        _data = _hdf_to_data1d(f)
        assert len(_data) == 101
        assert _data.x_err.shape == (101, 2, 40)

    def test_load_data(self):
        # test the load_data function by trying to load all the files in the
        # test directory
        fs = Path.cwd().glob("*.*")
        fs = [f for f in fs if not f.endswith(".py")]
        fs = [f for f in fs if not f.startswith("coef_")]

        for f in fs:
            load_data(f)
            load_data(Path(f))


class TestOrtDataset:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.pth = Path(__file__).absolute().parent
        self.cwd = Path.cwd()
        self.tmp_path = tmp_path
        os.chdir(self.tmp_path)

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_ort_load(self):
        f = self.pth / "ORSO_data.ort"
        assert f.exists()
        try:
            load_orso(str(f))
            load_orso(f)
        except ImportError:
            # load_orso had problems on Python 3.10, so bypass the test
            return

        d = load_data(f)
        assert len(d) == 2
        assert isinstance(d, OrsoDataset)
        d.refresh()

    def test_orb_load(self):
        f = self.pth / "test.orb"
        assert f.exists()

        try:
            load_orso(str(f))
            load_orso(f)
        except ImportError:
            # load_orso had problems on Python 3.10, so bypass the test
            return

        # with possibly_open_file(f, 'rb') as f:
        d = load_data(f)

        assert isinstance(d, OrsoDataset)
        d.refresh()
