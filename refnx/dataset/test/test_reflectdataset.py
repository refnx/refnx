import os.path

import pytest

from refnx.dataset import ReflectDataset, Data1D
import numpy as np
from numpy.testing import assert_equal, assert_


class TestReflectDataset:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        self.pth = os.path.dirname(os.path.abspath(__file__))

        data = ReflectDataset()

        x1 = np.linspace(0, 10, 115)
        y1 = 2 * x1
        e1 = np.ones_like(x1)
        dx1 = np.ones_like(x1)
        data.add_data((x1, y1, e1, dx1))
        self.data = data

        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_load(self):
        # load dataset from XML, via file handle
        dataset = ReflectDataset()
        with open(os.path.join(self.pth, "c_PLP0000708.xml")) as f:
            dataset.load(f)

        assert_equal(len(dataset), 90)
        assert_equal(90, np.size(dataset.x))

        # load dataset from XML, via string
        dataset = ReflectDataset()
        dataset.load(os.path.join(self.pth, "c_PLP0000708.xml"))

        assert_equal(len(dataset), 90)
        assert_equal(90, np.size(dataset.x))

        # load dataset from XML, via string
        dataset = ReflectDataset(os.path.join(self.pth, "c_PLP0000708.xml"))

        assert_equal(len(dataset), 90)
        assert_equal(90, np.size(dataset.x))

        # load dataset from .dat, via file handle
        dataset1 = ReflectDataset()
        with open(os.path.join(self.pth, "c_PLP0000708.dat")) as f:
            dataset1.load(f)

        assert_equal(len(dataset1), 90)
        assert_equal(90, np.size(dataset1.x))

        # load dataset from .dat, via string
        dataset2 = ReflectDataset()
        dataset2.load(os.path.join(self.pth, "c_PLP0000708.dat"))

        assert_equal(len(dataset2), 90)
        assert_equal(90, np.size(dataset2.x))

    def test_dot(self):
        # test with file formats from http://www.reflectometry.net/refdata.htm
        dataset1 = ReflectDataset()
        with open(os.path.join(self.pth, "dot.dat")) as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 31)
        assert dataset1.y_err is not None
        assert dataset1.x_err is None

        with open(os.path.join(self.pth, "dot.aft")) as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 36)
        assert dataset1.y_err is not None

        with open(os.path.join(self.pth, "dot.refl")) as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 59)
        assert dataset1.y_err is not None

        with open(os.path.join(self.pth, "dot.out")) as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 1122)
        assert dataset1.y_err is not None
        assert dataset1.x_err is not None

        with open(os.path.join(self.pth, "dot.mft")) as f:
            dataset1.load(f)
        assert_equal(len(dataset1), 213)
        assert dataset1.y_err is not None
        assert dataset1.x_err is not None

    def test_load_dat_with_header(self):
        # check that the file load works with a header
        a = ReflectDataset(os.path.join(self.pth, "c_PLP0000708.dat"))
        b = ReflectDataset(os.path.join(self.pth, "c_PLP0000708_header.dat"))
        c = ReflectDataset(os.path.join(self.pth, "c_PLP0000708_header2.dat"))
        assert_equal(len(a), len(b))
        assert_equal(len(a), len(c))

    def test_GH236(self):
        a = ReflectDataset(os.path.join(self.pth, "c_PLP0033831.txt"))
        assert_equal(len(a), 166)

    def test_loading_junk(self):
        # if you can't load anything from a datafile then you should get a
        # RuntimeError raised.
        from pytest import raises

        with raises(RuntimeError):
            ReflectDataset(os.path.join(self.pth, "../__init__.py"))

    def test_construction(self):
        # test we can construct a dataset directly from a file.
        pth = os.path.join(self.pth, "c_PLP0000708.xml")

        ReflectDataset(pth)

        with open(os.path.join(self.pth, "c_PLP0000708.xml")) as f:
            ReflectDataset(f)

        ReflectDataset(os.path.join(self.pth, "d_a.txt"))

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
        a = Data1D(os.path.join(self.pth, "c_PLP0033831.txt"))
        b = eval(repr(a))
        assert_equal(len(a), 166)
        assert_equal(len(b), 166)

        # load dataset from XML, via string
        a = ReflectDataset(os.path.join(self.pth, "c_PLP0000708.xml"))
        b = eval(repr(a))
        assert_equal(len(b), len(a))
        assert_equal(len(b), len(a))


class TestData1D:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        self.pth = os.path.dirname(os.path.abspath(__file__))
        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)

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
