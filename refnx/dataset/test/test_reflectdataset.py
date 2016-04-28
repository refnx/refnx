import unittest
from refnx.dataset import ReflectDataset, Data1D
import numpy as np
from numpy.testing import assert_equal, assert_
import os.path


path = os.path.dirname(os.path.abspath(__file__))

class TestReflectDataset(unittest.TestCase):

    def setUp(self):
        data = ReflectDataset()

        x1 = np.linspace(0, 10, 5)
        y1 = 2 * x1
        e1 = np.ones_like(x1)
        dx1 = np.ones_like(x1)
        data.add_data((x1, y1, e1, dx1))
        self.data = data

    def test_load(self):
        # test reflectivity calculation with values generated from Motofit
        dataset = ReflectDataset()
        with open(os.path.join(path, 'c_PLP0000708.xml')) as f:
            dataset.load(f)
        
        assert_equal(dataset.npoints, 90)
        assert_equal(90, np.size(dataset.x))
        
        dataset1 = ReflectDataset()
        with open(os.path.join(path, 'c_PLP0000708.dat')) as f:
            dataset1.load(f)
        
        assert_equal(dataset1.npoints, 90)
        assert_equal(90, np.size(dataset1.x))

    def test_construction(self):
        # test we can construct a dataset directly from a file.
        pth = os.path.join(path, 'c_PLP0000708.xml')

        dataset = ReflectDataset(pth)

        with open(os.path.join(path, 'c_PLP0000708.xml')) as f:
            dataset = ReflectDataset(f)

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

        assert_(data.npoints == 13)

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

        assert_(data.npoints == 13)

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

        assert_(data.npoints == 13)

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
        assert_(data3.npoints == 13)

        # test iadd of datasets
        data += data2
        assert_(data.npoints == 13)

    def test_save_xml(self):
        self.data.save_xml('test.xml')
        with open('test.xml', 'wb') as f:
            self.data.save_xml(f)


if __name__ == '__main__':
    unittest.main()