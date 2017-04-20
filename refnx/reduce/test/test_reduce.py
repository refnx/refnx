import unittest
import os
import os.path

from refnx.reduce import reduce_stitch, PlatypusReduce
from numpy.testing import (assert_equal, assert_allclose, assert_)
import xml.etree.ElementTree as ET
from refnx._lib import TemporaryDirectory


class TestReduce(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(__file__)
        self.path = path

        self.cwd = os.getcwd()
        self.tmpdir = TemporaryDirectory()
        os.chdir(self.tmpdir.name)
        return 0

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    def test_smoke(self):
        # a quick smoke test to check that the reduction can occur
        a, fname = reduce_stitch([708, 709, 710], [711, 711, 711],
                                 data_folder=self.path, rebin_percent=2)
        a.save('test1.dat')

    def test_reduction_method(self):
        # a quick smoke test to check that the reduction can occur
        a = PlatypusReduce('PLP0000711.nx.hdf', data_folder=self.path,
                           rebin_percent=4)

        # try reduction with the reduce method
        a.reduce('PLP0000708.nx.hdf', data_folder=self.path, rebin_percent=4)

        # try reduction with the __call__ method
        a('PLP0000708.nx.hdf', data_folder=self.path, rebin_percent=4)

        # this should also have saved a couple of files in the current
        # directory
        assert_(os.path.isfile('./PLP0000708_0.dat'))
        assert_(os.path.isfile('./PLP0000708_0.xml'))

        # try writing offspecular data
        a.write_offspecular('offspec.xml', 0)

    def test_event_reduction(self):
        # check that eventmode reduction can occur, and that there are the
        # correct number of datasets produced.
        a = PlatypusReduce(
            os.path.join(self.path, 'PLP0011613.nx.hdf'),
            reflect=os.path.join(self.path, 'PLP0011641.nx.hdf'),
            integrate=0, rebin_percent=2,
            eventmode=[0, 900, 1800])
        assert_equal(a.ydata.shape[0], 2)

        # check that the resolutions are pretty much the same
        assert_allclose(a.xdata_sd[0] / a.xdata[0],
                        a.xdata_sd[1] / a.xdata[1],
                        atol=0.001)

        # check that the (right?) timestamps are written into the datafile
        tree = ET.parse(os.path.join(os.getcwd(), 'PLP0011641_1.xml'))
        tree.find('.//REFentry').attrib['time']
        # TODO, timestamp is created in the local time stamp of the testing
        # machine. The following works if reduced with a computer in Australian
        # EST.
        # assert_(t == '2012-01-20T22:05:32')

        # what happens if you have too many frame bins
        a = PlatypusReduce(
            os.path.join(self.path, 'PLP0011613.nx.hdf'),
            reflect=os.path.join(self.path, 'PLP0011641.nx.hdf'),
            integrate=0, rebin_percent=2,
            eventmode=[0, 25200, 27000, 30000])
        assert_equal(a.ydata.shape[0], 1)


if __name__ == '__main__':
    unittest.main()
