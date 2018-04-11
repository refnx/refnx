import os
import os.path
import pytest

from numpy.testing import (assert_equal, assert_allclose, assert_)
import xml.etree.ElementTree as ET

from refnx.reduce import reduce_stitch, PlatypusReduce


class TestReduce(object):

    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir):
        self.pth = os.path.dirname(os.path.abspath(__file__))

        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_smoke(self):
        # a quick smoke test to check that the reduction can occur
        a, fname = reduce_stitch([708, 709, 710], [711, 711, 711],
                                 data_folder=self.pth, rebin_percent=2)
        a.save('test1.dat')

    def test_reduction_method(self):
        # a quick smoke test to check that the reduction can occur
        a = PlatypusReduce('PLP0000711.nx.hdf', data_folder=self.pth)

        # try reduction with the reduce method
        a.reduce('PLP0000708.nx.hdf', data_folder=self.pth, rebin_percent=4)

        # try reduction with the __call__ method
        a('PLP0000708.nx.hdf', data_folder=self.pth, rebin_percent=4)

        # this should also have saved a couple of files in the current
        # directory
        assert_(os.path.isfile('./PLP0000708_0.dat'))
        assert_(os.path.isfile('./PLP0000708_0.xml'))

        # try writing offspecular data
        a.write_offspecular('offspec.xml', 0)

    def test_free_liquids(self):
        # smoke test for free liquids
        a0 = PlatypusReduce('PLP0038418.nx.hdf', data_folder=self.pth)
        a1 = PlatypusReduce('PLP0038417.nx.hdf', data_folder=self.pth)

        # try reduction with the reduce method
        d0, r0 = a0.reduce('PLP0038420.nx.hdf', data_folder=self.pth,
                           rebin_percent=4)
        d1, r1 = a1.reduce('PLP0038421.nx.hdf', data_folder=self.pth,
                           rebin_percent=4)

    def test_event_reduction(self):
        # check that eventmode reduction can occur, and that there are the
        # correct number of datasets produced.
        a = PlatypusReduce(
            os.path.join(self.pth, 'PLP0011613.nx.hdf'))

        a.reduce(os.path.join(self.pth, 'PLP0011641.nx.hdf'),
                 integrate=0, rebin_percent=2, eventmode=[0, 900, 1800])

        assert_equal(a.y.shape[0], 2)

        # check that two datasets are written out.
        assert_(os.path.isfile('PLP0011641_0.dat'))
        assert_(os.path.isfile('PLP0011641_1.dat'))

        # check that the resolutions are pretty much the same
        assert_allclose(a.x_err[0] / a.x[0],
                        a.x_err[1] / a.x[1],
                        atol=0.001)

        # check that the (right?) timestamps are written into the datafile
        tree = ET.parse(os.path.join(os.getcwd(), 'PLP0011641_1.xml'))
        tree.find('.//REFentry').attrib['time']
        # TODO, timestamp is created in the local time stamp of the testing
        # machine. The following works if reduced with a computer in Australian
        # EST.
        # assert_(t == '2012-01-20T22:05:32')

        # # what happens if you have too many frame bins
        # # you should get the same number of fr
        # a = PlatypusReduce(
        #     os.path.join(self.pth, 'PLP0011613.nx.hdf'),
        #     reflect=os.path.join(self.pth, 'PLP0011641.nx.hdf'),
        #     integrate=0, rebin_percent=2,
        #     eventmode=[0, 25200, 27000, 30000])
        # assert_equal(a.ydata.shape[0], 3)
