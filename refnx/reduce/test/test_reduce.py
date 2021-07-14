import os
from os.path import join as pjoin
import os.path
from refnx.reduce.platypusnexus import calculate_wavelength_bins
from refnx.reduce.reduce import PolarisationEfficiency
import warnings
import pytest

from numpy.testing import assert_equal, assert_allclose
import xml.etree.ElementTree as ET

from refnx.reduce import (
    reduce_stitch,
    PlatypusReduce,
    ReductionOptions,
    SpatzReduce,
    SpinSet,
    PolarisedReduce,
)
from refnx.dataset import ReflectDataset


class TestPlatypusReduce:
    @pytest.mark.usefixtures("no_data_directory")
    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir, data_directory):
        self.pth = pjoin(data_directory, "reduce")

        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_smoke(self):
        # a quick smoke test to check that the reduction can occur
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a, fname = reduce_stitch(
                [708, 709, 710],
                [711, 711, 711],
                data_folder=self.pth,
                reduction_options={"rebin_percent": 2},
            )
            a.save("test1.dat")
            assert os.path.isfile("./test1.dat")

            # reduce_stitch should take a ReductionOptions dict
            opts = ReductionOptions()
            opts["rebin_percent"] = 2

            a2, fname = reduce_stitch(
                [708, 709, 710],
                [711, 711, 711],
                data_folder=self.pth,
                reduction_options=[opts] * 3,
            )
            a2.save("test2.dat")
            assert os.path.isfile("./test2.dat")
            assert_allclose(a.y, a2.y)

    def test_reduction_method(self):

        # a quick smoke test to check that the reduction can occur
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            a = PlatypusReduce("PLP0000711.nx.hdf", data_folder=self.pth)

            # try reduction with the reduce method
            a.reduce(
                "PLP0000708.nx.hdf",
                data_folder=self.pth,
                rebin_percent=4,
            )

            # try reduction with the __call__ method
            a(
                "PLP0000708.nx.hdf",
                data_folder=self.pth,
                rebin_percent=4,
            )

            # this should also have saved a couple of files in the current
            # directory
            assert os.path.isfile("./PLP0000708_0.dat")
            assert os.path.isfile("./PLP0000708_0.xml")

            # can we read the file
            ReflectDataset("./PLP0000708_0.dat")

            # try writing offspecular data
            a.write_offspecular("offspec.xml", 0)

    def test_free_liquids(self):
        # smoke test for free liquids
        a0 = PlatypusReduce("PLP0038418.nx.hdf", data_folder=self.pth)
        a1 = PlatypusReduce("PLP0038417.nx.hdf", data_folder=self.pth)

        # try reduction with the reduce method
        d0, r0 = a0.reduce(
            "PLP0038420.nx.hdf", data_folder=self.pth, rebin_percent=4
        )
        d1, r1 = a1.reduce(
            "PLP0038421.nx.hdf", data_folder=self.pth, rebin_percent=4
        )

    def test_event_reduction(self):
        # check that eventmode reduction can occur, and that there are the
        # correct number of datasets produced.

        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            a = PlatypusReduce(pjoin(self.pth, "PLP0011613.nx.hdf"))

            a.reduce(
                pjoin(self.pth, "PLP0011641.nx.hdf"),
                integrate=0,
                rebin_percent=2,
                eventmode=[0, 900, 1800],
            )

            assert_equal(a.y.shape[0], 2)

            # check that two datasets are written out.
            assert os.path.isfile("PLP0011641_0.dat")
            assert os.path.isfile("PLP0011641_1.dat")

            # check that the resolutions are pretty much the same
            assert_allclose(
                a.x_err[0] / a.x[0], a.x_err[1] / a.x[1], atol=0.001
            )

            # check that the (right?) timestamps are written into the datafile
            tree = ET.parse(pjoin(os.getcwd(), "PLP0011641_1.xml"))
            tree.find(".//REFentry").attrib["time"]
            # TODO, timestamp is created in the local time stamp of the testing
            # machine. The following works if reduced with a computer in
            # Australian EST.
            # assert_(t == '2012-01-20T22:05:32')

            # # what happens if you have too many frame bins
            # # you should get the same number of fr
            # a = PlatypusReduce(
            #     os.path.join(self.pth, 'PLP0011613.nx.hdf'),
            #     reflect=os.path.join(self.pth, 'PLP0011641.nx.hdf'),
            #     integrate=0, rebin_percent=2,
            #     eventmode=[0, 25200, 27000, 30000])
            # assert_equal(a.ydata.shape[0], 3)


class TestSpatzReduce:
    @pytest.mark.usefixtures("no_data_directory")
    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir, data_directory):
        self.pth = pjoin(data_directory, "reduce")

        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_smoke(self):
        # a quick smoke test to check that the reduction can occur
        # warnings filter for pixel size
        a, fname = reduce_stitch(
            [660, 661],
            [658, 659],
            data_folder=self.pth,
            prefix="SPZ",
            reduction_options={"rebin_percent": 2},
        )
        a.save("test1.dat")
        assert os.path.isfile("./test1.dat")

        # reduce_stitch should take a list of ReductionOptions dict,
        # separate dicts are used for different angles
        opts = ReductionOptions()
        opts["rebin_percent"] = 2

        a2, fname = reduce_stitch(
            [660, 661],
            [658, 659],
            data_folder=self.pth,
            prefix="SPZ",
            reduction_options=[opts] * 2,
        )
        a2.save("test2.dat")
        assert os.path.isfile("./test2.dat")
        assert_allclose(a.y, a2.y)

    def test_reduction_method(self):
        # a quick smoke test to check that the reduction can occur
        a = SpatzReduce("SPZ0000658.nx.hdf", data_folder=self.pth)

        # try reduction with the reduce method
        a.reduce(
            "SPZ0000660.nx.hdf",
            data_folder=self.pth,
            rebin_percent=4,
        )

        # try reduction with the __call__ method
        a(
            "SPZ0000660.nx.hdf",
            data_folder=self.pth,
            rebin_percent=4,
        )

        # this should also have saved a couple of files in the current
        # directory
        assert os.path.isfile("./SPZ0000660_0.dat")
        assert os.path.isfile("./SPZ0000660_0.xml")

        # try writing offspecular data
        a.write_offspecular("offspec.xml", 0)


class TestPolarisedReduce:
    @pytest.mark.usefixtures("no_data_directory")
    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir, data_directory):
        self.pth = pjoin(data_directory, "reduce", "PNR_files")

        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_polarised_reduction_method_4sc(self):
        # a quick smoke test to check that the reduction can occur
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            spinset_rb = SpinSet(
                down_down="PLP0012785.nx.hdf",
                up_up="PLP0012787.nx.hdf",
                up_down="PLP0012786.nx.hdf",
                down_up="PLP0012788.nx.hdf",
                data_folder=self.pth,
            )
            spinset_db = SpinSet(
                down_down="PLP0012793.nx.hdf",
                up_up="PLP0012795.nx.hdf",
                up_down="PLP0012794.nx.hdf",
                down_up="PLP0012796.nx.hdf",
                data_folder=self.pth,
            )
            a = PolarisedReduce(spinset_db)

            # try reduction with the reduce method
            a.reduce(
                spinset_rb,
                data_folder=self.pth,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=4,
            )
            # try reduction with the __call__ method
            a(
                spinset_rb,
                data_folder=self.pth,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=4,
            )

            # this should also have saved a couple of files in the current
            # directory
            assert os.path.isfile("./PLP0012785_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012786_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012787_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012788_0_PolCorr.dat")

            for reducer in a.reducers.values():
                with pytest.raises(AssertionError):
                    assert_equal(
                        reducer.reflected_beam.m_spec,
                        reducer.reflected_beam.m_spec_polcorr,
                    )

    def test_polarised_reduction_method_3sc(self):
        # a quick smoke test to check that the reduction can occur
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            spinset_rb = SpinSet(
                down_down="PLP0012785.nx.hdf",
                up_up="PLP0012787.nx.hdf",
                up_down="PLP0012786.nx.hdf",
                data_folder=self.pth,
            )
            spinset_db = SpinSet(
                down_down="PLP0012793.nx.hdf",
                up_up="PLP0012795.nx.hdf",
                up_down="PLP0012794.nx.hdf",
                data_folder=self.pth,
            )
            a = PolarisedReduce(spinset_db)

            # try reduction with the reduce method
            a.reduce(
                spinset_rb,
                data_folder=self.pth,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=4,
            )

            # try reduction with the __call__ method
            a(
                spinset_rb,
                data_folder=self.pth,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=4,
            )

            # this should also have saved a couple of files in the current
            # directory
            assert os.path.isfile("./PLP0012785_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012786_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012787_0_PolCorr.dat")

    def test_polarised_reduction_method_2sc(self):
        # a quick smoke test to check that the reduction can occur
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            spinset_rb = SpinSet(
                down_down="PLP0012785.nx.hdf",
                up_up="PLP0012787.nx.hdf",
                data_folder=self.pth,
            )
            spinset_db = SpinSet(
                down_down="PLP0012793.nx.hdf",
                up_up="PLP0012795.nx.hdf",
                data_folder=self.pth,
            )
            a = PolarisedReduce(spinset_db)

            # try reduction with the reduce method
            a.reduce(
                spinset_rb,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=3,
            )

            # try reduction with the __call__ method
            a(
                spinset_rb,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=3,
            )

            # this should also have saved a couple of files in the current
            # directory
            assert os.path.isfile("./PLP0012785_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012787_0_PolCorr.dat")

    def test_correction_changes(self):
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            spinset_rb = SpinSet(
                down_down="PLP0012785.nx.hdf",
                up_up="PLP0012787.nx.hdf",
                up_down="PLP0012786.nx.hdf",
                down_up="PLP0012788.nx.hdf",
                data_folder=self.pth,
            )
            spinset_db = SpinSet(
                down_down="PLP0012793.nx.hdf",
                up_up="PLP0012795.nx.hdf",
                up_down="PLP0012794.nx.hdf",
                down_up="PLP0012796.nx.hdf",
                data_folder=self.pth,
            )
            a = PolarisedReduce(spinset_db)

            # correct and reduce spectra
            a.reduce(
                spinset_rb,
                data_folder=self.pth,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=3,
            )

            for reducer in a.reducers.values():
                with pytest.raises(AssertionError):
                    assert_equal(
                        reducer.reflected_beam.m_spec,
                        reducer.reflected_beam.m_spec_polcorr,
                    )

    def test_file_output(self):
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            spinset_rb = SpinSet(
                down_down="PLP0012785.nx.hdf",
                up_up="PLP0012787.nx.hdf",
                up_down="PLP0012786.nx.hdf",
                down_up="PLP0012788.nx.hdf",
                data_folder=self.pth,
            )
            spinset_db = SpinSet(
                down_down="PLP0012793.nx.hdf",
                up_up="PLP0012795.nx.hdf",
                up_down="PLP0012794.nx.hdf",
                down_up="PLP0012796.nx.hdf",
                data_folder=self.pth,
            )
            a = PolarisedReduce(spinset_db)

            # reduce data
            a.reduce(
                spinset_rb,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=3,
            )

            # this should also have saved a few files in the current
            # directory
            assert os.path.isfile("./PLP0012785_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012787_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012786_0_PolCorr.dat")
            assert os.path.isfile("./PLP0012788_0_PolCorr.dat")

            # can we read the file
            dd = ReflectDataset("./PLP0012785_0_PolCorr.dat")
            uu = ReflectDataset("./PLP0012787_0_PolCorr.dat")
            ud = ReflectDataset("./PLP0012786_0_PolCorr.dat")
            du = ReflectDataset("./PLP0012788_0_PolCorr.dat")

            # check if the written data is the same as what is in the reducers
            assert_equal(dd.x, list(reversed(a.reducers["dd"].x[0])))
            assert_equal(du.x, list(reversed(a.reducers["du"].x[0])))
            assert_equal(ud.x, list(reversed(a.reducers["ud"].x[0])))
            assert_equal(uu.x, list(reversed(a.reducers["uu"].x[0])))

            assert_equal(dd.y, list(reversed(a.reducers["dd"].y_corr[0])))
            assert_equal(du.y, list(reversed(a.reducers["du"].y_corr[0])))
            assert_equal(ud.y, list(reversed(a.reducers["ud"].y_corr[0])))
            assert_equal(uu.y, list(reversed(a.reducers["uu"].y_corr[0])))


class TestPolarisationEfficiency:
    @pytest.mark.usefixtures("no_data_directory")
    @pytest.fixture(autouse=True)
    def setup_method(self, tmpdir, data_directory):
        self.pth = pjoin(data_directory, "reduce")

        self.cwd = os.getcwd()
        self.tmpdir = tmpdir.strpath
        os.chdir(self.tmpdir)
        return 0

    def teardown_method(self):
        os.chdir(self.cwd)

    def test_smoke(self):
        wavelength_axis = calculate_wavelength_bins(2.5, 12.5, 3)

        peff = PolarisationEfficiency(wavelength_axis)

        assert peff.combined_efficiency_matrix.shape == tuple(
            [len(wavelength_axis), 4, 4]
        )

    def test_config_difference(self):
        wavelength_axis = calculate_wavelength_bins(2.5, 12.5, 3)

        p_PF = PolarisationEfficiency(wavelength_axis, config="PF")
        p_full = PolarisationEfficiency(wavelength_axis, config="full")

        with pytest.raises(AssertionError):
            assert_equal(
                p_PF.combined_efficiency_matrix,
                p_full.combined_efficiency_matrix,
            )

    def test_input(self):
        wavelength_axis = calculate_wavelength_bins(2.5, 12.5, 3).reshape(
            1, -1
        )

        with pytest.raises(ValueError):
            peff = PolarisationEfficiency(wavelength_axis)

            assert peff
