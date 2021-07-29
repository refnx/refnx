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
                down_down=pjoin(self.pth, "PLP0012785.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012787.nx.hdf"),
                up_down=pjoin(self.pth, "PLP0012786.nx.hdf"),
                down_up=pjoin(self.pth, "PLP0012788.nx.hdf"),
            )
            spinset_db = SpinSet(
                down_down=pjoin(self.pth, "PLP0012793.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012795.nx.hdf"),
                up_down=pjoin(self.pth, "PLP0012794.nx.hdf"),
                down_up=pjoin(self.pth, "PLP0012796.nx.hdf"),
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

            assert_equal(
                a.reducers["dd"].direct_beam.m_spec,
                a.reducers["du"].direct_beam.m_spec,
            )
            assert_equal(
                a.reducers["uu"].direct_beam.m_spec,
                a.reducers["ud"].direct_beam.m_spec,
            )
        # Check that the shapes of all the spectra that need to be the
        # same are the same.
        for sc in a._reduced_successfully:
            reducer = a.reducers[sc]
            assert (
                reducer.reflected_beam.m_spec_polcorr.shape
                == reducer.reflected_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec_polcorr.shape
                == reducer.direct_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec_polcorr.shape
                == reducer.reflected_beam.m_spec_polcorr.shape
            )
            assert (
                reducer.reflected_beam.m_spec.shape
                == reducer.reflected_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec.shape
                == reducer.direct_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec.shape
                == reducer.reflected_beam.m_spec.shape
            )

            # Make sure there is a difference between m_spec and
            # m_spec_polcorr
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
                down_down=pjoin(self.pth, "PLP0012785.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012787.nx.hdf"),
                up_down=pjoin(self.pth, "PLP0012786.nx.hdf"),
            )
            spinset_db = SpinSet(
                down_down=pjoin(self.pth, "PLP0012793.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012795.nx.hdf"),
                up_down=pjoin(self.pth, "PLP0012794.nx.hdf"),
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

            # can we read the file
            dd = ReflectDataset("./PLP0012785_0_PolCorr.dat")
            uu = ReflectDataset("./PLP0012787_0_PolCorr.dat")
            ud = ReflectDataset("./PLP0012786_0_PolCorr.dat")

            # check if the written data is the same as what is in the reducers
            # Note: the order of the data is reversed in the reducers
            # compared to the .dat file
            assert_equal(dd.x, list(reversed(a.reducers["dd"].x[0])))
            assert_equal(ud.x, list(reversed(a.reducers["ud"].x[0])))
            assert_equal(uu.x, list(reversed(a.reducers["uu"].x[0])))

            assert_equal(dd.y, list(reversed(a.reducers["dd"].y_corr[0])))
            assert_equal(ud.y, list(reversed(a.reducers["ud"].y_corr[0])))
            assert_equal(uu.y, list(reversed(a.reducers["uu"].y_corr[0])))

            assert_equal(
                a.reducers["dd"].direct_beam.m_spec,
                a.reducers["du"].direct_beam.m_spec,
            )
            assert_equal(
                a.reducers["uu"].direct_beam.m_spec,
                a.reducers["ud"].direct_beam.m_spec,
            )
        # Check that the shapes of all the spectra that need to be the
        # same are the same.
        for sc in a._reduced_successfully:
            reducer = a.reducers[sc]
            assert (
                reducer.reflected_beam.m_spec_polcorr.shape
                == reducer.reflected_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec_polcorr.shape
                == reducer.direct_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec_polcorr.shape
                == reducer.reflected_beam.m_spec_polcorr.shape
            )
            assert (
                reducer.reflected_beam.m_spec.shape
                == reducer.reflected_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec.shape
                == reducer.direct_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec.shape
                == reducer.reflected_beam.m_spec.shape
            )

            # Make sure there is a difference between m_spec and
            # m_spec_polcorr
            with pytest.raises(AssertionError):
                assert_equal(
                    reducer.reflected_beam.m_spec,
                    reducer.reflected_beam.m_spec_polcorr,
                )

    def test_polarised_reduction_method_2sc(self):
        # a quick smoke test to check that the reduction can occur
        # warnings filter for pixel size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            spinset_rb = SpinSet(
                down_down=pjoin(self.pth, "PLP0012785.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012787.nx.hdf"),
            )
            spinset_db = SpinSet(
                down_down=pjoin(self.pth, "PLP0012793.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012795.nx.hdf"),
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

            # can we read the file
            dd = ReflectDataset("./PLP0012785_0_PolCorr.dat")
            uu = ReflectDataset("./PLP0012787_0_PolCorr.dat")

            # check if the written data is the same as what is in the reducers
            assert_equal(dd.x, list(reversed(a.reducers["dd"].x[0])))
            assert_equal(uu.x, list(reversed(a.reducers["uu"].x[0])))

            assert_equal(dd.y, list(reversed(a.reducers["dd"].y_corr[0])))
            assert_equal(uu.y, list(reversed(a.reducers["uu"].y_corr[0])))

            assert_equal(
                a.reducers["dd"].direct_beam.m_spec,
                a.reducers["du"].direct_beam.m_spec,
            )
            assert_equal(
                a.reducers["uu"].direct_beam.m_spec,
                a.reducers["ud"].direct_beam.m_spec,
            )
        # Check that the shapes of all the spectra that need to be the
        # same are the same.
        for sc in a._reduced_successfully:
            reducer = a.reducers[sc]
            assert (
                reducer.reflected_beam.m_spec_polcorr.shape
                == reducer.reflected_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec_polcorr.shape
                == reducer.direct_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec_polcorr.shape
                == reducer.reflected_beam.m_spec_polcorr.shape
            )
            assert (
                reducer.reflected_beam.m_spec.shape
                == reducer.reflected_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec.shape
                == reducer.direct_beam.m_spec_sd.shape
            )
            assert (
                reducer.direct_beam.m_spec.shape
                == reducer.reflected_beam.m_spec.shape
            )

            # Make sure there is a difference between m_spec and
            # m_spec_polcorr
            with pytest.raises(AssertionError):
                assert_equal(
                    reducer.reflected_beam.m_spec,
                    reducer.reflected_beam.m_spec_polcorr,
                )

    def test_nsf_spin_channels(self):
        """
        Check that the efficiency-corrected R++ channel is assigned properly
        in the reducer by comparing where the reflectivity drops below a
        certain threshold
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            spinset_rb = SpinSet(
                down_down=pjoin(self.pth, "PLP0012785.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012787.nx.hdf"),
            )
            spinset_db = SpinSet(
                down_down=pjoin(self.pth, "PLP0012793.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012795.nx.hdf"),
            )
            a = PolarisedReduce(spinset_db)

            # Reduce and correct data
            a.reduce(
                spinset_rb,
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=3,
            )

            # Get Q position where the reflectivity drops to 0.5 for the
            # "uu" and "dd" datasets
            q_dd = a.reducers["dd"].x[0][
                abs(a.reducers["dd"].y - 0.5).argmin()
            ]
            q_uu = a.reducers["uu"].x[0][
                abs(a.reducers["uu"].y - 0.5).argmin()
            ]
            # Check x_uu is larger than x_dd. This is because the spin up
            # neutrons see a higher potential for the magnetically saturated
            # permalloy film. This checks that the "uu" and "dd" spin channels
            # didn't get mixed up during the reduction process
            assert q_uu > q_dd


class TestPolarisationEfficiency:
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

    def test_manual_input_to_PolarisedReduce(self):
        """
        Test the manual input of pol_eff into a
        `refnx.reduce.PolarisedReduce` object and see if it is the same as
        an automatically created one
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Create SpinSets, PolarisedReduce object and ReductionOptions
            spinset_rb = SpinSet(
                down_down=pjoin(self.pth, "PLP0012785.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012787.nx.hdf"),
            )
            spinset_db = SpinSet(
                down_down=pjoin(self.pth, "PLP0012793.nx.hdf"),
                up_up=pjoin(self.pth, "PLP0012795.nx.hdf"),
            )
            a = PolarisedReduce(spinset_db)

            rdo = ReductionOptions(
                lo_wavelength=2.5,
                hi_wavelength=12.5,
                rebin_percent=3,
            )

            # Reduce and correct reflected beams
            a.reduce(spinset_rb, **rdo)

            # Create another polarised reducer to compare
            b = PolarisedReduce(spinset_db)
            # Create PolarisationEfficiency object and reduce
            pol_eff = PolarisationEfficiency(
                a.reducers["dd"].direct_beam.m_lambda[0],
                config="full",
            )
            b.reduce(spinset_rb, pol_eff=pol_eff, **rdo)

            for sc in a._reduced_successfully:
                assert_equal(a.reducers[sc].x, b.reducers[sc].x)
                assert_equal(a.reducers[sc].y, b.reducers[sc].y)
                assert_equal(a.reducers[sc].y_err, b.reducers[sc].y_err)
                assert_equal(a.reducers[sc].y_corr, b.reducers[sc].y_corr)
