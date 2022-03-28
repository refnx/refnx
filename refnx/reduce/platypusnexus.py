import io
import os
import os.path
import glob
import argparse
import re
import shutil
from time import gmtime, strftime
import string
import warnings
from contextlib import contextmanager
from enum import Enum

from scipy.optimize import leastsq, curve_fit
from scipy.stats import t
import pandas as pd
import numpy as np
import h5py

from refnx.reduce.peak_utils import peak_finder, centroid
import refnx.util.general as general
from refnx.util.general import resolution_double_chopper, _dict_compare
import refnx.util.ErrorProp as EP
from refnx.reduce.parabolic_motion import (
    find_trajectory,
    y_deflection,
    parabola,
)
from refnx.reduce.event import (
    events,
    process_event_stream,
    framebins_to_frames,
)
from refnx.reduce.rebin import rebin, rebin_along_axis
from refnx._lib import possibly_open_file


EXTENT_MULT = 2
PIXEL_OFFSET = 4

spectrum_template = """<?xml version="1.0"?>
<REFroot xmlns="">
<REFentry time="$time">
<Title>$title</Title>
<REFdata axes="lambda" rank="1" type="POINT"\
 spin="UNPOLARISED" dim="$n_spectra">
<Run filename="$runnumber"/>
<R uncertainty="dR">$r</R>
<lambda uncertainty="dlambda" units="1/A">$lmda</lambda>
<dR type="SD">$dr</dR>
<dlambda type="_FWHM" units="1/A">$dlmda</dlambda>
</REFdata>
</REFentry>
</REFroot>"""


def catalogue(start, stop, data_folder=None, prefix="PLP"):
    """
    Extract interesting information from Platypus NeXUS files.

    Parameters
    ----------
    start : int
        start cataloguing from this run number
    stop : int
        stop cataloguing at this run number
    data_folder : str, optional
        path specifying location of NeXUS files
    prefix : {'PLP', 'SPZ'}, optional
        str specifying whether you want to catalogue Platypus or Spatz files

    Returns
    -------
    catalog : pd.DataFrame
        Dataframe containing interesting parameters from Platypus Nexus files
    """
    info = ["filename", "end_time", "sample_name"]

    if prefix == "PLP":
        info += ["ss1vg", "ss2vg", "ss3vg", "ss4vg"]
    elif prefix == "SPZ":
        info += ["ss2hg", "ss3hg", "ss4hg"]

    info += [
        "omega",
        "twotheta",
        "total_counts",
        "bm1_counts",
        "time",
        "daq_dirname",
        "start_time",
    ]

    run_number = []
    d = {key: [] for key in info}

    if data_folder is None:
        data_folder = "."

    files = glob.glob(os.path.join(data_folder, prefix + "*.nx.hdf"))
    files.sort()
    files = [
        file
        for file in files
        if datafile_number(file, prefix=prefix) in range(start, stop + 1)
    ]

    for idx, file in enumerate(files):
        if prefix == "PLP":
            pn = PlatypusNexus(file)
        elif prefix == "SPZ":
            pn = SpatzNexus(file)
        else:
            raise RuntimeError("prefix not known yet")

        cat = pn.cat.cat
        run_number.append(idx)

        for key, val in d.items():
            data = cat[key]
            if np.size(data) > 1 or type(data) is np.ndarray:
                data = data[0]
            if type(data) is bytes:
                data = data.decode()

            d[key].append(data)

    df = pd.DataFrame(d, index=run_number, columns=info)

    return df


class Catalogue:
    """
    Extract relevant parts of a NeXus file for reflectometry reduction
    """

    def __init__(self, h5d):
        """
        Extract relevant parts of a NeXus file for reflectometry reduction
        Access information via dict access, e.g. cat['detector'].

        Parameters
        ----------
        h5d - HDF5 file handle
        """
        self.prefix = None

        d = {}
        file_path = os.path.realpath(h5d.filename)
        d["path"] = os.path.dirname(file_path)
        d["filename"] = h5d.filename
        try:
            d["end_time"] = h5d["entry1/end_time"][0]
        except KeyError:
            # Autoreduce field tests show that this key may not be present in
            # some files before final write.
            d["end_time"] = ""

        d["detector"] = h5d["entry1/data/hmm"][:]
        d["t_bins"] = h5d["entry1/data/time_of_flight"][:].astype("float64")
        d["x_bins"] = h5d["entry1/data/x_bin"][:]
        d["y_bins"] = h5d["entry1/data/y_bin"][:]

        d["bm1_counts"] = h5d["entry1/monitor/bm1_counts"][:]
        d["total_counts"] = h5d["entry1/instrument/detector/total_counts"][:]
        d["time"] = h5d["entry1/instrument/detector/time"][:]

        try:
            event_directory_name = h5d[
                "entry1/instrument/detector/daq_dirname"
            ][0]
            d["daq_dirname"] = event_directory_name.decode()
        except KeyError:
            # daq_dirname doesn't exist in this file
            d["daq_dirname"] = None

        d["ss2vg"] = h5d["entry1/instrument/slits/second/vertical/gap"][:]
        d["ss3vg"] = h5d["entry1/instrument/slits/third/vertical/gap"][:]
        d["ss4vg"] = h5d["entry1/instrument/slits/fourth/vertical/gap"][:]
        d["ss2hg"] = h5d["entry1/instrument/slits/second/horizontal/gap"][:]
        d["ss3hg"] = h5d["entry1/instrument/slits/third/horizontal/gap"][:]
        d["ss4hg"] = h5d["entry1/instrument/slits/fourth/horizontal/gap"][:]

        d["sample_distance"] = h5d[
            "entry1/instrument/parameters/sample_distance"
        ][:]
        d["slit2_distance"] = h5d[
            "entry1/instrument/parameters/slit2_distance"
        ][:]
        d["slit3_distance"] = h5d[
            "entry1/instrument/parameters/slit3_distance"
        ][:]
        d["collimation_distance"] = d["slit3_distance"] - d["slit2_distance"]
        try:
            san = (
                h5d["entry1/data/hmm"]
                .attrs["axes"]
                .decode("utf8")
                .split(":")[0]
            )
        except AttributeError:
            # the attribute could be a string already
            san = str(h5d["entry1/data/hmm"].attrs["axes"]).split(":")[0]
        finally:
            d["scan_axis_name"] = san

        d["scan_axis"] = h5d[f"entry1/data/{d['scan_axis_name']}"][:]

        try:
            d["start_time"] = h5d["entry1/instrument/detector/start_time"][:]
        except KeyError:
            # start times don't exist in this file
            d["start_time"] = None

        d["original_file_name"] = h5d["entry1/experiment/file_name"][:]
        d["sample_name"] = h5d["entry1/sample/name"][:]
        self.cat = d

    def __getattr__(self, item):
        try:
            return self.cat[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.cat = state["cat"]
        self.prefix = state["prefix"]

    @property
    def datafile_number(self):
        return datafile_number(self.filename, prefix=self.prefix)


class SpatzCatalogue(Catalogue):
    """
    Extract relevant parts of a NeXus file for reflectometry reduction
    """

    def __init__(self, h5d):
        """
        Extract relevant parts of a NeXus file for reflectometry reduction
        Access information via dict access, e.g. cat['detector'].

        Parameters
        ----------
        h5d - HDF5 file handle
        """
        super().__init__(h5d)
        self.prefix = "SPZ"

        d = self.cat

        # grab chopper settings
        master, slave, frequency, phase = self._chopper_values(h5d)
        d["master"] = master
        # slave == 2 --> chopper 2
        # slave == 3 --> chopper 2B
        # slave == 4 --> chopper 3
        d["slave"] = slave
        d["frequency"] = frequency
        d["phase"] = phase
        d["t_offset"] = None
        if "t_offset" in h5d:
            d["t_offset"] = h5d["entry1/instrument/parameters/t_offset"][:]

        d["chopper2_distance"] = h5d["entry1/instrument/ch02_distance/pos"][:]
        d["chopper2B_distance"] = h5d[
            "entry1/instrument/parameters/ch02b_distance"
        ][:]
        d["chopper3_distance"] = h5d[
            "entry1/instrument/parameters/ch03_distance"
        ][:]

        # collimation parameters
        # first and second collimation slits
        d["ss_coll1"] = h5d["entry1/instrument/slits/second/horizontal/gap"][:]
        d["ss_coll2"] = h5d["entry1/instrument/slits/third/horizontal/gap"][:]

        # sample omega, the nominal angle of incidence
        d["omega"] = h5d["entry1/sample/som"][:]
        d["som"] = d["omega"]
        # two theta value for detector arm.
        d["twotheta"] = h5d["entry1/instrument/detector/detrot"][:]
        d["detrot"] = d["twotheta"]
        d["dz"] = d["twotheta"]

        # detector longitudinal translation from sample
        d["dy"] = (
            h5d["entry1/instrument/detector/detector_distance/pos"][:]
            - d["sample_distance"]
        )

        # logical size (mm) of 1 pixel in the scattering plane
        try:
            d["qz_pixel_size"] = h5d[
                "entry1/instrument/parameters/qz_pixel_size"
            ][:]
        except KeyError:
            # older SPZ files didn't have qz_pixel_size
            d["qz_pixel_size"] = np.array([0.326])

    def _chopper_values(self, h5data):
        """
        Obtains chopper settings from NeXUS file

        Parameters
        ----------
        h5data : HDF5 NeXUS file
            datafile,

        Returns
        -------
        master, slave, frequency, phase : float, float, float, float
        """
        master = 1
        slave = 2

        d = self.cat

        chopper1_speed = h5data["entry1/instrument/chopper/c01/spee"][:]
        chopper2_speed = h5data["entry1/instrument/chopper/c02/spee"][:]
        chopper2B_speed = h5data["entry1/instrument/chopper/c2b/spee"][:]
        # chopper3_speed = h5data['entry1/instrument/chopper/c03/spee']

        ch1phase = h5data["entry1/instrument/chopper/c01/spha"]
        ch2phase = h5data["entry1/instrument/chopper/c02/spha"][:]
        ch2Bphase = h5data["entry1/instrument/chopper/c2b/spha"][:]
        # ch3phase = h5data['entry1/instrument/chopper/c03/spha']

        if chopper1_speed[0] > 2:
            master = 1
            d["master_phase_offset"] = h5data[
                "entry1/instrument/parameters/poff_c1_master"
            ][:]
            if chopper2_speed[0] > 2:
                slave = 2
            else:
                # chopper2B is slave
                slave = 3
            freq = chopper1_speed
            phase = ch2phase - ch1phase
            d["poff_c2_slave_1_master"] = h5data[
                "entry1/instrument/parameters/poff_c2_slave_1_master"
            ][:]
            d["poff_c2b_slave_1_master"] = h5data[
                "entry1/instrument/parameters/poff_c2b_slave_1_master"
            ][:]
        else:
            master = 2
            d["master_phase_offset"] = h5data[
                "entry1/instrument/parameters/poff_c2_master"
            ][:]
            d["poff_c2b_slave_2_master"] = h5data[
                "entry1/instrument/parameters/poff_c2b_slave_2_master"
            ][:]

            freq = chopper2_speed
            # if slave == 3 refers to chopper 2B
            assert (chopper2B_speed > 1).all()

            slave = 3
            phase = ch2Bphase - ch2phase

        # SPZ offsets measured on 20200116
        # with master = 1, slave = 2
        # master_phase_offset = -25.90
        # chopper2_phase_offset -0.22 degrees

        return master, slave, freq, phase


class PlatypusCatalogue(Catalogue):
    """
    Extract relevant parts of a NeXus file for reflectometry reduction
    """

    def __init__(self, h5d):
        """
        Extract relevant parts of a NeXus file for reflectometry reduction
        Access information via dict access, e.g. cat['detector'].

        Parameters
        ----------
        h5d - HDF5 file handle
        """
        super().__init__(h5d)
        self.prefix = "PLP"

        d = self.cat
        d["ss1vg"] = h5d["entry1/instrument/slits/first/vertical/gap"][:]
        d["ss1hg"] = h5d["entry1/instrument/slits/first/horizontal/gap"][:]

        d["omega"] = h5d["entry1/instrument/parameters/omega"][:]
        d["twotheta"] = h5d["entry1/instrument/parameters/twotheta"][:]

        d["sth"] = h5d["entry1/sample/sth"][:]
        d["mode"] = h5d["entry1/instrument/parameters/mode"][0].decode()

        master, slave, frequency, phase = self._chopper_values(h5d)
        d["master"] = master
        d["slave"] = slave
        d["frequency"] = frequency
        d["phase"] = phase
        d["chopper2_distance"] = h5d[
            "entry1/instrument/parameters/chopper2_distance"
        ][:]
        d["chopper3_distance"] = h5d[
            "entry1/instrument/parameters/chopper3_distance"
        ][:]
        d["chopper4_distance"] = h5d[
            "entry1/instrument/parameters/chopper4_distance"
        ][:]
        d["master_phase_offset"] = h5d[
            "entry1/instrument/parameters/chopper1_phase_offset"
        ][:]
        d["chopper2_phase_offset"] = h5d[
            "entry1/instrument/parameters/chopper2_phase_offset"
        ][:]
        d["chopper3_phase_offset"] = h5d[
            "entry1/instrument/parameters/chopper3_phase_offset"
        ][:]
        d["chopper4_phase_offset"] = h5d[
            "entry1/instrument/parameters/chopper4_phase_offset"
        ][:]
        # time offset for choppers if you're using a signal generator to
        # delay T0
        d["t_offset"] = None
        if "t_offset" in h5d:
            d["t_offset"] = h5d["entry1/instrument/parameters/t_offset"][:]

        d["guide1_distance"] = h5d[
            "entry1/instrument/parameters/guide1_distance"
        ][:]
        d["guide2_distance"] = h5d[
            "entry1/instrument/parameters/guide2_distance"
        ][:]

        # collimation parameters
        # first and second collimation slits
        d["ss_coll1"] = h5d["entry1/instrument/slits/second/vertical/gap"][:]
        d["ss_coll2"] = h5d["entry1/instrument/slits/third/vertical/gap"][:]

        d["dy"] = h5d["entry1/instrument/detector/longitudinal_translation"][:]
        d["dz"] = h5d["entry1/instrument/detector/vertical_translation"][:]

        # pixel size (mm) in scattering plane. y_pixels_per_mm is incorrect,
        # it should really be mm_per_y_pixel, but let's stick with the
        # historical error
        try:
            d["qz_pixel_size"] = h5d[
                "entry1/instrument/parameters/y_pixels_per_mm"
            ][:]
        except KeyError:
            # older PLP files didn't have y_pixels_per_mm, so use built in
            # value
            warnings.warn(
                "Setting default pixel size to 1.177", RuntimeWarning
            )
            d["qz_pixel_size"] = np.array([1.177])

    def _chopper_values(self, h5data):
        """
        Obtains chopper settings from NeXUS file

        Parameters
        ----------
        h5data : HDF5 NeXUS file
            datafile,

        Returns
        -------
        master, slave, frequency, phase : float, float, float, float
        """
        chopper1_speed = h5data["entry1/instrument/disk_chopper/ch1speed"]
        chopper2_speed = h5data["entry1/instrument/disk_chopper/ch2speed"]
        chopper3_speed = h5data["entry1/instrument/disk_chopper/ch3speed"]
        chopper4_speed = h5data["entry1/instrument/disk_chopper/ch4speed"]
        ch2phase = h5data["entry1/instrument/disk_chopper/ch2phase"]
        ch3phase = h5data["entry1/instrument/disk_chopper/ch3phase"]
        ch4phase = h5data["entry1/instrument/disk_chopper/ch4phase"]

        m = "entry1/instrument/parameters/master"
        s = "entry1/instrument/parameters/slave"
        if (
            s in h5data
            and m in h5data
            and h5data[m][0] in [1, 2, 3, 4]
            and h5data[s][0] in [1, 2, 3, 4]
        ):
            # master and slave parameters have to be set correctly in order
            # to use them.
            master = h5data["entry1/instrument/parameters/master"][0]
            slave = h5data["entry1/instrument/parameters/slave"][0]
        else:
            master = 1
            if abs(chopper2_speed[0]) > 10:
                slave = 2
            elif abs(chopper3_speed[0]) > 10:
                slave = 3
            else:
                slave = 4

        speeds = np.array(
            [chopper1_speed, chopper2_speed, chopper3_speed, chopper4_speed]
        )

        phases = np.array(
            [np.zeros_like(ch2phase), ch2phase, ch3phase, ch4phase]
        )

        return master, slave, speeds[0] / 60.0, phases[slave - 1]


class PolarisedCatalogue(PlatypusCatalogue):
    """
    Extract relevant parts of a polarised PLATYPUS
    NeXus file for reflectometry reduction.
    Access information via dict access, e.g. cat['pol_flip_freq'].

    Parameters
    ----------
    h5d - HDF5 file handle
    """

    def __init__(self, h5d):

        super().__init__(h5d)
        # Is there a magnet?
        self.is_magnet = False
        # Is there a cryocooler?
        self.is_cryo = False
        # Is there a power supply?
        self.is_power_supply = False

        d = self.cat
        self._polariser_flippers(d, h5d)
        self._analyser_flippers(d, h5d)
        self._check_sample_environments(d, h5d)

    def _polariser_flippers(self, d, h5d):
        d["pol_flip_freq"] = h5d[
            "entry1/instrument/polarizer_flipper/flip_frequency"
        ][0]
        d["pol_flip_current"] = h5d[
            "entry1/instrument/polarizer_flipper/flip_current"
        ][0]
        d["pol_flip_voltage"] = h5d[
            "entry1/instrument/polarizer_flipper/flip_voltage"
        ][0]
        d["pol_flip_status"] = h5d[
            "entry1/instrument/polarizer_flipper/flip_on"
        ][0]
        d["pol_guide_current"] = h5d[
            "entry1/instrument/polarizer_flipper/guide_current"
        ][0]

    def _analyser_flippers(self, d, h5d):
        d["anal_flip_freq"] = h5d[
            "entry1/instrument/analyzer_flipper/flip_frequency"
        ][0]
        d["anal_flip_current"] = h5d[
            "entry1/instrument/analyzer_flipper/flip_current"
        ][0]
        d["anal_flip_voltage"] = h5d[
            "entry1/instrument/analyzer_flipper/flip_voltage"
        ][0]
        d["anal_flip_status"] = h5d[
            "entry1/instrument/analyzer_flipper/flip_on"
        ][0]
        d["anal_guide_current"] = h5d[
            "entry1/instrument/analyzer_flipper/guide_current"
        ][0]
        d["anal_base"] = h5d["entry1/instrument/polarizer/anal_base"][0]
        d["anal_dist"] = h5d["entry1/instrument/polarizer/anal_distance"][0]
        d["rotation"] = h5d["entry1/instrument/polarizer/rotation"][0]
        d["z_trans"] = h5d["entry1/instrument/polarizer/z_translation"][0]

    def _check_sample_environments(self, d, h5d):
        try:
            # Try adding temperature sensor values to dict
            d["temp_sensorA"] = h5d["entry1/sample/tc1/sensor/sensorValueA"][0]
            d["temp_sensorB"] = h5d["entry1/sample/tc1/sensor/sensorValueB"][0]
            d["temp_sensorC"] = h5d["entry1/sample/tc1/sensor/sensorValueC"][0]
            d["temp_sensorD"] = h5d["entry1/sample/tc1/sensor/sensorValueD"][0]
            d["temp_setpt1"] = h5d["entry1/sample/tc1/sensor/setpoint1"][0]
            d["temp_setpt2"] = h5d["entry1/sample/tc1/sensor/setpoint2"][0]
            self.is_cryo = True
        except KeyError:
            # Temperature sensor not used in measurement - set to None
            d["temp_sensorA"] = None
            d["temp_sensorB"] = None
            d["temp_sensorC"] = None
            d["temp_sensorD"] = None
            d["temp_setpt1"] = None
            d["temp_setpt2"] = None
            self.is_cryo = False

        try:
            # Try adding voltage supply to dict
            d["pow_supply_volts"] = h5d["entry1/sample/power_supply/voltage"][
                0
            ]
            d["pow_supply_current"] = h5d["entry1/sample/power_supply/amps"][0]
            d["pow_supply_relay"] = h5d["entry1/sample/power_supply/relay"][0]
            self.is_power_supply = True
        except KeyError:
            # Voltage supply not used in measurement
            d["pow_supply_volts"] = None
            d["pow_supply_current"] = None
            d["pow_supply_relay"] = None
            self.is_power_supply = False

        try:
            # Try adding magnetic field sensor to dict
            d["magnet_current_set"] = h5d[
                "entry1/sample/ma1/sensor/desired_current"
            ][0]
            d["magnet_set_field"] = h5d[
                "entry1/sample/ma1/sensor/desired_field"
            ][0]
            d["magnet_measured_field"] = h5d[
                "entry1/sample/ma1/sensor/measured_field"
            ][0]
            d["magnet_output_current"] = h5d[
                "entry1/sample/ma1/sensor/nominal_outp_current"
            ][0]
            self.is_magnet = True
        except KeyError:
            # Magnetic field sensor not used in measurement - set to None
            d["magnet_current_set"] = None
            d["magnet_set_field"] = None
            d["magnet_measured_field"] = None
            d["magnet_output_current"] = None
            self.is_magnet = False


class SpinChannel(Enum):
    """
    Describes the incident and scattered spin state of a polarised neutron beam.
    """

    UP_UP = (1, 1)
    UP_DOWN = (1, 0)
    DOWN_UP = (0, 1)
    DOWN_DOWN = (0, 0)


class SpinSet(object):
    """
    Describes a set of spin-channels at a given angle of incidence,
    and can process beams with individual reduction options.

    Parameters
    ----------
    down_down   :   str or refnx.reduce.PlatypusNexus
        Input filename or PlatypusNexus object for the R-- spin
        channel.
    up_up       :   str or refnx.reduce.PlatypusNexus
        Input filename or PlatypusNexus object for the R++ spin
        channel.
    down_up     :   str or refnx.reduce.PlatypusNexus, optional
        Input filename or PlatypusNexus object for the R-+ spin
        channel.
    up_down     :   str or refnx.reduce.PlatypusNexus, optional
        Input filename or PlatypusNexus object for the R+- spin
        channel.

    Attributes
    ----------
    channels    :   dict
        Dictionary of each measured spin channel
            "dd"    :   refnx.reduce.PlatypusNexus (R--)
            "du"    :   refnx.reduce.PlatypusNexus or None (R-+)
            "ud"    :   refnx.reduce.PlatypusNexus or None (R+-)
            "uu"    :   refnx.reduce.PlatypusNexus (R++)
    sc_opts     :   dict of refnx.reduce.ReductionOptions
        Reduction options for each spin channel ("dd", "du", "ud", "uu)
    dd          :   refnx.reduce.PlatypusNexus
        R-- spin channel
    uu          :   refnx.reduce.PlatypusNexus
        R++ spin channel
    du          :   refnx.reduce.PlatypusNexus or None
        R-+ spin channel
    ud          :   refnx.reduce.PlatypusNexus or None
        R+- spin channel

    Notes
    -----
    Each of the `ReductionOptions` specified in `dd_opts,` etc, is used
    to specify the options used to reduce each spin channel. The following
    reduction options must be consistent and identical across all
    spin channels so as to maintain the same wavelength axis across
    the datasets:

    lo_wavelength    : key in refnx.reduce.ReductionOptions
    hi_wavelength    : key in refnx.reduce.ReductionOptions
    rebin_percent    : key in refnx.reduce.ReductionOptions
    wavelength_bins  : key in refnx.reduce.ReductionOptions
    """

    def __init__(self, down_down, up_up, down_up=None, up_down=None):
        # Currently only Platypus has polarisation elements
        self.reflect_klass = PlatypusNexus

        # initialise spin channels
        self.channels = {
            "dd": None,
            "du": None,
            "ud": None,
            "uu": None,
        }

        self.sc_opts = {
            "dd": {},
            "du": {},
            "ud": {},
            "uu": {},
        }

        # initialise reduction options for each spin channel
        reduction_options = ReductionOptions(
            lo_wavelength=2.5,
            hi_wavelength=12.5,
            rebin_percent=3,
        )
        # Put inputs into a dictionary to iterate over
        input_params = {
            "dd": down_down,
            "du": down_up,
            "ud": up_down,
            "uu": up_up,
        }
        _spin_channels = {
            "dd": SpinChannel.DOWN_DOWN,
            "du": SpinChannel.DOWN_UP,
            "ud": SpinChannel.UP_DOWN,
            "uu": SpinChannel.UP_UP,
        }

        # Load the files and check spin channels and flipper config
        for sc, input_param in input_params.items():
            if input_param is None:
                # Spin channel not measured
                continue
            elif isinstance(input_param, self.reflect_klass):
                # Spin channel inputted as PlatypusNexus object
                channel = input_param
            else:
                # Spin channel inputted as file string
                channel = self.reflect_klass(input_param)

            # print(sc, input_param, channel.spin_state, _spin_channels[sc])
            if channel.spin_state is _spin_channels[sc]:
                self.channels[sc] = channel
                self.sc_opts[sc] = reduction_options.copy()
            else:
                RuntimeError(
                    f"Supplied spin channel {_spin_channels[sc]} does not match flipper status"
                )

    @property
    def dd(self):
        return self.channels["dd"]

    @property
    def du(self):
        return self.channels["du"]

    @property
    def ud(self):
        return self.channels["ud"]

    @property
    def uu(self):
        return self.channels["uu"]

    @property
    def spin_channels(self):
        """
        Gives a quick indication of what spin channels were measured and
        are present in this SpinSet.

        Returns
        -------
        list of refnx.reduce.SpinChannel Enum values or None, depending on
        if the spin channel was measured.
        """
        return [
            self.channels[sc].spin_state.value
            if self.channels[sc] is not None
            else None
            for sc in self.channels
        ]

    def process(self, **reduction_options):
        """
        Process beams in SpinSet.

        If reduction_options is None, the reduction options for each spin
        channel are specified by the dictionary of spin channel reduction
        options `SpinSet.sc_opts` which are initialised to the
        standard options when constructing the object.

        If you wish to have unique reduction options for each spin channel,
        you need to ensure that the wavelength bins between each spin channel
        remain identical, otherwise a ValueError will be raised.

        If `reduction_options`
        is not None, then SpinSet.process() will use these options for all
        spin channels.

        Parameters
        ----------
        reduction_options : dict, optional
            A single dict of options used to process all spectra.
        """

        if reduction_options is not None:
            for sc in self.sc_opts:
                self.sc_opts[sc].update(reduction_options)

        # Check specific reduction options are the same across all
        # spin channels to ensure the same wavelength axis

        _wavelength_keys = [
            "lo_wavelength",
            "hi_wavelength",
            "rebin_percent",
            "wavelength_bins",
        ]

        # For each spin channel, if it is not empty (i.e. channel not
        # measured) then check that its reduction options are the same as down_down
        # reduction options for the keys in _wavelength_keys.

        for sc in self.sc_opts:
            if self.sc_opts[sc]:
                if not general._dict_compare_keys(
                    self.sc_opts["dd"], self.sc_opts[sc], *_wavelength_keys
                ):
                    raise ValueError(
                        "Reduction options `lo_wavelength`, `hi_wavelength`,"
                        " `rebin_percent`, and `wavelength_bins` must be"
                        "identical across spin channels to preserve a common"
                        "wavelength axis."
                    )

        for sc, channel in self.channels.items():
            if channel is None:
                continue
            else:
                channel.process(**self.sc_opts[sc])

    def plot_spectra(self, **kwargs):
        """
        Plots the processed spectrums for each spin state in the SpinSet

        Requires matplotlib to be installed
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set(xlabel="Wavelength ($\\AA$)", ylabel="Intensity (a.u.)")

        for spinch in [self.dd, self.du, self.ud, self.uu]:
            if spinch is None:
                continue
            x = spinch.processed_spectrum["m_lambda"][0]
            y = spinch.processed_spectrum["m_spec"][0]
            yerr = spinch.processed_spectrum["m_spec_sd"][0]
            ax.errorbar(x, y, yerr, label=spinch.cat.sample_name)
        return fig, ax


def basename_datafile(pth):
    """
    Given a NeXUS path return the basename minus the file extension.
    Parameters
    ----------
    pth : str

    Returns
    -------
    basename : str

    Examples
    --------
    >>> basename_datafile('a/b/c.nx.hdf')
    'c'
    """

    basename = os.path.basename(pth)
    return basename.split(".nx.hdf")[0]


def number_datafile(run_number, prefix="PLP"):
    """
    Given a run number figure out what the file name is.
    Given a file name, return the filename with the .nx.hdf extension

    Parameters
    ----------
    run_number : int or str

    prefix : str, optional
        The instrument prefix. Only used if `run_number` is an int

    Returns
    -------
    file_name : str

    Examples
    --------
    >>> number_datafile(708)
    'PLP0000708.nx.hdf'
    >>> number_datafile(708, prefix='QKK')
    'QKK0000708.nx.hdf'
    >>> number_datafile('PLP0000708.nx.hdf')
    'PLP0000708.nx.hdf'
    """
    try:
        num = abs(int(run_number))
        # you got given a run number
        return "{0}{1:07d}.nx.hdf".format(prefix, num)
    except ValueError:
        # you may have been given full filename
        if run_number.endswith(".nx.hdf"):
            return run_number
        else:
            return run_number + ".nx.hdf"


def datafile_number(fname, prefix="PLP"):
    """
    From a filename figure out what the run number was

    Parameters
    ----------
    fname : str
        The filename to be processed

    Returns
    -------
    run_number : int
        The run number

    Examples
    --------
    >>> datafile_number('PLP0000708.nx.hdf')
    708

    """
    rstr = ".*" + prefix + "([0-9]{7}).nx.hdf"
    regex = re.compile(rstr)

    _fname = os.path.basename(fname)
    r = regex.search(_fname)

    if r:
        return int(r.groups()[0])

    return None


class ReductionOptions(dict):
    """
    dict specifying the options for processing a Reflectometry dataset.

    Parameters
    ----------
    h5norm : str or HDF5 NeXus file
        If a str then `h5norm` is a path to the floodfield data, otherwise
        it is a hdf5 file handle containing the floodfield data.
    lo_wavelength : float
        The low wavelength cutoff for the rebinned data (A).
    hi_wavelength : float
        The high wavelength cutoff for the rebinned data (A).
    background : bool
        Should a background subtraction be carried out?
    direct : bool
        Is it a direct beam you measured? This is so a gravity correction
        can be applied (if the instrument needs one).
    omega : float
        Expected angle of incidence of beam. If this is None, then the
        rough angle of incidence is obtained from the NeXus file.
    twotheta : float
        Expected two theta value of specular beam. If this is None then
        the rough angle of incidence is obtained from the NeXus file.
    rebin_percent : float
        Specifies the rebinning percentage for the spectrum.  If
        `rebin_percent is None`, then no rebinning is done.
    wavelength_bins : array_like
        The wavelength bins for rebinning.  If `wavelength_bins is not
        None` then the `rebin_percent` parameter is ignored.
    normalise : bool
        Normalise by the monitor counts.
    integrate : int
        Specifies which scanpoints to use.

         - integrate == -1
           the spectrum is integrated over all the scanpoints.
         - integrate >= 0
           the individual spectra are calculated individually.
           If `eventmode is not None`, or `event_filter is not None` then
           integrate specifies which scanpoint to examine.

    eventmode : None or array_like
        If eventmode is `None` then the integrated detector image is used.
        If eventmode is an array then the array specifies the integration
        times (in seconds) for the detector image, e.g. [0, 20, 30] would
        result in two spectra. The first would contain data for 0 s to 20s,
        the second would contain data for 20 s to 30 s.  This option can
        only be used when `integrate >= -1`.
        If eventmode has zero length (e.g. []), then a single time interval
        for the entire acquisition is used, [0, acquisition_time].  This
        would source the image from the eventmode file, rather than the
        NeXUS file. The two approaches will probably not give
        identical results, because the eventmode method adjusts the total
        acquisition time and beam monitor counts to the frame number of the
        last event detected (which may be quite different if the count rate
        is very low). This parameter is disregarded if `event_filter` is
        provided.
    event_folder : None or str
        Specifies the path for the eventmode data. If
        `event_folder is None` then the eventmode data is assumed to reside
        in the same directory as the NeXUS file. If event_folder is a
        string, then the string specifies the path to the eventmode data.
    peak_pos : -1, None, or (float, float)
        Options for finding specular peak position and peak standard
        deviation.

        - -1
           use `manual_beam_find`.
        - None
           use the automatic beam finder, falling back to
           `manual_beam_find` if it's provided.
        - (float, float)
           specify the peak and peak standard deviation.

    peak_pos_tol : None or (float, float)
        Convergence tolerance for the beam position and width to be
        accepted from successive beam-finder calculations; see the
        `tol` parameter in the `find_specular_ridge` function.
    background_mask : array_like
        An array of bool that specifies which y-pixels to use for
        background subtraction.  Should be the same length as the number of
        y pixels in the detector image.  Otherwise an automatic mask is
        applied (if background is True).
    normalise_bins : bool
        Divides the intensity in each wavelength bin by the width of the
        bin. This allows one to compare spectra even if they were processed
        with different rebin percentages.
    manual_beam_find : callable, optional
        A function which allows the location of the specular ridge to be
        determined. Has the signature `f(detector, detector_err, name)`
        where `detector` and `detector_err` is the detector image and its
        uncertainty, and name is a `str` specifying the name of
        the dataset.
        `detector` and `detector_err` have shape (n, t, {x, y}) where `n`
        is the number of detector images, `t` is the number of
        time-of-flight bins and `x` or `y` is the number of x or y pixels.
        The function should return a tuple,
        `(centre, centre_sd, lopx, hipx, background_pixels)`. `centre`,
        `centre_sd`, `lopx`, `hipx` should be arrays of shape `(n, )`,
        specifying the beam centre, beam width (standard deviation), lowest
        pixel of foreground region, highest pixel of foreground region.
        `background_pixels` is a list of length `n`. Each of the entries
        should contain arrays of pixel numbers that specify the background
        region for each of the detector images.
    event_filter : callable, optional
        A function, that processes the event stream, returning a `detector`
        array, and a `frame_count` array. `detector` has shape
        `(N, T, Y, X)`, where `N` is the number of detector images, `T` is
        the number of time bins (`len(t_bins)`), etc. `frame_count` has
        shape `(N,)` and contains the number of frames for each of the
        detector images. The frame_count is used to determine what fraction
        of the overall monitor counts should be ascribed to each detector
        image (by dividing by the total number of frames). The function has
        signature:

        detector, frame_count = event_filter(loaded_events,
                                             t_bins=None,
                                             y_bins=None,
                                             x_bins=None)

        `loaded_events` is a 4-tuple of numpy arrays:
        `(f_events, t_events, y_events, x_events)`, where `f_events`
        contains the frame number for each neutron, landing at position
        `x_events, y_events` on the detector, with time-of-flight
        `t_events`.
    """

    def __init__(
        self,
        h5norm=None,
        lo_wavelength=2.5,
        hi_wavelength=19.0,
        background=True,
        direct=False,
        omega=None,
        twotheta=None,
        rebin_percent=1.0,
        wavelength_bins=None,
        normalise=True,
        integrate=-1,
        eventmode=None,
        event_folder=None,
        peak_pos=None,
        peak_pos_tol=None,
        background_mask=None,
        normalise_bins=True,
        manual_beam_find=None,
        event_filter=None,
    ):
        super().__init__()
        self["h5norm"] = h5norm
        self["lo_wavelength"] = lo_wavelength
        self["hi_wavelength"] = hi_wavelength
        self["background"] = background
        self["direct"] = direct
        self["omega"] = omega
        self["twotheta"] = twotheta
        self["rebin_percent"] = rebin_percent
        self["wavelength_bins"] = wavelength_bins
        self["normalise"] = normalise
        self["integrate"] = integrate
        self["eventmode"] = eventmode
        self["event_folder"] = event_folder
        self["peak_pos"] = peak_pos
        self["peak_pos_tol"] = peak_pos_tol
        self["background_mask"] = background_mask
        self["normalise_bins"] = normalise_bins
        self["manual_beam_find"] = manual_beam_find
        self["event_filter"] = event_filter


class ReflectNexus:
    def __init__(self):
        self.cat = None

        self.processed_spectrum = dict()

        # _arguments is a dict that contains all the parameters used to call
        # `process`. If the arguments don't change then you shouldn't need to
        # call process again, thereby saving time.
        self._arguments = {}
        self.prefix = None

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif item in self.processed_spectrum:
            return self.processed_spectrum[item]
        else:
            raise AttributeError

    def __getstate__(self):
        dct = self.__dict__
        dct["_arguments"].pop("manual_beam_find")
        return dct

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _short_circuit_process(self, _arguments):
        """
        Returns the truth that two sets of arguments from successive calls to
        the `process` method are the same.

        Parameters
        ----------
        _arguments : dict
            arguments passed to the `process` method

        Returns
        -------
        val : bool
            Truth that __arguments is the same as self.__arguments
        """

        return _dict_compare(_arguments, self._arguments)

    def write_spectrum_dat(self, f, scanpoint=0):
        """
        This method writes a dat representation of the corrected spectrum to
        file.

        Parameters
        ----------
        f : file-like or str
            The file to write the spectrum to, or a str that specifies the file
            name
        scanpoint : int
            Which scanpoint to write

        Returns
        -------
        processed : bool
            If the file hasn't been processed then the `processed is False` and
            vice versa
        """
        if self.processed_spectrum is None:
            return False

        m_lambda = self.processed_spectrum["m_lambda"][scanpoint]
        m_spec = self.processed_spectrum["m_spec"][scanpoint]
        m_spec_sd = self.processed_spectrum["m_spec_sd"][scanpoint]
        m_lambda_fwhm = self.processed_spectrum["m_lambda_fwhm"][scanpoint]

        stacked_data = np.c_[m_lambda, m_spec, m_spec_sd, m_lambda_fwhm]
        np.savetxt(f, stacked_data, delimiter="\t")

        return True

    def write_spectrum_xml(self, f, scanpoint=0):
        """
        This method writes an XML representation of the corrected spectrum to
        file.

        Parameters
        ----------
        f : file-like or str
            The file to write the spectrum to, or a str that specifies the file
            name
        scanpoint : int
            Which scanpoint to write
        """
        if self.processed_spectrum is None:
            return

        s = string.Template(spectrum_template)
        d = dict()
        d["title"] = self.cat.sample_name
        d["time"] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

        m_lambda = self.processed_spectrum["m_lambda"]
        m_spec = self.processed_spectrum["m_spec"]
        m_spec_sd = self.processed_spectrum["m_spec_sd"]
        m_lambda_fwhm = self.processed_spectrum["m_lambda_fwhm"]

        # sort the data
        sorted = np.argsort(self.m_lambda[0])

        r = m_spec[:, sorted]
        lmda = m_lambda[:, sorted]
        dlmda = m_lambda_fwhm[:, sorted]
        dr = m_spec_sd[:, sorted]
        d["n_spectra"] = self.processed_spectrum["n_spectra"]
        d["runnumber"] = "PLP{:07d}".format(self.cat.datafile_number)

        d["r"] = repr(r[scanpoint].tolist()).strip(",[]")
        d["dr"] = repr(dr[scanpoint].tolist()).strip(",[]")
        d["lmda"] = repr(lmda[scanpoint].tolist()).strip(",[]")
        d["dlmda"] = repr(dlmda[scanpoint].tolist()).strip(",[]")
        thefile = s.safe_substitute(d)

        with possibly_open_file(f, "wb") as g:
            if "b" in g.mode:
                thefile = thefile.encode("utf-8")

            g.write(thefile)
            g.truncate()

        return True

    def plot(self, point=0, fig=None):
        """
        Plot a processed spectrum.

        Requires matplotlib be installed.

        Parameters
        ----------
        point: int or sequence, optional
            The spectrum number to be plotted. By default the first spectrum
            will be plotted. Pass `-1` to plot all spectra at once.
        fig: Figure instance, optional
            If `fig` is not supplied then a new figure is created. Otherwise
            the graph is created on the current axes on the supplied figure.

        Returns
        -------
        fig, ax : :class:`matplotlib.Figure`, :class:`matplotlib.Axes`
            `matplotlib` figure and axes objects.

        """
        lam, spec, spec_sd, _ = self.spectrum

        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        if hasattr(point, "len"):
            for p in point:
                ax.plot(lam[p], spec[p])
        elif point == -1:
            for p in range(len(lam)):
                ax.plot(lam[p], spec[p])
        else:
            ax.plot(lam[point], spec[point])

        return fig, ax

    @property
    def spectrum(self):
        return (
            self.processed_spectrum["m_lambda"],
            self.processed_spectrum["m_spec"],
            self.processed_spectrum["m_spec_sd"],
            self.processed_spectrum["m_lambda_fwhm"],
        )

    def detector_average_unwanted_direction(self, detector):
        """
        Averages over non-collimated beam direction
        """
        raise NotImplementedError()

    def create_detector_norm(self, h5norm):
        raise NotImplementedError()

    def beam_divergence(self, scanpoint):
        # works out the beam divergence for a given scan point
        cat = self.cat
        return general.div(
            cat.ss_coll1[scanpoint],
            cat.ss_coll2[scanpoint],
            cat.collimation_distance[0],
        )[0]

    def estimated_beam_width_at_detector(self, scanpoint):
        raise NotImplementedError()

    def phase_angle(self, scanpoint):
        """
        Calculates the phase angle for a given scanpoint

        Parameters
        ----------
        scanpoint : int
            The scanpoint you're interested in

        Returns
        -------
        phase_angle, master_opening : float
            The phase angle and angular opening of the master chopper in
            degrees.
        """
        raise NotImplementedError()

    def time_offset(
        self,
        master_phase_offset,
        master_opening,
        freq,
        phase_angle,
        z0,
        flight_distance,
        tof_hist,
        t_offset=None,
    ):
        raise NotImplementedError()

    def correct_for_gravity(
        self, detector, detector_sd, m_lambda, lo_wavelength, hi_wavelength
    ):
        # default implementation is no gravity correction
        return detector, detector_sd, None

    def process(self, **reduction_options):
        r"""
        Processes the ReflectNexus object to produce a time of flight spectrum.
        The processed spectrum is stored in the `processed_spectrum` attribute.
        The specular spectrum is also returned from this function.

        Parameters
        ----------
        h5norm : str or HDF5 NeXus file
            If a str then `h5norm` is a path to the floodfield data, otherwise
            it is a hdf5 file handle containing the floodfield data.
        lo_wavelength : float
            The low wavelength cutoff for the rebinned data (A).
        hi_wavelength : float
            The high wavelength cutoff for the rebinned data (A).
        background : bool
            Should a background subtraction be carried out?
        direct : bool
            Is it a direct beam you measured? This is so a gravity correction
            can be applied (if the instrument needs one).
        omega : float
            Expected angle of incidence of beam. If this is None, then the
            rough angle of incidence is obtained from the NeXus file.
        twotheta : float
            Expected two theta value of specular beam. If this is None then
            the rough angle of incidence is obtained from the NeXus file.
        rebin_percent : float
            Specifies the rebinning percentage for the spectrum.  If
            `rebin_percent is None`, then no rebinning is done.
        wavelength_bins : array_like
            The wavelength bins for rebinning.  If `wavelength_bins is not
            None` then the `rebin_percent` parameter is ignored.
        normalise : bool
            Normalise by the monitor counts.
        integrate : int
            Specifies which scanpoints to use.

             - integrate == -1
               the spectrum is integrated over all the scanpoints.
             - integrate >= 0
               the individual spectra are calculated individually.
               If `eventmode is not None`, or `event_filter is not None` then
               integrate specifies which scanpoint to examine.

        eventmode : None or array_like
            If eventmode is `None` then the integrated detector image is used.
            If eventmode is an array then the array specifies the integration
            times (in seconds) for the detector image, e.g. [0, 20, 30] would
            result in two spectra. The first would contain data for 0 s to 20s,
            the second would contain data for 20 s to 30 s.  This option can
            only be used when `integrate >= -1`.
            If eventmode has zero length (e.g. []), then a single time interval
            for the entire acquisition is used, [0, acquisition_time].  This
            would source the image from the eventmode file, rather than the
            NeXUS file. The two approaches will probably not give
            identical results, because the eventmode method adjusts the total
            acquisition time and beam monitor counts to the frame number of the
            last event detected (which may be quite different if the count rate
            is very low). This parameter is disregarded if `event_filter` is
            provided.
        event_folder : None or str
            Specifies the path for the eventmode data. If
            `event_folder is None` then the eventmode data is assumed to reside
            in the same directory as the NeXUS file. If event_folder is a
            string, then the string specifies the path to the eventmode data.
        peak_pos : -1, None, or (float, float)
            Options for finding specular peak position and peak standard
            deviation.

            - -1
               use `manual_beam_find`.
            - None
               use the automatic beam finder, falling back to
               `manual_beam_find` if it's provided.
            - (float, float)
               specify the peak and peak standard deviation.

        peak_pos_tol : (float, float) or None
            Convergence tolerance for the beam position and width to be
            accepted from successive beam-finder calculations; see the
            `tol` parameter in the `find_specular_ridge` function.
        background_mask : array_like
            An array of bool that specifies which y-pixels to use for
            background subtraction.  Should be the same length as the number of
            y pixels in the detector image.  Otherwise an automatic mask is
            applied (if background is True).
        normalise_bins : bool
            Divides the intensity in each wavelength bin by the width of the
            bin. This allows one to compare spectra even if they were processed
            with different rebin percentages.
        manual_beam_find : callable, optional
            A function which allows the location of the specular ridge to be
            determined. Has the signature `f(detector, detector_err, name)`
            where `detector` and `detector_err` is the detector image and its
            uncertainty, and name is a `str` specifying the name of
            the dataset.
            `detector` and `detector_err` have shape (n, t, {x, y}) where `n`
            is the number of detector images, `t` is the number of
            time-of-flight bins and `x` or `y` is the number of x or y pixels.
            The function should return a tuple,
            `(centre, centre_sd, lopx, hipx, background_pixels)`. `centre`,
            `centre_sd`, `lopx`, `hipx` should be arrays of shape `(n, )`,
            specifying the beam centre, beam width (standard deviation), lowest
            pixel of foreground region, highest pixel of foreground region.
            `background_pixels` is a list of length `n`. Each of the entries
            should contain arrays of pixel numbers that specify the background
            region for each of the detector images.
        event_filter : callable, optional
            A function, that processes the event stream, returning a `detector`
            array, and a `frame_count` array. `detector` has shape
            `(N, T, Y, X)`, where `N` is the number of detector images, `T` is
            the number of time bins (`len(t_bins)`), etc. `frame_count` has
            shape `(N,)` and contains the number of frames for each of the
            detector images. The frame_count is used to determine what fraction
            of the overall monitor counts should be ascribed to each detector
            image (by dividing by the total number of frames). The function has
            signature:

            detector, frame_count = event_filter(loaded_events,
                                                 t_bins=None,
                                                 y_bins=None,
                                                 x_bins=None)

            `loaded_events` is a 4-tuple of numpy arrays:
            `(f_events, t_events, y_events, x_events)`, where `f_events`
            contains the frame number for each neutron, landing at position
            `x_events, y_events` on the detector, with time-of-flight
            `t_events`.

        Notes
        -----
        After processing this object contains the following attributes:

        - path - path to the data file
        - datafilename - name of the datafile
        - datafile_number - datafile number.
        - m_topandtail - the corrected 2D detector image,
                         (n_spectra, TOF, {X, Y})
        - m_topandtail_sd - corresponding standard deviations
        - n_spectra - number of spectra in processed data
        - bm1_counts - beam montor counts, (n_spectra,)
        - m_spec - specular intensity, (n_spectra, TOF)
        - m_spec_sd - corresponding standard deviations
        - m_beampos - beam_centre for each spectrum, (n_spectra, )
        - m_lambda - wavelengths for each spectrum, (n_spectra, TOF)
        - m_lambda_fwhm - corresponding FWHM of wavelength distribution
        - m_lambda_hist - wavelength bins for each spectrum,
                          (n_spectra, TOF + 1)
        - m_spec_tof - TOF for each wavelength bin, (n_spectra, TOF)
        - mode - the experimental mode, e.g. FOC/MT/POL/POLANAL/SB/DB
        - detector_z - detector height or angle, (n_spectra, )
        - detector_y - sample-detector distance, (n_spectra, )
        - domega - collimation uncertainty
        - lopx - lowest extent of specular beam (in pixels), (n_spectra, )
        - hipx - highest extent of specular beam (in pixels), (n_spectra, )
        - reduction_options - dict of options used to process the spectra

        Returns
        -------
        m_lambda, m_spec, m_spec_sd: np.ndarray
            Arrays containing the wavelength, specular intensity as a function
            of wavelength, standard deviation of specular intensity

        """
        options = ReductionOptions()
        options.update(reduction_options)

        h5norm = options["h5norm"]
        lo_wavelength = options["lo_wavelength"]
        hi_wavelength = options["hi_wavelength"]
        background = options["background"]
        direct = options["direct"]
        omega = options["omega"]
        twotheta = options["twotheta"]
        rebin_percent = options["rebin_percent"]
        wavelength_bins = options["wavelength_bins"]
        normalise = options["normalise"]
        integrate = options["integrate"]
        eventmode = options["eventmode"]
        event_folder = options["event_folder"]
        peak_pos = options["peak_pos"]
        peak_pos_tol = options["peak_pos_tol"]
        background_mask = options["background_mask"]
        normalise_bins = options["normalise_bins"]
        manual_beam_find = options["manual_beam_find"]
        event_filter = options["event_filter"]

        # it can be advantageous to save processing time if the arguments
        # haven't changed
        # if you've already processed, then you may not need to process again
        if self.processed_spectrum and self._short_circuit_process(options):
            return (
                self.processed_spectrum["m_lambda"],
                self.processed_spectrum["m_spec"],
                self.processed_spectrum["m_spec_sd"],
            )
        else:
            self._arguments = options

        cat = self.cat

        scanpoint = 0

        # beam monitor counts for normalising data
        bm1_counts = cat.bm1_counts.astype("float64")

        # TOF bins
        TOF = cat.t_bins.astype("float64")

        # This section controls how multiple detector images are handled.
        # We want event streaming.
        if eventmode is not None or event_filter is not None:
            scanpoint = integrate
            if integrate == -1:
                scanpoint = 0

            output = self.process_event_stream(
                scanpoint=scanpoint,
                frame_bins=eventmode,
                event_folder=event_folder,
                event_filter=event_filter,
            )
            detector, frame_count, bm1_counts = output

            start_time = np.zeros(np.size(detector, 0))
            if cat.start_time is not None:
                start_time += cat.start_time[scanpoint]
                start_time[1:] += (
                    np.cumsum(frame_count)[:-1] / cat.frequency[scanpoint]
                )
        else:
            # we don't want detector streaming
            detector = cat.detector
            scanpoint = 0

            # integrate over all spectra
            if integrate == -1:
                detector = np.sum(detector, 0)[
                    np.newaxis,
                ]
                bm1_counts[:] = np.sum(bm1_counts)

            start_time = np.zeros(np.size(detector, 0))
            if cat.start_time is not None:
                for idx in range(start_time.size):
                    start_time[idx] = cat.start_time[idx]

        n_spectra = np.size(detector, 0)

        # Up until this point detector.shape=(N, T, Y, X)
        # average to (N, T, Y) - platypus or (N, T, X) - spatz
        detector = self.detector_average_unwanted_direction(detector)

        # calculate the counting uncertainties
        detector_sd = np.sqrt(detector)
        bm1_counts_sd = np.sqrt(bm1_counts)

        # detector normalisation with a water file
        if h5norm is not None:
            with _possibly_open_hdf_file(h5norm, "r") as f:
                # shape ({x, y},)
                detector_norm, detector_norm_sd = self.create_detector_norm(f)

            # detector has shape (N, T, Y), shape of detector_norm should
            # broadcast to (1, 1, y)
            # TODO: Correlated Uncertainties?
            detector, detector_sd = EP.EPdiv(
                detector, detector_sd, detector_norm, detector_norm_sd
            )

        # shape of these is (n_spectra, TOFbins)
        m_spec_tof_hist = np.zeros(
            (n_spectra, np.size(TOF, 0)), dtype="float64"
        )
        m_lambda_hist = np.zeros((n_spectra, np.size(TOF, 0)), dtype="float64")
        m_spec_tof_hist[:] = TOF

        """
        chopper to detector distances
        note that if eventmode is specified the n_spectra is NOT
        equal to the number of entries in e.g. /longitudinal_translation
        this means you have to copy values in from the correct scanpoint
        """
        flight_distance = np.zeros(n_spectra, dtype="float64")
        d_cx = np.zeros(n_spectra, dtype="float64")
        detpositions = np.zeros(n_spectra, dtype="float64")

        # The angular divergence of the instrument
        domega = np.zeros(n_spectra, dtype="float64")
        estimated_beam_width = np.zeros(n_spectra, dtype="float64")

        phase_angle = np.zeros(n_spectra, dtype="float64")

        # process each of the spectra taken in the detector image
        original_scanpoint = scanpoint
        for idx in range(n_spectra):
            freq = cat.frequency[scanpoint]

            # calculate the angular divergence
            domega[idx] = self.beam_divergence(scanpoint)

            """
            estimated beam width in pixels at detector
            """
            estimated_beam_width[idx] = self.estimated_beam_width_at_detector(
                scanpoint
            )

            """
            work out the total flight length
            IMPORTANT: this varies as a function of twotheta. This is
            because the Platypus detector does not move on an arc.
            At high angles chod can be ~ 0.75% different. This is will
            visibly shift fringes.
            """
            if omega is None:
                omega = cat.omega[scanpoint]
            if twotheta is None:
                twotheta = cat.twotheta[scanpoint]
            output = self.chod(omega, twotheta, scanpoint=scanpoint)
            flight_distance[idx], d_cx[idx] = output

            # calculate nominal phase openings
            phase_angle[idx], master_opening = self.phase_angle(scanpoint)

            """
            toffset - the time difference between the magnet pickup on the
            choppers (TTL pulse), which is situated in the middle of the
            chopper window, and the trailing edge of master chopper, which is
            supposed to be time0.  However, if there is a phase opening this
            time offset has to be relocated slightly, as time0 is not at the
            trailing edge.
            """
            t_offset = self.time_offset(
                cat.master_phase_offset[0],
                master_opening,
                freq,
                phase_angle[idx],
                d_cx[0],
                flight_distance[idx],
                m_spec_tof_hist[idx],
                t_offset=cat.t_offset,
            )

            m_spec_tof_hist[idx] -= t_offset

            detpositions[idx] = cat.dy[scanpoint]

            if eventmode is not None or event_filter is not None:
                m_spec_tof_hist[:] = TOF - t_offset
                flight_distance[:] = flight_distance[0]
                detpositions[:] = detpositions[0]
                domega[:] = domega[0]
                estimated_beam_width[:] = estimated_beam_width[0]
                d_cx[:] = d_cx[0]
                phase_angle[:] = phase_angle[0]
                break
            else:
                scanpoint += 1

        scanpoint = original_scanpoint

        # convert TOF to lambda
        # m_spec_tof_hist (n, t) and chod is (n,)
        m_lambda_hist = general.velocity_wavelength(
            1.0e3 * flight_distance[:, np.newaxis] / m_spec_tof_hist
        )

        m_lambda = 0.5 * (m_lambda_hist[:, 1:] + m_lambda_hist[:, :-1])
        TOF -= t_offset

        # gravity correction if direct beam
        if direct:
            # TODO: Correlated Uncertainties?
            detector, detector_sd, m_gravcorrcoefs = self.correct_for_gravity(
                detector, detector_sd, m_lambda, lo_wavelength, hi_wavelength
            )

        # where is the specular ridge?
        if peak_pos == -1:
            # you always want to find the beam manually
            ret = manual_beam_find(
                detector, detector_sd, os.path.basename(cat.filename)
            )
            beam_centre, beam_sd, lopx, hipx, bp = ret

            full_backgnd_mask = np.zeros_like(detector, dtype=bool)
            for i, v in enumerate(bp):
                full_backgnd_mask[i, :, v] = True

        elif peak_pos is None:
            # absolute tolerance in beam pixel position for auto peak finding
            # derived as a fraction of detector pixel size. 0.0142 mm at
            # dy = 2512 corresponds to 0.0003 degrees.
            try:
                atol, rtol = peak_pos_tol
            except (ValueError, TypeError):
                # TypeError for unpacking None (currently the default option)
                # ValueError for unpacking a single number (historical
                # behaviour)
                atol = 0.0142 / self.cat.qz_pixel_size[0]
                rtol = 0.015

            # use the auto finder, falling back to manual_beam_find
            ret = find_specular_ridge(
                detector,
                detector_sd,
                tol=(atol, rtol),
                manual_beam_find=manual_beam_find,
                name=os.path.basename(cat.filename),
            )
            beam_centre, beam_sd, lopx, hipx, full_backgnd_mask = ret
        else:
            # the specular ridge has been specified
            beam_centre = np.ones(n_spectra) * peak_pos[0]
            beam_sd = np.ones(n_spectra) * peak_pos[1]
            lopx, hipx, bp = fore_back_region(beam_centre, beam_sd)

            full_backgnd_mask = np.zeros_like(detector, dtype=bool)
            for i, v in enumerate(bp):
                full_backgnd_mask[i, :, v] = True

        lopx = lopx.astype(int)
        hipx = hipx.astype(int)

        # Warning if the beam appears to be much broader than the divergence
        # would predict. The use of 30% tolerance is a guess. This might happen
        # if the beam finder includes incoherent background region by mistake.
        if not np.allclose(estimated_beam_width, hipx - lopx + 1, rtol=0.3):
            warnings.warn(
                "The foreground width (%s) estimate"
                " does not match the divergence of the beam (%s)."
                " Consider checking with manual beam finder."
                % (str(hipx - lopx + 1), str(estimated_beam_width))
            )

        if np.size(beam_centre) != n_spectra:
            raise RuntimeError(
                "The number of beam centres should be equal"
                " to the number of detector images."
            )

        """
        Rebinning in lambda for all detector
        Rebinning is the default option, but sometimes you don't want to.
        detector shape input is (n, t, y)
        """
        if wavelength_bins is not None:
            rebinning = wavelength_bins
        elif 0.0 < rebin_percent < 15.0:
            rebinning = calculate_wavelength_bins(
                lo_wavelength, hi_wavelength, rebin_percent
            )

        # rebin_percent percentage is zero. No rebinning, just cutoff
        # wavelength
        else:
            rebinning = m_lambda_hist[0, :]
            rebinning = rebinning[
                np.searchsorted(rebinning, lo_wavelength) : np.searchsorted(
                    rebinning, hi_wavelength
                )
            ]

        """
        now do the rebinning for all the N detector images
        rebin.rebinND could do all of these at once.  However, m_lambda_hist
        could vary across the range of spectra.  If it was the same I could
        eliminate the loop.
        """
        output = []
        output_sd = []
        for idx in range(n_spectra):
            # TODO: Correlated Uncertainties?
            plane, plane_sd = rebin_along_axis(
                detector[idx],
                m_lambda_hist[idx],
                rebinning,
                y1_sd=detector_sd[idx],
            )
            output.append(plane)
            output_sd.append(plane_sd)

        detector = np.array(output)
        detector_sd = np.array(output_sd)

        if len(detector.shape) == 2:
            detector = detector[
                np.newaxis,
            ]
            detector_sd = detector_sd[
                np.newaxis,
            ]

        # (1, T)
        m_lambda_hist = np.atleast_2d(rebinning)

        """
        Divide the detector intensities by the width of the wavelength bin.
        This is so the intensities between different rebinning strategies can
        be compared.
        """
        if normalise_bins:
            div = 1 / np.diff(m_lambda_hist[0])[:, np.newaxis]
            detector, detector_sd = EP.EPmulk(detector, detector_sd, div)

        # convert the wavelength base to a timebase
        m_spec_tof_hist = (
            0.001
            * flight_distance[:, np.newaxis]
            / general.wavelength_velocity(m_lambda_hist)
        )

        m_lambda = 0.5 * (m_lambda_hist[:, 1:] + m_lambda_hist[:, :-1])

        m_spec_tof = (
            0.001
            * flight_distance[:, np.newaxis]
            / general.wavelength_velocity(m_lambda)
        )

        m_spec = np.zeros((n_spectra, np.size(detector, 1)))
        m_spec_sd = np.zeros_like(m_spec)

        # background subtraction
        if background:
            if background_mask is not None:
                # background_mask is (Y), need to make 3 dimensional (N, T, Y)
                # first make into (T, Y)
                backgnd_mask = np.repeat(
                    background_mask[np.newaxis, :], detector.shape[1], axis=0
                )
                # make into (N, T, Y)
                full_backgnd_mask = np.repeat(
                    backgnd_mask[np.newaxis, :], n_spectra, axis=0
                )

            # TODO: Correlated Uncertainties?
            detector, detector_sd = background_subtract(
                detector, detector_sd, full_backgnd_mask
            )

        """
        top and tail the specular beam with the known beam centres.
        All this does is produce a specular intensity with shape (N, T),
        i.e. integrate over specular beam
        """
        for i in range(n_spectra):
            m_spec[i] = np.sum(detector[i, :, lopx[i] : hipx[i] + 1], axis=1)
            sd = np.sum(detector_sd[i, :, lopx[i] : hipx[i] + 1] ** 2, axis=1)
            m_spec_sd[i] = np.sqrt(sd)

        # assert np.isfinite(m_spec).all()
        # assert np.isfinite(m_specSD).all()
        # assert np.isfinite(detector).all()
        # assert np.isfinite(detectorSD).all()

        # normalise by beam monitor 1.
        if normalise:
            m_spec, m_spec_sd = EP.EPdiv(
                m_spec,
                m_spec_sd,
                bm1_counts[:, np.newaxis],
                bm1_counts_sd[:, np.newaxis],
            )

            output = EP.EPdiv(
                detector,
                detector_sd,
                bm1_counts[:, np.newaxis, np.newaxis],
                bm1_counts_sd[:, np.newaxis, np.newaxis],
            )
            detector, detector_sd = output

        """
        now work out dlambda/lambda, the resolution contribution from
        wavelength.
        van Well, Physica B,  357(2005) pp204-207), eqn 4.
        this is only an approximation for our instrument, as the 2nd and 3rd
        discs have smaller openings compared to the master chopper.
        Therefore the burst time needs to be looked at.
        """
        tau_da = m_spec_tof_hist[:, 1:] - m_spec_tof_hist[:, :-1]

        m_lambda_fwhm = resolution_double_chopper(
            m_lambda,
            z0=d_cx[:, np.newaxis] / 1000.0,
            freq=cat.frequency[:, np.newaxis],
            L=flight_distance[:, np.newaxis] / 1000.0,
            H=cat.ss_coll2[original_scanpoint] / 1000.0,
            xsi=phase_angle[:, np.newaxis],
            tau_da=tau_da,
        )
        m_lambda_fwhm *= m_lambda

        # put the detector positions and mode into the dictionary as well.
        detector_z = cat.dz
        detector_y = cat.dy

        try:
            mode = cat.mode
        except AttributeError:
            # no mode for SPZ
            mode = None

        d = dict()
        d["path"] = cat.path
        d["datafilename"] = cat.filename
        d["datafile_number"] = cat.datafile_number

        if h5norm is not None:
            if type(h5norm) == h5py.File:
                d["normfilename"] = h5norm.filename
            else:
                d["normfilename"] = h5norm
        d["m_topandtail"] = detector
        d["m_topandtail_sd"] = detector_sd
        d["n_spectra"] = n_spectra
        d["bm1_counts"] = bm1_counts
        d["m_spec"] = m_spec
        d["m_spec_sd"] = m_spec_sd
        d["m_beampos"] = beam_centre
        d["m_beampos_sd"] = beam_sd
        d["m_lambda"] = m_lambda
        d["m_lambda_fwhm"] = m_lambda_fwhm
        d["m_lambda_hist"] = m_lambda_hist
        d["m_spec_tof"] = m_spec_tof
        d["mode"] = mode
        d["detector_z"] = detector_z
        d["detector_y"] = detector_y
        d["domega"] = domega
        d["lopx"] = lopx
        d["hipx"] = hipx
        d["start_time"] = start_time
        d["reduction_options"] = options

        self.processed_spectrum = d
        return m_lambda, m_spec, m_spec_sd

    def process_event_stream(
        self,
        t_bins=None,
        x_bins=None,
        y_bins=None,
        frame_bins=None,
        scanpoint=0,
        event_folder=None,
        event_filter=None,
    ):
        """
        Processes the event mode dataset for the NeXUS file. Assumes that
        there is a event mode directory in the same directory as the NeXUS
        file, as specified by in 'entry1/instrument/detector/daq_dirname'

        Parameters
        ----------
        t_bins : array_like, optional
            specifies the time bins required in the image
        x_bins : array_like, optional
            specifies the x bins required in the image
        y_bins : array_like, optional
            specifies the y bins required in the image
        scanpoint : int, optional
            Scanpoint you are interested in
        event_folder : None or str
            Specifies the path for the eventmode data. If
            `event_folder is None` then the eventmode data is assumed to reside
            in the same directory as the NeXUS file. If event_folder is a
            string, then the string specifies the path to the eventmode data.
        frame_bins : array_like, optional
            specifies the frame bins required in the image. If
            frame_bins = [5, 10, 120] you will get 2 images.  The first starts
            at 5s and finishes at 10s. The second starts at 10s and finishes
            at 120s. If frame_bins has zero length, e.g. [], then a single
            interval consisting of the entire acquisition time is used:
            [0, acquisition_time]. If `event_filter` is provided then this
            parameter is ignored.
        event_filter : callable, optional
            A function, that processes the event stream, returning a `detector`
            array, and a `frame_count` array. `detector` has shape
            `(N, T, Y, X)`, where `N` is the number of detector images, `T` is
            the number of time bins (`len(t_bins)`), etc. `frame_count` has
            shape `(N,)` and contains the number of frames for each of the
            detector images. The frame_count is used to determine what fraction
            of the overall monitor counts should be ascribed to each detector
            image. The function has signature:

            detector, frame_count = event_filter(loaded_events,
                                                 t_bins=None,
                                                 y_bins=None,
                                                 x_bins=None)

            `loaded_events` is a 4-tuple of numpy arrays:
            `(f_events, t_events, y_events, x_events)`, where `f_events`
            contains the frame number for each neutron, landing at position
            `x_events, y_events` on the detector, with time-of-flight
            `t_events`.

        Returns
        -------
        detector, frame_count, bm1_counts : np.ndarray, np.ndarray, np.ndarray

        Create a new detector image based on the t_bins, x_bins, y_bins and
        frame_bins you supply to the method (these should all be lists/numpy
        arrays specifying the edges of the required bins). If these are not
        specified, then the default bins are taken from the nexus file. This
        would essentially return the same detector image as the nexus file.
        However, you can specify the frame_bins list to generate detector
        images based on subdivided periods of the total acquisition.
        For example if frame_bins = [5, 10, 120] you will get 2 images.  The
        first starts at 5s and finishes at 10s. The second starts at 10s
        and finishes at 120s. The frame_bins are clipped to the total
        acquisition time if necessary.
        `frame_count` is how many frames went into making each detector image.

        """
        cat = self.cat

        if not t_bins:
            t_bins = cat.t_bins
        if not y_bins:
            y_bins = cat.y_bins
        if not x_bins:
            x_bins = cat.x_bins
        if frame_bins is None or np.size(frame_bins) == 0:
            frame_bins = [0, cat.time[scanpoint]]

        total_acquisition_time = cat.time[scanpoint]
        frequency = cat.frequency[scanpoint]

        bm1_counts_for_scanpoint = cat.bm1_counts[scanpoint]

        event_directory_name = cat.daq_dirname

        _eventpath = cat.path
        if event_folder is not None:
            _eventpath = event_folder

        stream_filename = os.path.join(
            _eventpath,
            event_directory_name,
            f"DATASET_{scanpoint}",
            "EOS.bin",
        )

        with io.open(stream_filename, "rb") as f:
            last_frame = int(frame_bins[-1] * frequency)
            loaded_events, end_events = events(f, max_frames=last_frame)

        # convert frame_bins to list of filter frames
        frames = framebins_to_frames(np.asfarray(frame_bins) * frequency)

        if event_filter is not None:
            output = event_filter(loaded_events, t_bins, y_bins, x_bins)
        else:
            output = process_event_stream(
                loaded_events, frames, t_bins, y_bins, x_bins
            )

        detector, frame_count = output

        bm1_counts = (
            frame_count
            * bm1_counts_for_scanpoint
            / total_acquisition_time
            / frequency
        )

        return detector, frame_count, bm1_counts


class PlatypusNexus(ReflectNexus):
    """
    Processes Platypus NeXus files to produce an intensity vs wavelength
    spectrum

    Parameters
    ----------
    h5data : HDF5 NeXus file or str
        An HDF5 NeXus file for Platypus, or a `str` containing the path
        to one
    """

    def __init__(self, h5data):
        """
        Initialises the PlatypusNexus object.
        """
        super().__init__()
        self.prefix = "PLP"
        with _possibly_open_hdf_file(h5data, "r") as f:
            self.cat = PlatypusCatalogue(f)
            if self.cat.mode in ["POL", "POLANAL"]:
                self.cat = PolarisedCatalogue(f)

                # Set spin channels based of flipper statuses
                if self.cat.mode == "POL":
                    if self.cat.pol_flip_current > 0.1:
                        self.spin_state = SpinChannel.UP_UP
                    else:
                        self.spin_state = SpinChannel.DOWN_DOWN

                if self.cat.mode == "POLANAL":
                    if (
                        self.cat.pol_flip_current > 0.1
                        and self.cat.anal_flip_current > 0.1
                    ):
                        self.spin_state = SpinChannel.UP_UP
                    elif (
                        self.cat.pol_flip_current > 0.1
                        and self.cat.anal_flip_current < 0.1
                    ):
                        self.spin_state = SpinChannel.UP_DOWN
                    elif (
                        self.cat.pol_flip_current < 0.1
                        and self.cat.anal_flip_current > 0.1
                    ):
                        self.spin_state = SpinChannel.DOWN_UP
                    elif (
                        self.cat.pol_flip_current < 0.1
                        and self.cat.anal_flip_current < 0.1
                    ):
                        self.spin_state = SpinChannel.DOWN_DOWN

    def detector_average_unwanted_direction(self, detector):
        """
        Averages over non-collimated beam direction
        """
        # Up until this point detector.shape=(N, T, Y, X)
        # pre-average over x, leaving (n, t, y) also convert to dp
        return np.sum(detector, axis=3, dtype="float64")

    def create_detector_norm(self, h5norm):
        """
        Produces a detector normalisation array for a neutron detector.
        Here we average over N, T and X to provide a relative efficiency for
        each y wire.

        Parameters
        ----------
        h5norm : hdf5 file
            Containing a flood field run (water)

        Returns
        -------
        norm, norm_sd : array_like
            1D array containing the normalisation data for each y pixel
        """
        x_bins = self.cat.x_bins
        return create_detector_norm(h5norm, x_bins[0], x_bins[1], axis=3)

    def estimated_beam_width_at_detector(self, scanpoint):
        cat = self.cat
        L23 = cat.cat["collimation_distance"]
        L3det = (
            cat.dy[scanpoint] + cat.sample_distance[0] - cat.slit3_distance[0]
        )
        ebw = general.height_of_beam_after_dx(
            cat.ss2vg[scanpoint], cat.ss3vg[scanpoint], L23, L3det
        )
        umb, penumb = ebw
        # convolve in detector resolution (~2.2 mm?)
        # first convert to beam sd, convolve in detector, and expand sd
        # back to total foreground width
        # use average of umb and penumb, the calc assumes a rectangular
        # distribution
        penumb = (
            np.sqrt((0.289 * 0.5 * (umb + penumb)) ** 2.0 + 2.2**2)
            * EXTENT_MULT
            * 2
        )
        # we need it in pixels
        return penumb / cat.qz_pixel_size[0]

    def correct_for_gravity(
        self, detector, detector_sd, m_lambda, lo_wavelength, hi_wavelength
    ):
        cat = self.cat

        return correct_for_gravity(
            detector,
            detector_sd,
            m_lambda,
            cat.collimation_distance,
            cat.dy,
            lo_wavelength,
            hi_wavelength,
            qz_pixel_size=cat.qz_pixel_size[0],
        )

    def time_offset(
        self,
        master_phase_offset,
        master_opening,
        freq,
        phase_angle,
        z0,
        flight_distance,
        tof_hist,
        t_offset=None,
    ):
        """
        Timing offsets for Platypus chopper system, includes a gravity
        correction for phase angle
        """
        DISCRADIUS = 350.0

        # calculate initial time offset from the pickup being slightly in
        # the wrong place
        m_offset = 1.0e6 * master_phase_offset / (2.0 * 360.0 * freq)
        # make a correction for the phase angle
        total_offset = m_offset - 1.0e6 * phase_angle / (360 * 2 * freq)

        # assumes that the pickup/T_0 signal is issued from middle
        # of chopper window. But you can override this by supplying a t_offset.
        # This is for where a signal generator has been used to offset that t_0
        if t_offset is not None:
            total_offset += t_offset
        else:
            total_offset += 1.0e6 * master_opening / (2 * 360 * freq)

        ###########################################
        # now make a gravity correction to total_offset
        # work out velocities for each bin edge
        velocities = 1.0e3 * flight_distance / (tof_hist - total_offset)

        angles = find_trajectory(
            self.cat.collimation_distance / 1000.0, 0, velocities
        )

        # work out distance from 1st coll slit to middle of chopper pair
        # TODO ASSUMES CHOPPER 1 IS MASTER, FIX SO IT COULD BE ANY MASTER
        d_c1 = -self.cat.slit2_distance
        d_slave = d_c1 + z0

        corr_t_offset = np.zeros_like(tof_hist)

        # assumes that the pickups/T_0 signal is issued from middle
        # of chopper window. `t_offset` is for where a signal generator
        # has been used to offset that t_0.
        if t_offset is not None:
            corr_t_offset += t_offset / 1.0e6
        else:
            corr_t_offset += master_opening / (2 * 360 * freq)

        for i, (velocity, angle) in enumerate(zip(velocities, angles)):
            parab = parabola(angle, velocity)
            h_c1 = parab(d_c1 / 1000.0) * 1000.0
            h_slave = parab(d_slave / 1000.0) * 1000.0
            # angle_corr comes about because the parabolic nature of long
            # wavelength neutrons creates an apparent phase opening
            angle_corr = np.degrees(np.arctan((h_slave - h_c1) / DISCRADIUS))
            # master_corr comes about because the beam for long wavelength
            # neutrons is lower than the optical axis, so it takes a little
            # longer for the beam to start travelling through chopper system.
            # as such you need to decrease their tof by increasing the
            # t_offset
            master_corr = -np.degrees(np.arctan(h_c1 / DISCRADIUS))
            corr_t_offset[i] += (master_phase_offset + master_corr) / (
                2.0 * 360.0 * freq
            )
            corr_t_offset[i] -= (phase_angle + angle_corr) / (360 * 2 * freq)
        corr_t_offset *= 1e6

        return corr_t_offset

    def phase_angle(self, scanpoint=0):
        """
        Calculates the phase angle for a given scanpoint

        Parameters
        ----------
        scanpoint : int
            The scanpoint you're interested in

        Returns
        -------
        phase_angle, master_opening : float
            The phase angle and angular opening of the master chopper in
            degrees.
        """
        disc_openings = (60.0, 10.0, 25.0, 60.0)
        O_C1d, O_C2d, O_C3d, O_C4d = disc_openings

        cat = self.cat
        master = cat.master
        slave = cat.slave
        disc_phase = cat.phase[scanpoint]
        phase_angle = 0

        if master == 1:
            phase_angle += 0.5 * O_C1d
            master_opening = O_C1d
        elif master == 2:
            phase_angle += 0.5 * O_C2d
            master_opening = O_C2d
        elif master == 3:
            phase_angle += 0.5 * O_C3d
            master_opening = O_C3d

        # the phase_offset is defined as the angle you have to add to the
        # calibrated blind opening to get to the nominal optically blind
        # chopper opening.
        # e.g. Nominal opening for optically may be at 42.5 degrees
        # but the calibrated optically blind position is 42.2 degrees
        # the chopper_phase_offset would be 0.3 degrees.
        if slave == 2:
            phase_angle += 0.5 * O_C2d
            phase_angle += -disc_phase - cat.chopper2_phase_offset[0]
        elif slave == 3:
            phase_angle += 0.5 * O_C3d
            phase_angle += -disc_phase - cat.chopper3_phase_offset[0]
        elif slave == 4:
            phase_angle += 0.5 * O_C4d
            phase_angle += disc_phase - cat.chopper4_phase_offset[0]

        return phase_angle, master_opening

    def chod(self, omega=0.0, twotheta=0.0, scanpoint=0):
        """
        Calculates the flight length of the neutrons in the Platypus
        instrument.

        Parameters
        ----------
        omega : float, optional
            Rough angle of incidence
        twotheta : float, optional
            Rough 2 theta angle
        scanpoint : int, optional
            Which dataset is being considered

        Returns
        -------
        chod, d_cx : float, float
            Flight distance (mm), distance between chopper discs (mm)
        """

        chod = 0

        # guide 1 is the single deflection mirror (SB)
        # its distance is from chopper 1 to the middle of the mirror (1m long)
        # guide 2 is the double deflection mirror (DB)
        # its distance is from chopper 1 to the middle of the second of the
        # compound mirrors! (a bit weird, I know).

        cat = self.cat
        mode = cat.mode

        # Find out chopper pairing
        master = cat.master
        slave = cat.slave

        d_cx = 0

        if master == 1:
            chod = 0
        elif master == 2:
            chod -= cat.chopper2_distance[0]
            d_cx -= chod
        elif master == 3:
            chod -= cat.chopper3_distance[0]
            d_cx -= chod
        else:
            raise ValueError(
                "Chopper pairing should be one of '12', '13',"
                "'14', '23', '24', '34'"
            )

        if slave == 2:
            chod -= cat.chopper2_distance[0]
            d_cx += cat.chopper2_distance[0]
        elif slave == 3:
            chod -= cat.chopper3_distance[0]
            d_cx += cat.chopper3_distance[0]
        elif slave == 4:
            chod -= cat.chopper4_distance[0]
            d_cx += cat.chopper4_distance[0]

        # start of flight length is midway between master and slave, but master
        # may not necessarily be disk 1. However, all instrument lengths are
        # measured from disk1
        chod /= 2.0

        if mode in ["FOC", "POL", "MT", "POLANAL"]:
            chod += cat.sample_distance[0]
            chod += cat.dy[scanpoint] / np.cos(np.radians(twotheta))

        elif mode == "SB":
            # assumes guide1_distance is in the MIDDLE OF THE MIRROR
            chod += cat.guide1_distance[0]
            chod += (cat.sample_distance[0] - cat.guide1_distance[0]) / np.cos(
                np.radians(omega)
            )
            if twotheta > omega:
                chod += cat.dy[scanpoint] / np.cos(
                    np.radians(twotheta - omega)
                )
            else:
                chod += cat.dy[scanpoint] / np.cos(
                    np.radians(omega - twotheta)
                )

        elif mode == "DB":
            # guide2_distance in in the middle of the 2nd compound mirror
            # guide2_distance - longitudinal length from midpoint1 -> midpoint2
            #  + direct length from midpoint1->midpoint2
            chod += cat.guide2_distance[0] + 600.0 * np.cos(
                np.radians(1.2)
            ) * (1 - np.cos(np.radians(2.4)))

            # add on distance from midpoint2 to sample
            chod += (cat.sample_distance[0] - cat.guide2_distance[0]) / np.cos(
                np.radians(4.8)
            )

            # add on sample -> detector
            if twotheta > omega:
                chod += cat.dy[scanpoint] / np.cos(np.radians(twotheta - 4.8))
            else:
                chod += cat.dy[scanpoint] / np.cos(np.radians(4.8 - twotheta))

        return chod, d_cx


class SpatzNexus(ReflectNexus):
    """
    Processes Spatz NeXus files to produce an intensity vs wavelength
    spectrum

    Parameters
    ----------
    h5data : HDF5 NeXus file or str
        An HDF5 NeXus file for Spatz, or a `str` containing the path
        to one
    """

    def __init__(self, h5data):
        """
        Initialises the SpatzNexus object.
        """
        super().__init__()
        self.prefix = "SPZ"
        with _possibly_open_hdf_file(h5data, "r") as f:
            self.cat = SpatzCatalogue(f)

    def detector_average_unwanted_direction(self, detector):
        """
        Averages over non-collimated beam direction
        """
        # Up until this point detector.shape=(N, T, Y, X)
        # pre-average over Y, leaving (n, t, x) also convert to dp
        return np.sum(detector, axis=2, dtype="float64")

    def create_detector_norm(self, h5norm):
        """
        Produces a detector normalisation array for a neutron detector.
        Here we average over N, T and Y to provide a relative efficiency for
        each X wire.

        Parameters
        ----------
        h5norm : hdf5 file
            Containing a flood field run (water)

        Returns
        -------
        norm, norm_sd : array_like
            1D array containing the normalisation data for each x pixel
        """
        y_bins = self.cat.y_bins
        return create_detector_norm(h5norm, y_bins[0], y_bins[1], axis=2)

    def estimated_beam_width_at_detector(self, scanpoint):
        cat = self.cat
        L23 = cat.cat["collimation_distance"]
        L3det = (
            cat.dy[scanpoint] + cat.sample_distance[0] - cat.slit3_distance[0]
        )
        ebw = general.height_of_beam_after_dx(
            cat.ss2hg[scanpoint], cat.ss3hg[scanpoint], L23, L3det
        )
        umb, penumb = ebw
        # convolve in detector resolution (~2.2 mm?)
        # first convert to beam sd, convolve in detector, and expand sd
        # back to total foreground width
        # use average of umb and penumb, the calc assumes a rectangular
        # distribution
        penumb = (
            np.sqrt((0.289 * 0.5 * (umb + penumb)) ** 2.0 + 2.2**2)
            * EXTENT_MULT
            * 2
        )
        # we need it in pixels
        return penumb / cat.qz_pixel_size[0]

    def time_offset(
        self,
        master_phase_offset,
        master_opening,
        freq,
        phase_angle,
        z0,
        flight_distance,
        tof_hist,
        t_offset=None,
    ):
        """
        Timing offsets for Spatz chopper system
        return total_offset
        """
        # calculate initial time offset from the phase angle and master
        # chopper offset.
        m_offset = 1.0e6 * master_phase_offset / (2.0 * 360.0 * freq)
        total_offset = m_offset + 1.0e6 * phase_angle / (360 * 2 * freq)

        # assumes that the pickup is in the middle of the chopper disc. But
        # you can override this by supplying a t_offset value.
        if t_offset is not None:
            total_offset += t_offset
        else:
            total_offset += 1.0e6 * master_opening / (2 * 360 * freq)

        return total_offset

    def phase_angle(self, scanpoint=0):
        """
        Calculates the phase angle for a given scanpoint

        Parameters
        ----------
        scanpoint : int
            The scanpoint you're interested in

        Returns
        -------
        phase_angle, master_opening : float
            The phase angle and angular opening of the master chopper in
            degrees.
        """
        disc_openings = (26.0, 42.0, 43.5, 126.0)
        O_C1d, O_C2d, O_C2Bd, O_C3d = disc_openings

        cat = self.cat
        master = cat.master
        slave = cat.slave
        disc_phase = cat.phase[scanpoint]
        phase_angle = 0

        if master == 1:
            phase_angle += 0.5 * O_C1d
            master_opening = O_C1d
        elif master == 2:
            phase_angle += 0.5 * O_C2d
            master_opening = O_C2d

        # the phase_offset is defined as the angle you have to add to the
        # calibrated blind opening to get to the nominal optically blind
        # chopper opening.
        # e.g. Nominal opening for optically blind may be at 34 degrees
        # but the calibrated optically blind position is 34.22 degrees
        # the chopper_phase_offset would be -0.22 degrees.
        if slave == 2:
            phase_angle += 0.5 * O_C2d
            phase_angle += -disc_phase - cat.poff_c2_slave_1_master[0]
        elif slave == 3:
            # chopper 2B
            phase_angle += 0.5 * O_C2Bd
            if master == 1:
                phase_angle += -disc_phase - cat.poff_c2b_slave_1_master[0]
            elif master == 2:
                phase_angle += -disc_phase - cat.poff_c2b_slave_2_master[0]

        return phase_angle, master_opening

    def chod(self, omega=0.0, twotheta=0.0, scanpoint=0):
        """
        Calculates the flight length of the neutrons in the Spatz
        instrument.

        Parameters
        ----------
        omega : float, optional
            Rough angle of incidence
        twotheta : float, optional
            Rough 2 theta angle
        scanpoint : int, optional
            Which dataset is being considered

        Returns
        -------
        chod, d_cx : float, float
            Flight distance (mm), distance between chopper discs (mm)
        """
        chod = 0

        cat = self.cat

        # Find out chopper pairing
        master = cat.master
        slave = cat.slave

        if master == 1:
            chod = cat.sample_distance

            if slave == 2:
                d_cx = cat.chopper2_distance[0]
            elif slave == 3:
                d_cx = cat.chopper2B_distance[0]
            else:
                raise RuntimeError("Couldn't figure out chopper spacing")

        elif master == 2:
            chod = cat.sample_distance - cat.chopper2_distance[0]
            if slave == 3:
                # chopper2B is the slave
                d_cx = cat.chopper2B_distance[0] - cat.chopper2_distance[0]
            else:
                raise RuntimeError("Couldn't figure out chopper spacing")

        chod += cat.dy[scanpoint] - 0.5 * d_cx

        return chod, d_cx


def background_subtract(detector, detector_sd, background_mask):
    """
    Background subtraction of Platypus detector image.
    Shape of detector is (N, T, {X, Y}), do a linear background subn for each
    (N, T) slice.

    Parameters
    ----------
    detector : np.ndarray
        detector array with shape (N, T, {X, Y}).
    detector_sd : np.ndarray
        standard deviations for detector array
    background_mask : array_like
        array of bool with shape (N, T, {X, Y}) that specifies which X or Y
        pixels to use for background subtraction.

    Returns
    -------
    detector, detector_sd : np.ndarray, np.ndarray
        Detector image with background subtracted
    """
    ret = np.zeros_like(detector)
    ret_sd = np.zeros_like(detector)

    for idx in np.ndindex(detector.shape[0:2]):
        ret[idx], ret_sd[idx] = background_subtract_line(
            detector[idx], detector_sd[idx], background_mask[idx]
        )
    return ret, ret_sd


def background_subtract_line(profile, profile_sd, background_mask):
    """
    Performs a linear background subtraction on a 1D peak profile

    Parameters
    ----------
    profile : np.ndarray
        1D profile
    profile_sd : np.ndarray
        standard deviations for profile
    background_mask : array_like
        array of bool that specifies which Y pixels to use for background
        subtraction.

    Returns
    -------
    profile_subt, profile_subt_err : np.ndarray, np.ndarray
        Background subtracted profile and its uncertainty
    """

    # which values to use as a background region
    mask = np.array(background_mask).astype("bool")
    x_vals = np.where(mask)[0]

    if np.size(x_vals) < 2:
        # can't do a background subtraction if you have less than 2 points in
        # the background
        return profile, profile_sd

    try:
        y_vals = profile[x_vals]
    except IndexError:
        print(x_vals)

    y_sdvals = profile_sd[x_vals]
    x_vals = x_vals.astype("float")

    # some SD values may have 0 SD, which will screw up curvefitting.
    y_sdvals = np.where(y_sdvals == 0, 1, y_sdvals)

    # equation for a straight line
    def f(x, a, b):
        return a + b * x

    # estimate the linear fit
    y_bar = np.mean(y_vals)
    x_bar = np.mean(x_vals)
    bhat = np.sum((x_vals - x_bar) * (y_vals - y_bar))
    bhat /= np.sum((x_vals - x_bar) ** 2)
    ahat = y_bar - bhat * x_bar

    # get the weighted fit values
    # we know the absolute sigma values
    popt, pcov = curve_fit(
        f,
        x_vals,
        y_vals,
        sigma=y_sdvals,
        p0=np.array([ahat, bhat]),
        absolute_sigma=True,
    )

    def CI(xx, pcovmat):
        return (
            pcovmat[0, 0]
            + pcovmat[1, 0] * xx
            + pcovmat[0, 1] * xx
            + pcovmat[1, 1] * (xx**2)
        )

    bkgd = f(np.arange(np.size(profile, 0)), popt[0], popt[1])

    # now work out confidence intervals
    # TODO, should this be confidence interval or prediction interval?
    # if you try to do a fit which has a singular matrix
    if np.isfinite(pcov).all():
        bkgd_sd = np.asarray(
            [CI(x, pcov) for x in np.arange(len(profile))], dtype="float64"
        )
    else:
        bkgd_sd = np.zeros_like(bkgd)

    bkgd_sd = np.sqrt(bkgd_sd)

    # get the t value for a two sided student t test at the 68.3 confidence
    # level
    bkgd_sd *= t.isf(0.1585, np.size(x_vals, 0) - 2)

    return EP.EPsub(profile, profile_sd, bkgd, bkgd_sd)


def find_specular_ridge(
    detector,
    detector_sd,
    search_increment=50,
    tol=(0.05, 0.015),
    manual_beam_find=None,
    name=None,
):
    """
    Find the specular ridges in a detector(n, t, {x, y}) plot.

    Parameters
    ----------
    detector : array_like
        detector array
    detector_sd : array_like
        standard deviations of detector array
    search_increment : int
        specifies the search increment for the location process.
    tol : (float, float) tuple
        specifies tolerances for finding the specular beam.
        tol[0] is the absolute change (in pixels) in beam centre location
        below which peak finding stops.
        tol[1] is the relative change in beam width below which peak finding
        stops.
    manual_beam_find : callable, optional
        A function which allows the location of the specular ridge to be
        determined. Has the signature `f(detector, detector_err, name)`
        where `detector` and `detector_err` is the detector image and its
        uncertainty, and name is a `str` specifying the name of
        the dataset.
        `detector` and `detector_err` have shape (n, t, {x, y}) where `n`
        is the number of detector images, `t` is the number of
        time-of-flight bins and `x` or `y` is the number of x or y pixels.
        The function should return a tuple,
        `(centre, centre_sd, lopx, hipx, background_pixels)`. `centre`,
        `centre_sd`, `lopx`, `hipx` should be arrays of shape `(n, )`,
        specifying the beam centre, beam width (standard deviation), lowest
        pixel of foreground region, highest pixel of foreground region.
        `background_pixels` is a list of length `n`. Each of the entries
        should contain arrays of pixel numbers that specify the background
        region for each of the detector images.
    name: str
            Name of the dataset
    Returns
    -------
    centre, SD, lopx, hipx, background_mask : np.ndarrays
        peak centre, standard deviation of peak width, lowest pixel to be
        included from background region, highest pixel to be included from
        background region, array specifying points to be used for background
        subtraction
        `np.size(centre) == n`.

    Notes
    -----
    The search for the beam centre proceeds by taking the last
    `search_increment` time bins, summing over the time axis and finding
    the beam centre and width along the y-axis. It then repeats the process
    with the last `2 * search_increment` time bins. This process is repeated
    until the relative change in beam centre and width is lower than `tol`.
    This process is designed to locate the specular ridge, even in the
    presence of incoherent scattering.
    """
    beam_centre = np.zeros(np.size(detector, 0))
    beam_sd = np.zeros_like(beam_centre)

    # unpack the tolerances
    atol, rtol = tol

    # lopx and hipx specify the foreground region to integrate over
    lopx = np.zeros_like(beam_centre, dtype=int)
    hipx = np.zeros_like(beam_centre, dtype=int)

    # background mask specifies which pixels are background
    background_mask = np.zeros_like(detector, dtype=bool)

    search_increment = abs(search_increment)

    n_increments = (
        np.size(detector, 1) - search_increment
    ) // search_increment

    # we want to integrate over the following pixel region
    for j in range(np.size(detector, 0)):
        last_centre = -1.0
        last_sd = -1.0
        converged = False
        for i in range(n_increments):
            how_many = -search_increment * (1 + i)
            det_subset = detector[j, how_many:]
            det_sd_subset = detector_sd[j, how_many:]

            # Uncertainties code takes a while to run
            # total_y = np.sum(det_subset, axis=0)
            y_cross = np.sum(det_subset, axis=0)
            y_cross_sd = np.sqrt(np.sum(det_sd_subset**2.0, axis=0))

            # find the centroid and gauss peak in the last sections of the TOF
            # plot
            try:
                centroid, gauss_peak = peak_finder(y_cross, sigma=y_cross_sd)
            except RuntimeError:
                continue

            if np.allclose(
                gauss_peak[0], last_centre, atol=atol
            ) and np.allclose(gauss_peak[1], last_sd, rtol=rtol, atol=0):
                last_centre = gauss_peak[0]
                last_sd = gauss_peak[1]
                converged = True
                break

            last_centre = gauss_peak[0]
            last_sd = gauss_peak[1]

        if not converged:
            warnings.warn(
                "specular ridge search did not work properly"
                " using last known centre",
                RuntimeWarning,
            )
            if manual_beam_find is not None:
                ret = manual_beam_find(detector[j], detector_sd[j], name)
                beam_centre[j], beam_sd[j], lopx[j], hipx[j], bp = ret
                background_mask[j, :, bp[0]] = True

                # don't assign to beam_centre, etc, at the end of this loop
                continue

        beam_centre[j] = last_centre
        beam_sd[j] = np.abs(last_sd)
        lp, hp, bp = fore_back_region(beam_centre[j], beam_sd[j])
        lopx[j] = lp
        hipx[j] = hp

        # bp are the background pixels. Clip to the range of the detector
        bp = np.clip(bp[0], 0, np.size(detector, 2) - 1)
        bp = np.unique(bp)
        background_mask[j, :, bp] = True

    # the foreground region needs to be totally contained within the
    # detector
    if (lopx < 0).any():
        raise ValueError(
            "The foreground region for one of the detector"
            " images extends below pixel 0."
        )
    if (hipx > np.size(detector, 2) - 1).any():
        raise ValueError(
            "The foreground region for one of the detector"
            " images extends above the largest detector"
            " pixel."
        )

    return beam_centre, beam_sd, lopx, hipx, background_mask


def fore_back_region(beam_centre, beam_sd):
    """
    Calculates the fore and background regions based on the beam centre and
    width

    Parameters
    ----------
    beam_centre : float
        beam_centre
    beam_sd : float
        beam width (standard deviation)

    Returns
    -------
    lopx, hipx, background_pixels: float, float, list
        Lowest pixel of foreground region
        Highest pixel of foreground region
        Pixels that are in the background region
        Each of these should have `len(lopx) == len(beam_centre)`
    """
    _b_centre = np.array(beam_centre)
    _b_sd = np.array(beam_sd)

    lopx = np.floor(_b_centre - _b_sd * EXTENT_MULT).astype("int")
    hipx = np.ceil(_b_centre + _b_sd * EXTENT_MULT).astype("int")

    background_pixels = []

    # limit of background regions
    # from refnx.reduce.platypusnexus
    y1 = np.atleast_1d(np.round(lopx - PIXEL_OFFSET).astype("int"))
    y0 = np.atleast_1d(
        np.round(lopx - PIXEL_OFFSET - (EXTENT_MULT * _b_sd)).astype("int")
    )

    y2 = np.atleast_1d(np.round(hipx + PIXEL_OFFSET).astype("int"))
    y3 = np.atleast_1d(
        np.round(hipx + PIXEL_OFFSET + (EXTENT_MULT * _b_sd)).astype("int")
    )

    # now generate background pixels
    for i in range(np.size(y0)):
        background_pixels.append(
            np.r_[np.arange(y0[i], y1[i] + 1), np.arange(y2[i], y3[i] + 1)]
        )

    return lopx, hipx, background_pixels


def correct_for_gravity(
    detector,
    detector_sd,
    lamda,
    coll_distance,
    sample_det,
    lo_wavelength,
    hi_wavelength,
    theta=0,
    qz_pixel_size=0.294,
):
    """
    Returns a gravity corrected yt plot, given the data, its associated errors,
    the wavelength corresponding to each of the time bins, and the trajectory
    of the neutrons. Low lambda and high Lambda are wavelength cutoffs to
    ignore.

    Parameters
    ----------
    detector : np.ndarray
        Detector image. Has shape (N, T, Y)
    detector_sd : np.ndarray
        Standard deviations of detector image
    lamda : np.ndarray
        Wavelengths corresponding to the detector image, has shape (N, T)
    coll_distance : float
        Collimation distance between slits, mm
    sample_det : float
        Sample - detector distance, mm
    lo_wavelength : float
        Low wavelength cut off, Angstrom
    hi_wavelength : float
        High wavelength cutoff, Angstrom
    theta : float
        Angle between second collimation slit, first collimation slit, and
        horizontal
    qz_pixel_size: float
        size of one pixel on the detector

    Returns
    -------
    corrected_data, corrected_data_sd, m_gravcorrcoefs :
        np.ndarray, np.ndarray, np.ndarray

        Corrected image. This is a theoretical prediction where the spectral
        ridge is for each wavelength.  This will be used to calculate the
        actual angle of incidence in the reduction process.

    """
    x_init = np.arange((np.size(detector, axis=2) + 1) * 1.0) - 0.5

    m_gravcorrcoefs = np.zeros((np.size(detector, 0)), dtype="float64")

    corrected_data = np.zeros_like(detector)
    corrected_data_sd = np.zeros_like(detector)

    for spec in range(np.size(detector, 0)):
        neutron_speeds = general.wavelength_velocity(lamda[spec])
        trajectories = find_trajectory(
            coll_distance / 1000.0, theta, neutron_speeds
        )
        travel_distance = (coll_distance + sample_det[spec]) / 1000.0

        # centres(t,)
        # TODO, don't use centroids, use Gaussian peak
        centroids = np.apply_along_axis(centroid, 1, detector[spec])
        lopx = np.searchsorted(lamda[spec], lo_wavelength)
        hipx = np.searchsorted(lamda[spec], hi_wavelength)

        def f(tru_centre):
            deflections = y_deflection(
                trajectories[lopx:hipx],
                neutron_speeds[lopx:hipx],
                travel_distance,
            )

            model = 1000.0 * deflections / qz_pixel_size + tru_centre
            diff = model - centroids[lopx:hipx, 0]
            diff = diff[~np.isnan(diff)]
            return diff

        # find the beam centre for an infinitely fast neutron
        x0 = np.array([np.nanmean(centroids[lopx:hipx, 0])])
        res = leastsq(f, x0)
        m_gravcorrcoefs[spec] = res[0][0]

        total_deflection = 1000.0 * y_deflection(
            trajectories, neutron_speeds, travel_distance
        )
        total_deflection /= qz_pixel_size

        x_rebin = x_init.T + total_deflection[:, np.newaxis]
        for wavelength in range(np.size(detector, axis=1)):
            output = rebin(
                x_init,
                detector[spec, wavelength],
                x_rebin[wavelength],
                y1_sd=detector_sd[spec, wavelength],
            )

            corrected_data[spec, wavelength] = output[0]
            corrected_data_sd[spec, wavelength] = output[1]

    return corrected_data, corrected_data_sd, m_gravcorrcoefs


def create_detector_norm(h5norm, bin_min, bin_max, axis):
    """
    Produces a detector normalisation array for a neutron detector
    (N, T, Y, X).
    The data in h5norm is averaged over N, T to start with, leaving
    a (Y, X) array. The data is then average over Y (axis=2) or X (axis=3)
    to provide a relative efficiency.

    Parameters
    ----------
    h5norm : hdf5 file
        Containing a flood field run (water)
    bin_min : float, int
        Minimum bin location to use
    bin_max : float, int
        Maximum bin location to use
    axis : int
        If axis = 2 the efficiency array is produced in the X direction.
        If axis = 3 the efficiency array is produced in the Y direction.

    Returns
    -------
    norm, norm_sd : array_like
        1D array containing the normalisation data for each y pixel
    """
    if axis not in (2, 3):
        raise RuntimeError("axis must be 2 or 3.")

    # sum over N and T
    detector = h5norm["entry1/data/hmm"]
    if axis == 3:
        norm_bins = h5norm["entry1/data/x_bin"]
    elif axis == 2:
        norm_bins = h5norm["entry1/data/y_bin"]

    # find out what pixels to use
    x_low = np.searchsorted(norm_bins, bin_min, sorter=np.argsort(norm_bins))
    x_low = np.argsort(norm_bins)[x_low]
    x_high = np.searchsorted(norm_bins, bin_max, sorter=np.argsort(norm_bins))
    x_high = np.argsort(norm_bins)[x_high]

    if x_low > x_high:
        x_low, x_high = x_high, x_low

    norm = np.sum(detector, axis=(0, 1), dtype="float64")

    # By this point you have norm[y][x]
    if axis == 3:
        norm = norm[:, x_low:x_high]
        norm = np.sum(norm, axis=1)
    elif axis == 2:
        norm = norm[x_low:x_high, :]
        norm = np.sum(norm, axis=0)

    mean = np.mean(norm)
    return norm / mean, np.sqrt(norm) / mean


def calculate_wavelength_bins(lo_wavelength, hi_wavelength, rebin_percent):
    """
    Calculates optimal logarithmically spaced wavelength histogram bins. The
    bins are equal size in log10 space, but they may not be exactly be
    `rebin_percent` in size. The limits would have to change slightly for that.

    Parameters
    ----------
    lo_wavelength : float
        Low wavelength cutoff
    hi_wavelength : float
        High wavelength cutoff
    rebin_percent : float
        Rebinning percentage

    Returns
    -------
    wavelength_bins : np.ndarray
    """
    frac = (rebin_percent / 100.0) + 1
    lowspac = rebin_percent / 100.0 * lo_wavelength
    hispac = rebin_percent / 100.0 * hi_wavelength

    lowl = lo_wavelength - lowspac / 2.0
    hil = hi_wavelength + hispac / 2.0
    num_steps = int(np.floor(np.log10(hil / lowl) / np.log10(frac)) + 1)
    rebinning = np.logspace(np.log10(lowl), np.log10(hil), num=num_steps)
    return rebinning


def accumulate_HDF_files(files):
    r"""
    Accumulates HDF files together, writing an accumulated file in the current
    directory. The accumulated datafile is written in the current directory
    (os.getcwd()) and has a filename based on the first file, prepended by
    'ADD\_'. For example, if the first file is PLP0000708.nx.hdf then the
    accumulated file is ADD_PLP0000708.nx.hdf.

    Parameters
    ----------
    files : list
        Strings specifying NeXUS filenames to be added together.

    """
    # don't do anything if no files were supplied.
    if not len(files):
        return None

    # the first file is the "master file", lets copy it.
    file = files[0]

    pth = _check_HDF_file(file)
    if not pth:
        raise ValueError("All files must refer to an hdf5 file")

    new_name = "ADD_" + os.path.basename(pth)

    shutil.copy(pth, os.path.join(os.getcwd(), new_name))

    master_file = os.path.join(os.getcwd(), new_name)
    with h5py.File(master_file, "r+") as h5master:
        # now go through each file and accumulate numbers:
        for file in files[1:]:
            pth = _check_HDF_file(file)
            h5data = h5py.File(pth, "r")

            h5master["entry1/data/hmm"][0] += h5data["entry1/data/hmm"][0]
            h5master["entry1/monitor/bm1_counts"][0] += h5data[
                "entry1/monitor/bm1_counts"
            ][0]
            h5master["entry1/instrument/detector/total_counts"][0] += h5data[
                "entry1/instrument/detector/total_counts"
            ][0]
            h5master["entry1/instrument/detector/time"][0] += h5data[
                "entry1/instrument/detector/time"
            ][0]
            h5master.flush()

            h5data.close()


def _check_HDF_file(h5data):
    # If a file is an HDF5 file, then return the filename.
    # otherwise return False
    if type(h5data) == h5py.File:
        return h5data.filename
    else:
        with h5py.File(h5data, "r") as h5data:
            if type(h5data) == h5py.File:
                return h5data.filename

    return False


@contextmanager
def _possibly_open_hdf_file(f, mode="r"):
    """
    Context manager for hdf5 files.

    Parameters
    ----------
    f : file-like or str
        If `f` is a file, then yield the file. If `f` is a str then open the
        file and yield the newly opened file.
        On leaving this context manager the file is closed, if it was opened
        by this context manager (i.e. `f` was a string).
    mode : str, optional
        mode is an optional string that specifies the mode in which the file
        is opened.

    Yields
    ------
    g : file-like
        On leaving the context manager the file is closed, if it was opened by
        this context manager.
    """
    close_file = False
    if type(f) == h5py.File:
        g = f
    else:
        g = h5py.File(f, mode)
        close_file = True
    yield g
    if close_file:
        g.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some Platypus NeXUS"
        "files to produce their TOF "
        "spectra."
    )

    parser.add_argument(
        "file_list",
        metavar="N",
        type=int,
        nargs="+",
        help="integer file numbers",
    )

    parser.add_argument(
        "-b",
        "--bdir",
        type=str,
        help="define the location to find the nexus files",
    )

    parser.add_argument(
        "-d",
        "--direct",
        action="store_true",
        default=False,
        help="is the file a direct beam?",
    )

    parser.add_argument(
        "-r",
        "--rebin",
        type=float,
        help="rebin percentage for the wavelength -1<rebin<10",
        default=1,
    )

    parser.add_argument(
        "-ll",
        "--lolambda",
        type=float,
        help="lo wavelength cutoff for the rebinning",
        default=2.5,
    )

    parser.add_argument(
        "-hl",
        "--hilambda",
        type=float,
        help="lo wavelength cutoff for the rebinning",
        default=19.0,
    )

    parser.add_argument(
        "-i",
        "--integrate",
        type=int,
        help="-1 to integrate all spectra, otherwise enter the"
        " spectrum number.",
        default=-1,
    )
    args = parser.parse_args()

    for file in args.file_list:
        fname = "PLP%07d.nx.hdf" % file
        path = os.path.join(args.bdir, fname)
        try:
            a = PlatypusNexus(path)
            a.process(
                lo_wavelength=args.lolambda,
                hi_wavelength=args.hilambda,
                direct=args.direct,
                rebin_percent=args.rebin,
                integrate=args.integrate,
            )

            fname = "PLP%07d.spectrum" % file
            out_fname = os.path.join(args.bdir, fname)

            integrate = args.integrate
            if args.integrate < 0:
                integrate = 0

            a.write_spectrum_dat(out_fname, scanpoint=integrate)

        except IOError:
            print("Couldn't find file: %d.  Use --basedir option" % file)
