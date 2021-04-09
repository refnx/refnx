import string
from copy import deepcopy
import os.path
from time import gmtime, strftime
from multiprocessing import Queue
from threading import Thread
import time

import numpy as np
import pandas as pd
import h5py

from refnx.reduce.platypusnexus import (
    PlatypusNexus,
    ReflectNexus,
    number_datafile,
    basename_datafile,
    SpatzNexus,
    ReductionOptions,
)
from refnx.util import ErrorProp as EP
import refnx.util.general as general
from refnx.reduce.parabolic_motion import (
    parabola_line_intersection_point,
    find_trajectory,
)
from refnx.dataset import ReflectDataset
from refnx._lib import possibly_open_file


_template_ref_xml = """<?xml version="1.0"?>
<REFroot xmlns="">
<REFentry time="$time">
<Title>$title</Title>
<User>$user</User>
<REFsample>
<ID>$sample</ID>
</REFsample>
<REFdata axes="Qz:Qx" rank="2" type="POINT" \
spin="UNPOLARISED" dim="$_numpointsz:$_numpointsy">
<Run filename="$_rnumber" preset="" size="">
</Run>
<R uncertainty="dR">$_r</R>
<Qz uncertainty="dQz" units="1/A">$_qz</Qz>
<dR type="SD">$_dr</dR>
<Qx type="_FWHM" units="1/A">$_qx</Qx>
</REFdata>
</REFentry>
</REFroot>"""


class ReflectReduce:
    def __init__(self, direct, prefix, data_folder=None):

        self.data_folder = os.path.curdir
        if data_folder is not None:
            self.data_folder = data_folder

        if prefix == "PLP":
            self.reflect_klass = PlatypusNexus
        elif prefix == "SPZ":
            self.reflect_klass = SpatzNexus
        else:
            raise ValueError(
                "Instrument prefix not known. Must be one of" " ['PLP']"
            )

        if isinstance(direct, ReflectNexus):
            self.direct_beam = direct
        elif type(direct) is str:
            direct = os.path.join(self.data_folder, direct)
            self.direct_beam = self.reflect_klass(direct)
        else:
            self.direct_beam = self.reflect_klass(direct)

        self.prefix = prefix

    def __call__(self, reflect, scale=1.0, save=True, **reduction_options):
        return self.reduce(
            reflect, scale=scale, save=save, **reduction_options
        )

    def reduce(self, reflect, scale=1.0, save=True, **reduction_options):
        """
        Reduction of a single dataset.

        The reduction uses the direct beam specified during construction of
        this object. This method reduces all the spectra present in the
        reflected beam file (see platypusnexus.PlatypusNexus.process for
        eventmode specification and other related options), but aggregates
        all data in the direct beam spectrum.

        Parameters
        ----------
        reflect : string, hdf5 file-handle or PlatypusNexus object
            A string containing the path to the specularly reflected hdf5 file,
            the hdf5 file itself, or a PlatypusNexus object.
        scale : float, optional
            Divide all the reflectivity values by this number.
        save : bool, optional
            If `True` then the reduced dataset is saved to the current
            directory, with a name os.path.basename(reflect)
        reduction_options : dict, optional
            Options passed directly to `refnx.reduce.PlatypusNexus.process`,
            for processing of individual spectra. Look at that method docstring
            for specification of options.

        Returns
        -------
        datasets, reduction : tuple

        datasets : sequence of ReflectDataset

        reduction : dict
            Contains the following entries:

            - 'x' : np.ndarray
                Q values, shape (N, T).
            - 'x_err' : np.ndarray
                Uncertainty in Q values (FWHM), shape (N, T).
            - 'y' : np.ndarray
                Specular Reflectivity, shape (N, T)
            - 'y_err' : np.ndarray
                Uncertainty in specular reflectivity (SD), shape (N, T)
            - 'omega' : np.ndarray
                Angle of incidence, shape (N, T)
            - 'm_ref' : np.ndarray
                Offspecular reflectivity map, shape (N, T, Y)
            - 'm_ref_err' : np.ndarray
                uncertainty in offspecular reflectivity, shape (N, T, Y)
            - 'm_qz' : np.ndarray
                Qz for offspecular map, shape (N, T, Y)
            - 'm_qx' : np.ndarray
                Qx for offspecular map, shape (N, T, Y)
            - 'n_spectra' : int
                number of reflectivity spectra
            - 'datafile_number' : int
                run number for the reflected beam
            - 'fname' : list
                the saved filenames

        N corresponds to the number of spectra
        T corresponds to the number of Q (wavelength) bins
        Y corresponds to the number of y pixels on the detector.

        Notes
        -----
        All the values returned from this method are also contained as instance
        attributes for this object.

        Examples
        --------

        >>> from refnx.reduce import PlatypusReduce
        >>> # set up with a direct beam
        >>> reducer = PlatypusReduce('PLP0000711.nx.hdf')
        >>> datasets, reduction = reducer.reduce('PLP0000708.nx.hdf',
        ...                                      rebin_percent=3.)

        """
        reflect_keywords = reduction_options.copy()
        direct_keywords = reduction_options.copy()

        # get the direct beam spectrum
        direct_keywords["direct"] = True
        direct_keywords["integrate"] = -1
        if (
            "peak_pos" in direct_keywords
            and hasattr(direct_keywords["peak_pos"], "len")
            and len(direct_keywords["peak_pos"]) == 2
        ):
            # don't use a user specified peak_pos for direct and reflected
            # beams, only for the reflected beam. Leave the computer to find
            # the direct beam pos. Alternatively one can use the manual beam
            # finder
            direct_keywords.pop("peak_pos")

        if "eventmode" in direct_keywords:
            direct_keywords.pop("eventmode")

        if "event_filter" in direct_keywords:
            direct_keywords.pop("event_filter")

        self.direct_beam.process(**direct_keywords)

        # get the reflected beam spectrum
        reflect_keywords["direct"] = False
        if isinstance(reflect, ReflectNexus):
            self.reflected_beam = reflect
        elif type(reflect) is str:
            reflect = os.path.join(self.data_folder, reflect)
            self.reflected_beam = self.reflect_klass(reflect)
        else:
            self.reflected_beam = self.reflect_klass(reflect)

        # Got to use the same wavelength bins as the direct spectrum.
        # done this way around to save processing direct beam over and over
        reflect_keywords["wavelength_bins"] = self.direct_beam.m_lambda_hist[0]

        self.reflected_beam.process(**reflect_keywords)

        self.save = save
        self._reduce_single_angle(scale)

        """
        --Specular Reflectivity--
        Use the (constant wavelength) spectra that have already been integrated
        over 2theta (in processnexus) to calculate the specular reflectivity.
        Beware: this is because m_topandtail has already been divided through
        by monitor counts and error propagated (at the end of processnexus).
        Thus, the 2theta pixels are correlated to some degree. If we use the 2D
        plot to calculate reflectivity
        (sum {Iref_{2theta, lambda}}/I_direct_{lambda}) then the error bars in
        the reflectivity turn out much larger than they should be.
        """
        ydata, ydata_sd = EP.EPdiv(
            self.reflected_beam.m_spec,
            self.reflected_beam.m_spec_sd,
            self.direct_beam.m_spec,
            self.direct_beam.m_spec_sd,
        )

        # calculate the 1D Qz values.
        xdata = general.q(self.omega_corrected, self.reflected_beam.m_lambda)
        xdata_sd = (
            self.reflected_beam.m_lambda_fwhm / self.reflected_beam.m_lambda
        ) ** 2
        xdata_sd += (
            self.reflected_beam.domega[:, np.newaxis] / self.omega_corrected
        ) ** 2
        xdata_sd = np.sqrt(xdata_sd) * xdata

        """
        ---Offspecular reflectivity---
        normalise the counts in the reflected beam by the direct beam
        spectrum this gives a reflectivity. Also propagate the errors,
        leaving the fractional variance (dr/r)^2.
        --Note-- that adjacent y-pixels (same wavelength) are correlated in
        this treatment, so you can't just sum over them.
        i.e. (c_0 / d) + ... + c_n / d) != (c_0 + ... + c_n) / d
        """
        m_ref, m_ref_sd = EP.EPdiv(
            self.reflected_beam.m_topandtail,
            self.reflected_beam.m_topandtail_sd,
            self.direct_beam.m_spec[:, :, np.newaxis],
            self.direct_beam.m_spec_sd[:, :, np.newaxis],
        )

        # you may have had divide by zero's.
        m_ref = np.where(np.isinf(m_ref), 0, m_ref)
        m_ref_sd = np.where(np.isinf(m_ref_sd), 0, m_ref_sd)

        # calculate the Q values for the detector pixels.  Each pixel has
        # different 2theta and different wavelength, ASSUME that they have the
        # same angle of incidence
        qx, qy, qz = general.q2(
            self.omega_corrected[:, :, np.newaxis],
            self.m_twotheta,
            0,
            self.reflected_beam.m_lambda[:, :, np.newaxis],
        )

        reduction = {}
        reduction["x"] = self.x = xdata
        reduction["x_err"] = self.x_err = xdata_sd
        reduction["y"] = self.y = ydata / scale
        reduction["y_err"] = self.y_err = ydata_sd / scale
        reduction["m_ref"] = self.m_ref = m_ref
        reduction["m_ref_err"] = self.m_ref_err = m_ref_sd
        reduction["qz"] = self.m_qz = qz
        reduction["qx"] = self.m_qx = qx
        reduction["nspectra"] = self.n_spectra
        reduction["start_time"] = self.reflected_beam.start_time
        reduction[
            "datafile_number"
        ] = self.datafile_number = self.reflected_beam.datafile_number

        fnames = []
        datasets = []
        datafilename = self.reflected_beam.datafilename
        datafilename = os.path.basename(datafilename.split(".nx.hdf")[0])

        header = self._create_metadata_header()

        for i in range(self.n_spectra):
            data_tup = self.data(scanpoint=i)
            datasets.append(ReflectDataset(data_tup))

        if self.save:
            for i, dataset in enumerate(datasets):
                fname = f"{datafilename}_{i}.dat"
                fnames.append(fname)
                with open(fname, "wb") as f:
                    dataset.save(f, header=header)

                fname = f"{datafilename}_{i}.xml"
                with open(fname, "wb") as f:
                    dataset.save_xml(f, start_time=reduction["start_time"][i])

        reduction["fname"] = fnames
        return datasets, deepcopy(reduction)

    def data(self, scanpoint=0):
        """
        The specular reflectivity

        Parameters
        ----------
        scanpoint: int
            Find a particular specular reflectivity image. scanpoints upto
            `self.n_spectra - 1` can be specified.

        Returns
        -------
        (Q, R, dR, dQ): np.ndarray tuple
            dR is standard deviation, dQ is FWHM
        """
        return (
            self.x[scanpoint],
            self.y[scanpoint],
            self.y_err[scanpoint],
            self.x_err[scanpoint],
        )

    def data2d(self, scanpoint=0):
        """
        The offspecular data

        Parameters
        ----------
        scanpoint: int
            Find a particular offspecular image. scanpoints upto
            self.n_spectra - 1 can be specified.

        Returns
        -------
        (Qz, Qx, R, dR): np.ndarrays
        """

        return (
            self.m_qz[scanpoint],
            self.m_qx[scanpoint],
            self.m_ref[scanpoint],
            self.m_ref_err[scanpoint],
        )

    def scale(self, scale):
        """
        Divides the reflectivity values by this scale factor

        Parameters
        ----------
        scale: float
            Divides the reflectivity values by a constant amount
        """
        self.m_ref /= scale
        self.m_ref_err /= scale
        self.y /= scale
        self.y_err /= scale

    def write_offspecular(self, f, scanpoint=0):
        d = dict()
        d["time"] = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        d["_rnumber"] = self.reflected_beam.datafile_number
        d["_numpointsz"] = np.size(self.m_ref, 1)
        d["_numpointsy"] = np.size(self.m_ref, 2)

        s = string.Template(_template_ref_xml)

        # filename = 'off_PLP{:07d}_{:d}.xml'.format(self._rnumber, index)
        d["_r"] = repr(self.m_ref[scanpoint].tolist()).strip(",[]")
        d["_qz"] = repr(self.m_qz[scanpoint].tolist()).strip(",[]")
        d["_dr"] = repr(self.m_ref_err[scanpoint].tolist()).strip(",[]")
        d["_qx"] = repr(self.m_qx[scanpoint].tolist()).strip(",[]")

        thefile = s.safe_substitute(d)

        with possibly_open_file(f, "wb") as g:
            if "b" in g.mode:
                thefile = thefile.encode("utf-8")

            g.write(thefile)
            g.truncate()

    def _create_metadata_header(self):
        header = []
        header.append(
            f"reflected_beam_number: {self.reflected_beam.datafilename}"
        )
        header.append(f"direct_run_number: {self.direct_beam.datafilename}")
        header.append(f"samplename: {self.reflected_beam.cat.sample_name}")

        ps_dct = self.reflected_beam.processed_spectrum
        header.append(f"beam_centre: {ps_dct['m_beampos']}")
        header.append(f"beam_sd: {ps_dct['m_beampos_sd']}")
        header.append(f"lopx: {ps_dct['hipx']}")
        header.append(f"hipx: {ps_dct['lopx']}")
        rdo = ps_dct["reduction_options"]
        header.append(f"rebin_percent: {rdo['rebin_percent']}")
        header.append(f"background: {rdo['background']}")
        header.append(f"lo_wavelength: {rdo['lo_wavelength']}")
        header.append(f"hi_wavelength: {rdo['hi_wavelength']}")

        header.append(
            "Warning: the format of this header may change at any"
            " time. Do not rely on it staying constant"
        )
        header.append("Q (1/A), R, dR (sigma), dQ (1/A, FWHM)")
        return "\n".join(header)


class PlatypusReduce(ReflectReduce):
    """
    Reduces Platypus reflectometer data to give the specular reflectivity.
    Offspecular data maps are also produced.

    Parameters
    ----------
    direct : string, hdf5 file-handle or PlatypusNexus object
        A string containing the path to the direct beam hdf5 file,
        the hdf5 file itself, or a PlatypusNexus object.
    data_folder : str, optional
        Where is the raw data stored?

    Examples
    --------

    >>> from refnx.reduce import PlatypusReduce
    >>> reducer = PlatypusReduce('PLP0000711.nx.hdf')
    >>> datasets, reduced = reducer.reduce('PLP0000711.nx.hdf',
    ...                                    rebin_percent=2)

    """

    def __init__(self, direct, data_folder=None, **kwds):

        super().__init__(direct, "PLP", data_folder=data_folder)

    def _reduce_single_angle(self, scale=1):
        """
        Reduce a single angle.
        """
        n_spectra = self.reflected_beam.n_spectra
        n_tpixels = np.size(self.reflected_beam.m_topandtail, 1)
        n_ypixels = np.size(self.reflected_beam.m_topandtail, 2)

        # calculate omega and two_theta depending on the mode.
        mode = self.reflected_beam.mode

        # we'll need the wavelengths to calculate gravity effects.
        wavelengths = self.reflected_beam.m_lambda
        m_twotheta = np.zeros((n_spectra, n_tpixels, n_ypixels))

        detector_z_difference = (
            self.reflected_beam.detector_z - self.direct_beam.detector_z
        )

        beampos_z_difference = (
            self.reflected_beam.m_beampos - self.direct_beam.m_beampos
        )

        Y_PIXEL_SPACING = self.reflected_beam.cat.qz_pixel_size[0]

        total_z_deflection = (
            detector_z_difference + beampos_z_difference * Y_PIXEL_SPACING
        )

        if mode in ["FOC", "POL", "POLANAL", "MT"]:
            # omega_nom.shape = (N, )
            omega_nom = np.degrees(
                np.arctan(total_z_deflection / self.reflected_beam.detector_y)
                / 2.0
            )

            """
            Wavelength specific angle of incidence correction
            This involves:
            1) working out the trajectory of the neutrons through the
            collimation system.
            2) where those neutrons intersect the sample.
            3) working out the elevation of the neutrons when they hit the
            sample.
            4) correcting the angle of incidence.
            """
            speeds = general.wavelength_velocity(wavelengths)
            collimation_distance = self.reflected_beam.cat.collimation_distance
            s2_sample_distance = (
                self.reflected_beam.cat.sample_distance
                - self.reflected_beam.cat.slit2_distance
            )

            # work out the trajectories of the neutrons for them to pass
            # through the collimation system.
            trajectories = find_trajectory(
                collimation_distance / 1000.0, 0, speeds
            )

            # work out where the beam hits the sample
            res = parabola_line_intersection_point(
                s2_sample_distance / 1000,
                0,
                trajectories,
                speeds,
                omega_nom[:, np.newaxis],
            )
            intersect_x, intersect_y, x_prime, elevation = res

            # correct the angle of incidence with a wavelength dependent
            # elevation.
            omega_corrected = omega_nom[:, np.newaxis] - elevation

            m_twotheta += np.arange(n_ypixels * 1.0)[np.newaxis, np.newaxis, :]
            m_twotheta -= self.direct_beam.m_beampos[:, np.newaxis, np.newaxis]
            m_twotheta *= Y_PIXEL_SPACING
            m_twotheta += detector_z_difference
            m_twotheta /= self.reflected_beam.detector_y[
                :, np.newaxis, np.newaxis
            ]
            m_twotheta = np.arctan(m_twotheta)
            m_twotheta = np.degrees(m_twotheta)

            # you may be reflecting upside down, reverse the sign.
            upside_down = np.sign(omega_corrected[:, 0])
            m_twotheta *= upside_down[:, np.newaxis, np.newaxis]
            omega_corrected *= upside_down[:, np.newaxis]

        elif mode in ["SB", "DB"]:
            # the angle of incidence is half the two theta of the reflected
            # beam
            omega = (
                np.arctan(total_z_deflection / self.reflected_beam.detector_y)
                / 2.0
            )

            # work out two theta for each of the detector pixels
            m_twotheta += np.arange(n_ypixels * 1.0)[np.newaxis, np.newaxis, :]
            m_twotheta -= self.direct_beam.m_beampos[:, np.newaxis, np.newaxis]
            m_twotheta += detector_z_difference
            m_twotheta -= self.reflected_beam.detector_y[
                :, np.newaxis, np.newaxis
            ] * np.tan(omega[:, np.newaxis, np.newaxis])

            m_twotheta /= self.reflected_beam.detector_y[
                :, np.newaxis, np.newaxis
            ]
            m_twotheta = np.arctan(m_twotheta)
            m_twotheta += omega[:, np.newaxis, np.newaxis]

            # still in radians at this point
            # add an extra dimension, because omega_corrected needs to be the
            # angle of incidence for each wavelength. I.e. should be
            # broadcastable to (N, T)
            omega_corrected = np.degrees(omega)[:, np.newaxis]
            m_twotheta = np.degrees(m_twotheta)

        self.omega_corrected = omega_corrected
        self.m_twotheta = m_twotheta
        self.n_spectra = n_spectra


class PolarisedReduce:
    """
    Reduces a direct beam and reflected beam spinset to produce
    a polarised neutron reflectivity curve that is corrected
    for polarisation efficiency.


    Parameters
    ----------
    spin_set_direct :   refnx.reduce.SpinSet
                    SpinSet object of direct beams in experiment
            
    """
    def __init__(self, spin_set_direct):
        self.spin_set_direct = spin_set_direct
        self.reducers = {}
        for sc in ["dd", "du"]:
            self.reducers[sc] = PlatypusReduce(spin_set_direct.dd)
        for sc in ["uu", "ud"]:
            self.reducers[sc] = PlatypusReduce(spin_set_direct.uu)

    def reduce(self, spin_set_reflect, **reduction_options):
        # get a default set of reduction options
        options = ReductionOptions()
        options.update(reduction_options)

        # set up the wavelength bins
        if options["wavelength_bins"] is None:
            wb = calculate_wavelength_bins(
                options["lo_wavelength"],
                options["hi_wavelength"],
                options["rebin_percent"],
            )
            options["wavelength_bins"] = wb

        reduced_successfully = []

        # go through each spin channel and reduce it
        for sc in ["dd", "uu", "du", "ud"]:
            # first get the correct reduction options
            rdo = getattr(spin_set_reflect, f"{sc}_opts")
            if rdo is None:
                rdo = options
            else:
                # overwrite properties that need to be common
                rdo["wavelength_bins"] = options["wavelength_bins"]

            db = self.reducers[sc]
            rb = getattr(self.spin_set_reflect, sc)
            if rb is not None:
                db.reduce(rb, save=False, **rdo)
                reduced_successfully.append(sc)
            else:
                # no reflected beam for a spin channel
                continue

        # by this point an unpolarised reduction has been done. But we need to
        # correct the spectra for PNR. The following spectra (N, T) should be
        # overwritten:
        # TODO
        # self.reducers["dd"].reflected_beam.m_spec
        # self.reducers["dd"].reflected_beam.m_spec_sd
        # self.reducers["dd"].direct_beam.m_spec
        # self.reducers["dd"].direct_beam.m_spec_sd
        # THIS IS WHERE THE MAGIC HAPPENS

        # once the wavelength spectra have been corrected/overwritten then the
        # reflectivities need to be recalculated.
        # this doesn't correct the offspecular
        for sc in reduced_successfully:
            reducer = self.reducers[sc]
            # TODO call something polcorr?
            reducer.y, reducer.y_err = EP.EPdiv(
                reducer.reflected_beam.m_spec,
                reducer.reflected_beam.m_spec_sd,
                reducer.direct_beam.m_spec,
                reducer.direct_beam.m_spec_sd,
            )
            # now write out the corrected reflectivity files
            fnames = []
            datasets = []
            datafilename = reducer.reflected_beam.datafilename
            datafilename = os.path.basename(datafilename.split(".nx.hdf")[0])

            for i in range(np.size(reducer.y, 0)):
                data_tup = reducer.data(scanpoint=i)
                datasets.append(ReflectDataset(data_tup))

                for i, dataset in enumerate(datasets):
                    fname = f"{datafilename}_{i}_POLCORR.dat"
                    fnames.append(fname)
                    with open(fname, "wb") as f:
                        dataset.save(f)


def _check_spin_channel(data, channel):
    """
    Checks that the data measured corresponds
    to the correct spin channel

    Parameters
    ----------
    data : PlatypusNexus object
        Polarised spectra for a given spin channel
    channel : SpinChannel Enum object
        nominal spin channel - (0,0), (0,1), (1,0), or (1,1)

    Output: Boolean value.
        True if spin channel matches flipper settings
        in data, and false if not.
    """

    if data.spin_state == channel:
        return True
    else:
        return False


def polarised_correction(
    I00=None, I01=None, I10=None, I11=None, reduction_options=None
):
    """
    Applies polarisation efficiency correction to a
    set of polarised PlatypusNexus objects.

    Parameters
    ----------
    I00 : PlatypusNexus object
        Polarised spectra for ++ spin channel
    I01 : PlatypusNexus object or None
        Polarised spectra for +- spin channel
    I10 : PlatypusNexus object or None
        Polarised spectra for -+ spin channel
    I11 : PlatypusNexus object
        Polarised spectra for -- spin channel

    Returns
    -------
    mm, mp, pm, pp  :   PlatypusNexus or None,
                        PlatypusNexus or None,
                        PlatypusNexus or None,
                        PlatypusNexus or None
    The corrected datasets within the PlatypusNexus
    object, if the spin channel was measured
    """

    # Check that the input spin channels match the
    # flipper settings for those spin cross sections
    if (I11 is not None and _check_spin_channel(I11, SpinChannel.UP_UP) is False):
        print("Error: Input for ++ channel doesn't match flipper statuses!")
        return -1
    elif (I01 is not None and _check_spin_channel(I01, SpinChannel.DOWN_UP) is False):
        print("Error: Input for -+ channel doesn't match flipper statuses!")
        return -1
    elif (I10 is not None and _check_spin_channel(I10, SpinChannel.UP_DOWN) is False):
        print("Error: Input for +- channel doesn't match flipper statuses!")
        return -1
    elif (I00 is not None and _check_spin_channel(I00, SpinChannel.DOWN_DOWN) is False):
        print("Error: Input for -- channel doesn't match flipper statuses!")
        return -1

    # Check if R++ or R-- channel is missing
    if I00 is None:
        print("Error: Missing R-- channel")
        return -1
    elif I11 is None:
        print("Error: Missing R++ channel")
        return -1

    # Initialise reduction options if not passed to function
    if reduction_options is None:
        reduction_options = {
            "background": True,
            "lo_wavelength": 2.5,
            "hi_wavelength": 12.5,
            "polarised": True
        }

    for channel in _not_none(I00, I01, I10, I11):
        if not channel.processed_spectrum:
            channel.process(**reduction_options)

    # Check what combination of spin channels were measured
    if (
        (I00 is not None)
        and (I01 is not None)
        and (I10 is not None)
        and (I11 is not None)
    ):

        mm_spec, mp_spec, pm_spec, pp_spec = correct_POL_efficiencies(
            I11.processed_spectrum['m_lambda'][0],
            I00=I00.processed_spectrum['m_spec'][0],
            I01=I01.processed_spectrum['m_spec'][0],
            I10=I10.processed_spectrum['m_spec'][0],
            I11=I11.processed_spectrum['m_spec'][0]
        )
        I00.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in mm_spec]])
        I01.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in mp_spec]])
        I10.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in pm_spec]])
        I11.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in pp_spec]])
    elif I01 is None and I10 is not None:
        print("One spin-flip channel detected. Assuming R+- = R-+.")
        I01 = I10

        mm_spec, mp_spec, pm_spec, pp_spec = correct_POL_efficiencies(
            I11.processed_spectrum['m_lambda'][0],
            I00=I00.processed_spectrum['m_spec'][0],
            I01=I01.processed_spectrum['m_spec'][0],
            I10=I10.processed_spectrum['m_spec'][0],
            I11=I11.processed_spectrum['m_spec'][0]
        )
        I00.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in mm_spec]])
        I01.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in mp_spec]])
        I10.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in pm_spec]])
        I11.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in pp_spec]])
    elif I10 is None and I01 is not None:
        print("One spin-flip channel detected. Assuming R+- = R-+.")
        I10 = I01

        mm_spec, mp_spec, pm_spec, pp_spec = correct_POL_efficiencies(
            I11.processed_spectrum['m_lambda'][0],
            I00=I00.processed_spectrum['m_spec'][0],
            I01=I01.processed_spectrum['m_spec'][0],
            I10=I10.processed_spectrum['m_spec'][0],
            I11=I11.processed_spectrum['m_spec'][0]
        )
        I00.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in mm_spec]])
        I01.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in mp_spec]])
        I10.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in pm_spec]])
        I11.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in pp_spec]])
    elif I10 is None and I01 is None:
        print("No spin-flip channel detected. Assuming R+- = R-+ = 0")
        I01 = np.zeros_like(I11.processed_spectrum['m_spec'][0])
        I10 = np.zeros_like(I11.processed_spectrum['m_spec'][0])

        mm_spec, mp_spec, pm_spec, pp_spec = correct_POL_efficiencies(
            I11.processed_spectrum['m_lambda'][0],
            I00=I00.processed_spectrum['m_spec'][0],
            I01=I01,
            I10=I10,
            I11=I11.processed_spectrum['m_spec'][0]
        )
        I00.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in mm_spec]])
        I01 = None
        I10 = None
        I11.processed_spectrum['m_spec_polcorr'] = np.asarray(
            [[val[0] for val in pp_spec]])
    return I00, I01, I10, I11


def correct_POL_efficiencies(
    wavelength, I00=None, I01=None, I10=None, I11=None
):
    """
    Applies matrix correction of polarisation
    inefficiencies to the measured spectra.

    Parameters
    ----------
    I00 : PlatypusNexus object
        Polarised spectra for ++ spin channel
    I01 : PlatypusNexus object
        Polarised spectra for +- spin channel
    I10 : PlatypusNexus object
        Polarised spectra for -+ spin channel
    I11 : PlatypusNexus object
        Polarised spectra for -- spin channel

    Returns
    -------
    mm, mp, pm, pp : PlatypusNexus object
    with the efficiency-corrected spectrum
    """
    # The values used below are from Thomas Saerbeck's initial
    # commissioning of the PLATYPUS polarisation system.
    # TODO: Create a workflow that automatically fits
    # commissioning data with an efficiency function to be used
    # in the polarised_correction routine.

    # Define polariser efficiency as function of wavelength
    p1a = 0.993
    p1b = 0.57
    p1c = 0.47
    P1 = p1a - p1b * p1c**(wavelength).astype('float')

    # Define analyser efficiency as function of wavelength
    p2a = 0.993
    p2b = 0.57
    p2c = 0.51
    P2 = p2a - p2b * p2c**(wavelength).astype('float')

    # Define flipper1 and flipper2 efficiencies
    F1 = np.full((len(wavelength), 1), 0.003).astype('float')
    F2 = np.full((len(wavelength), 1), 0.003).astype('float')

    if I01 is None and I10 is not None:
        # For the case when only one spin-flip channel is recorded
        I01 = I10

    if I01 is None and I10 is None:
        # For the case when only R++ and R-- are recorded
        F2 = np.full((len(wavelength), 1), 0.000)
        P2 = np.full((len(wavelength), 1), 1.000)

    P1 = (1 + P1) / 2
    P2 = (1 + P2) / 2

    # Define polarised efficiency matrix
    polariser_eff_matrix = [
        np.array(
            [
                [(1 - p1l), 0, p1l, 0],
                [0, (1 - p1l), 0, p1l],
                [p1l, 0, (1 - p1l), 0],
                [0, p1l, 0, (1 - p1l)]
            ]
        ) for p1l in P1
    ]

    # Define analyser efficiency matrix
    analyser_eff_matrix = [
        np.array(
            [
                [(1 - p2l), p2l, 0, 0],
                [p2l, (1 - p2l), 0, 0],
                [0, 0, (1 - p2l), p2l],
                [0, 0, p2l, (1 - p2l)]
            ]
        ) for p2l in P2
    ]
    # Define flipper 1 efficiency matrix
    flipper1_eff_matrix = [
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [f1l, 0, (1 - f1l), 0],
                [0, f1l, 0, (1 - f1l)]
            ]
        ).astype('float') for f1l in F1
    ]
    # Define flipper 2 efficiency matrix
    flipper2_eff_matrix = [
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [f2l, 0, (1 - f2l), 0],
                [0, f2l, 0, (1 - f2l)]
            ]
        ).astype('float') for f2l in F2
    ]

    # Create efficiency matrices from individual
    # component efficiency matrices
    efficiencies_matrix = [
        flip1 @ flip2 @ pol @ anal for flip1, flip2, pol, anal in zip(
            flipper1_eff_matrix,
            flipper2_eff_matrix,
            polariser_eff_matrix,
            analyser_eff_matrix
        )
    ]

    efficiencies_matrix = np.array(efficiencies_matrix)
    # Invert efficiency matrices for each wavelength
    inv_eff_matrix = [
        np.linalg.inv(eff_mat) for eff_mat in efficiencies_matrix
    ]
    # Apply efficency matrices to processed PlatypusNexus spectra
    corrected_data = [
        corr_matrix @ [[dd_pt], [du_pt], [ud_pt], [uu_pt]]
        for (corr_matrix, dd_pt, du_pt, ud_pt, uu_pt) in zip(
            inv_eff_matrix, I00, I01, I10, I11
        )
    ]

    I00_polcorr = [corr_pt[0] for corr_pt in np.array(corrected_data)]
    I01_polcorr = [corr_pt[1] for corr_pt in np.array(corrected_data)]
    I10_polcorr = [corr_pt[2] for corr_pt in np.array(corrected_data)]
    I11_polcorr = [corr_pt[3] for corr_pt in np.array(corrected_data)]

    return I00_polcorr, I01_polcorr, I10_polcorr, I11_polcorr


class SpatzReduce(ReflectReduce):
    """
    Reduces Spatz reflectometer data to give the specular reflectivity.
    Offspecular data maps are also produced.

    Parameters
    ----------
    direct : string, hdf5 file-handle or SpatzNexus object
        A string containing the path to the direct beam hdf5 file,
        the hdf5 file itself, or a SpatzNexus object.
    data_folder : str, optional
        Where is the raw data stored?

    Examples
    --------

    >>> from refnx.reduce import SpatzReduce
    >>> reducer = SpatzReduce('SPZ0000711.nx.hdf')
    >>> datasets, reduced = reducer.reduce('SPZ0000711.nx.hdf',
    ...                                    rebin_percent=2)

    """

    def __init__(self, direct, data_folder=None, **kwds):
        super().__init__(direct, "SPZ", data_folder=data_folder)

    def _reduce_single_angle(self, scale=1):
        """
        Reduce a single angle.
        """
        n_spectra = self.reflected_beam.n_spectra
        n_tpixels = np.size(self.reflected_beam.m_topandtail, 1)
        n_xpixels = np.size(self.reflected_beam.m_topandtail, 2)

        m_twotheta = np.zeros((n_spectra, n_tpixels, n_xpixels))

        detrot_difference = (
            self.reflected_beam.detector_z - self.direct_beam.detector_z
        )

        # difference in pixels between reflected position and direct beam
        # at the two different detrots.
        QZ_PIXEL_SPACING = self.reflected_beam.cat.qz_pixel_size[0]
        dy = self.reflected_beam.detector_y

        # convert that pixel difference to angle (in small angle approximation)
        # higher `som` leads to lower m_beampos. i.e. higher two theta
        # is at lower pixel values
        beampos_2theta_diff = -(
            self.reflected_beam.m_beampos - self.direct_beam.m_beampos
        )
        beampos_2theta_diff *= QZ_PIXEL_SPACING / dy[0]
        beampos_2theta_diff = np.degrees(beampos_2theta_diff)

        total_2theta_deflection = detrot_difference + beampos_2theta_diff

        # omega_nom.shape = (N, )
        omega_nom = total_2theta_deflection / 2.0
        omega_corrected = omega_nom[:, np.newaxis]

        m_twotheta += np.arange(n_xpixels * 1.0)[np.newaxis, np.newaxis, :]
        m_twotheta -= self.direct_beam.m_beampos[:, np.newaxis, np.newaxis]
        # minus sign in following line because higher two theta is at lower
        # pixel values
        m_twotheta *= -QZ_PIXEL_SPACING / dy[:, np.newaxis, np.newaxis]
        m_twotheta = np.degrees(m_twotheta)
        m_twotheta += detrot_difference

        # you may be reflecting upside down, reverse the sign.
        upside_down = np.sign(omega_corrected[:, 0])
        m_twotheta *= upside_down[:, np.newaxis, np.newaxis]
        omega_corrected *= upside_down[:, np.newaxis]

        self.omega_corrected = omega_corrected
        self.m_twotheta = m_twotheta
        self.n_spectra = n_spectra


def reduce_stitch(
    reflect_list,
    direct_list,
    data_folder=None,
    prefix="PLP",
    trim_trailing=True,
    save=True,
    scale=1.0,
    reduction_options=None,
):
    """
    Reduces a list of reflected beam run numbers and a list of corresponding
    direct beam run numbers from the Platypus/Spatz reflectometers. If there
    are multiple reflectivity files they are spliced together.

    Parameters
    ----------
    reflect_list : list
        Reflected beam run numbers, e.g. `[708, 709, 710]`
        708 corresponds to the file PLP0000708.nx.hdf.
    direct_list : list
        Direct beam run numbers, e.g. `[711, 711, 711]`
    data_folder : str, optional
        Where is the raw data stored?
    prefix : str, optional
        The instrument filename prefix.
    trim_trailing : bool, optional
        When datasets are spliced together do you want to remove points in the
        overlap region from the preceding dataset?
    save : bool, optional
        If `True` then the spliced file is written to a file (in the working
        directory) with a name like: `c_PLP0000708.dat`.
    scale : float, optional
        Scales the data by this value.
    reduction_options : None, dict, or list of dict, optional
        Options passed directly to `refnx.reduce.PlatypusNexus.process`,
        for processing of individual spectra. Look at that method docstring
        for specification of options. If an individual dict then the same
        options are used to process all datasets. A list (or sequence) of
        dict can be used to specify different options for each datasets. If
        None, then a default set of reduction options will be used.

    Returns
    -------
    combined_dataset, reduced_filename : refnx.dataset.ReflectDataset, str
        The combined dataset and the file name of the reduced data, if it was
        saved. If it wasn't saved `reduced_filename` is `None`.

    Notes
    -----
    The `prefix` is used to specify the run numbers to a filename.
    For example a run number of 10, and a prefix of `PLP` resolves to a
    NeXus filename of 'PLP0000010.nx.hdf'.

    Examples
    --------

    >>> from refnx.reduce import reduce_stitch
    >>> dataset, fname = reduce_stitch([708, 709, 710],
    ...                                [711, 711, 711],
    ...                                reduction_options={"rebin_percent": 2})

    """
    options = [ReductionOptions()] * len(reflect_list)
    try:
        if reduction_options is not None:
            options = []
            for i in range(len(reflect_list)):
                if isinstance(reduction_options[i], dict):
                    options.append(reduction_options[i])
                else:
                    options.append(ReductionOptions())
    except KeyError:
        # reduction_options may be an individual dict
        if isinstance(reduction_options, dict):
            options = [reduction_options] * len(reflect_list)

    # now reduce all the files.
    zipped = zip(reflect_list, direct_list, options)

    combined_dataset = ReflectDataset()

    if data_folder is None:
        data_folder = os.getcwd()

    if prefix == "PLP":
        reducer_klass = PlatypusReduce
    elif prefix == "SPZ":
        reducer_klass = SpatzReduce
    else:
        raise ValueError("Incorrect prefix specified")

    for index, val in enumerate(zipped):
        reflect_datafile = os.path.join(
            data_folder, number_datafile(val[0], prefix=prefix)
        )
        direct_datafile = os.path.join(
            data_folder, number_datafile(val[1], prefix=prefix)
        )

        reducer = reducer_klass(direct_datafile)
        datasets, fnames = reducer.reduce(
            reflect_datafile, save=save, **val[2]
        )

        if not index:
            datasets[0].scale(scale)

        combined_dataset.add_data(
            datasets[0].data, requires_splice=True, trim_trailing=trim_trailing
        )

    fname_dat = None

    if save:
        # this will give us <fname>.nx.hdf
        # if reflect_list was an integer you'll get PLP0000708.nx.hdf
        fname = number_datafile(reflect_list[0], prefix=prefix)
        # now chop off .nx.hdf extension
        fname = basename_datafile(fname)

        fname_dat = "c_{0}.dat".format(fname)
        with open(fname_dat, "wb") as f:
            combined_dataset.save(f)
        fname_xml = "c_{0}.xml".format(fname)
        with open(fname_xml, "wb") as f:
            combined_dataset.save_xml(f)

    return combined_dataset, fname_dat


class AutoReducer:
    """
    Auto-reduces reflectometry data.

    Watches a datafolder for new/modified NeXUS files and reduces them.

    Parameters
    ----------
    direct_beams: list of {str, h5data}
        list of str, or list of h5py file handles pointing to
        direct beam runs

    scale: float or array-like
        Scale factors corresponding to each direct beam.

    reduction_options: dict, or list of dict
        Specifies the reduction options for each of the direct beams.
        A default set of options is provided by
        `refnx.reduce.ReductionOptions`.

    data_folder: {str, Path}
        Path to the data folder containing the data to be reduced.

    Notes
    -----
    Requires that the 'watchdog' package be installed. Starts two threads that
    are responsible for doing the reduction.
    """

    def __init__(
        self, direct_beams, scale=1, reduction_options=None, data_folder="."
    ):
        from watchdog.observers import Observer
        from refnx.reduce._auto_reduction import NXEH

        self.data_folder = data_folder

        # deal with reduction options first
        options = [ReductionOptions()] * len(direct_beams)
        try:
            if reduction_options is not None:
                options = []
                for i in range(len(direct_beams)):
                    if isinstance(reduction_options[i], dict):
                        options.append(reduction_options[i])
                    else:
                        options.append(ReductionOptions())
        except KeyError:
            # reduction_options may be an individual dict
            if isinstance(reduction_options, dict):
                options = [reduction_options] * len(direct_beams)

        # deal with scale factors
        scale_factors = np.broadcast_to(scale, len(direct_beams))
        scale_factors = scale_factors.astype(np.float)

        zipped = zip(direct_beams, options, scale_factors)

        # work out what type of instrument you have
        d0 = direct_beams[0]
        self.redn_klass = PlatypusReduce
        self.reflect_klass = PlatypusNexus
        if (isinstance(d0, str) and d0.startswith("SPZ")) or (
            isinstance(d0, h5py.File) and d0.filename.startswith("SPZ")
        ):
            self.redn_klass = SpatzReduce
            self.reflect_klass = SpatzNexus

        self.direct_beams = {}
        for direct_beam, ro, scale_factor in zipped:
            rn = self.reflect_klass(direct_beam)
            db = self.redn_klass(rn)
            fname = os.path.basename(rn.cat.filename)
            self.direct_beams[fname] = {
                "reflectnexus": rn,
                "reducer": db,
                "collimation": np.r_[rn.cat.ss_coll1, rn.cat.ss_coll2],
                "reduction_options": ro,
                "scale": scale_factor,
            }

        self.redn_cache = {}
        self._redn_cache_tbl = pd.DataFrame(
            columns=["fname", "sample_name", "omega"]
        )

        # start watching the data_folder
        self.queue = Queue()

        event_handler = NXEH(self.queue)
        observer = Observer()
        observer.schedule(event_handler, path=self.data_folder)
        observer.start()

        self.worker = Thread(target=self)
        self.worker.setDaemon(True)
        self.worker.start()

    def __call__(self):
        while True:
            if not self.queue.empty():
                # file might still be being written by SICS? allow a bit of
                # time for it to complete.
                time.sleep(1.5)

                event = self.queue.get()
                # print(event.src_path)
                rb = self.reflect_klass(event.src_path)
                fname = os.path.basename(rb.cat.filename)
                db = self.match_direct_beam(rb)

                if db is not None:
                    # the reduction
                    entry = self.direct_beams[db]
                    reducer = entry["reducer"]
                    opts = entry["reduction_options"]
                    scale = entry["scale"]
                    try:
                        datasets, _ = reducer.reduce(rb, scale=scale, **opts)
                    except Exception as e:
                        # don't want to stop reducing if there is an error
                        # somewhere
                        print(e)
                        continue

                    print(f"Reduced: {fname}")

                    for i, dataset in enumerate(datasets):
                        dataset.filename = f"{fname.rstrip('.nx.hdf')}_{i}.dat"

                    # save the reduced files in a cache
                    sample_name = rb.cat.sample_name.tobytes()
                    sample_name = sample_name.decode("utf-8")[:-1]

                    omega = float(rb.cat.omega[0])
                    self.redn_cache[fname] = {
                        "datasets": datasets,
                        "sample_name": sample_name,
                        "omega": omega,
                    }

                    data = {
                        "fname": [fname],
                        "sample_name": [sample_name],
                        "omega": omega,
                    }
                    if not self._redn_cache_tbl.fname.str.match(fname).any():
                        # see if the entry is already in the _redn_cache_tbl
                        # if it is, then you don't want to add it again
                        entry = pd.DataFrame(data=data)
                        self._redn_cache_tbl = self._redn_cache_tbl.append(
                            entry
                        )

                    # now splice matching datasets
                    ds = self.match_datasets(self.redn_cache[fname])
                    if len(ds) > 1:
                        try:
                            c = self.splice_datasets(ds)
                        except Exception as e:
                            print(e)
                            continue
                        print(
                            f"Combined into: {c}, {[d.filename for d in ds]}"
                        )
            else:
                time.sleep(5.0)

    def match_direct_beam(self, rb):
        """
        Finds the direct beam associated with a reflection measurement.
        Matching is done by finding identical collimation conditions.

        Parameters
        ----------
        rb: {PlatypusNexus, SpatzNexus}
            The reflectometry run.

        Returns
        -------
        db: str
            The direct beam file name that matches the reflection measurement
            This is used to look up an entry in `AutoReducer.direct_beams`.
        """
        # the reflected-direct beam match is done via slit sizes
        # (not infallible)
        collimation = np.r_[rb.cat.ss_coll1, rb.cat.ss_coll2]
        for k, v in self.direct_beams.items():
            if np.allclose(collimation, v["collimation"], atol=0.01):
                return k
        return None

    def match_datasets(self, dct):
        """
        Finds all the datasets in `AutoReducer.redn_cache` that share an
        *identical* `dct["sample_name"]`, but may have been measured at
        different angles.

        Parameters
        ----------
        dct: dict
            dct.keys() = ["fname", "sample_name", "omega"]

        Returns
        -------
        datasets: list of `refnx.dataset.Data1D`
            Datasets that share the same sample_name as `dct['sample_name']`
        """
        tbl = self._redn_cache_tbl
        # exact match
        fnames = tbl.fname[
            tbl.sample_name.str.match(f"^{dct['sample_name']}$")
        ]
        datasets = [self.redn_cache[fname]["datasets"][0] for fname in fnames]

        return datasets

    def splice_datasets(self, ds):
        """
        Combines datasets together.

        Parameters
        ----------
        ds: list of `refnx.dataset.Data1D`
            The datasets to splice together.

        Returns
        -------
        fname: str
            The name of the combined dataset

        Notes
        -----
        The combined dataset is saved as `f"c_{d.filename}.dat"`,
        where d is the dataset with the lowest average Q value
        from ds.
        """
        appended_ds = ReflectDataset()

        datasets = []
        average_q = []
        for d in ds:
            dataset = ReflectDataset(d)
            average_q.append(np.mean(dataset.x))
            datasets.append(dataset)

        idxs = np.argsort(average_q)
        # sort datasets according to average Q.
        datasets = [d for _, d in sorted(zip(idxs, datasets))]

        for dataset in datasets:
            appended_ds += dataset

        fname = datasets[0].filename.rstrip(".dat")
        fname = fname.split("_")[0]
        fname = f"c_{fname}.dat"
        appended_ds.save(fname)
        return fname


if __name__ == "__main__":
    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

    a = reduce_stitch([708, 709, 710], [711, 711, 711], rebin_percent=2)

    a.save("test1.dat")

    print(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))
