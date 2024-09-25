"""
An experimental simulator for a TOF neutron reflectometer
"""

__author__ = "Andrew Nelson"
__copyright__ = "Copyright 2019, Andrew Nelson"
__license__ = "3 clause BSD"

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.stats import rv_continuous, trapezoid, norm, uniform
from scipy.optimize import brentq
from scipy._lib._util import check_random_state

from refnx.reduce.platypusnexus import (
    calculate_wavelength_bins,
    create_reflect_nexus,
    PlatypusNexus,
)
from refnx.reduce import parabolic_motion as pm
from refnx.util import general, ErrorProp
from refnx.util._resolution_kernel import AngularDivergence
from refnx.reflect import Slab, Structure, SLD, ReflectModel
from refnx.dataset import ReflectDataset


class SpectrumDist(rv_continuous):
    """
    The `SpectrumDist` object is a `scipy.stats` like object to describe the
    neutron intensity as a function of wavelength. You can use the `pdf, cdf,
    ppf, rvs` methods like you would a `scipy.stats` distribution. Of
    particular interest is the `rvs` method which randomly samples neutrons
    whose distribution obeys the direct beam spectrum. Random variates are
    generated the `rv_continuous` superclass by classical generation of
    uniform noise coupled with the `ppf`. `ppf` is approximated by linear
    interpolation of `q` into a pre-calculated inverse `cdf`.
    """

    def __init__(self, x, y):
        super(SpectrumDist, self).__init__(a=np.min(x), b=np.max(x))
        self._x = x

        # normalise the distribution
        area = simpson(y, x=x)
        y /= area
        self._y = y

        # an InterpolatedUnivariate spline models the spectrum
        self.spl = IUS(x, y)

        # fudge_factor required because integral of the spline is not exactly 1
        self.fudge_factor = self.spl.integral(self.a, self.b)

        # calculate a gridded and sampled version of the CDF.
        # this can be used with interpolation for quick calculation
        # of ppf (necessary for quick rvs)
        self._x_interpolated_cdf = np.linspace(np.min(x), np.max(x), 1000)
        self._interpolated_cdf = self.cdf(self._x_interpolated_cdf)

    def _pdf(self, x):
        return self.spl(x) / self.fudge_factor

    def _cdf(self, x):
        xflat = x.ravel()

        def f(x):
            return self.spl.integral(self.a, x) / self.fudge_factor

        v = map(f, xflat)

        r = np.fromiter(v, dtype=float).reshape(x.shape)
        return r

    def _f(self, x, qq):
        return self._cdf(x) - qq

    def _g(self, qq, *args):
        return brentq(self._f, self._a, self._b, args=(qq,) + args)

    def _ppf(self, q, *args):
        qflat = q.ravel()
        """
        _a, _b = self._get_support(*args)

        def f(x, qq):
            return self._cdf(x) - qq

        def g(qq):
            return brentq(f, _a, _b, args=(qq,) + args, xtol=1e-3)

        v = map(g, qflat)

        cdf = _CDF(self.spl, self.fudge_factor, _a, _b)
        g = _G(cdf)

        with Pool() as p:
            v = p.map(g, qflat)
            r = np.fromiter(v, dtype=float).reshape(q.shape)
        """
        # approximate the ppf using a sampled+interpolated CDF
        # the commented out methods are more accurate, but are at least
        # 3 orders of magnitude slower.
        r = np.interp(qflat, self._interpolated_cdf, self._x_interpolated_cdf)
        return r.reshape(q.shape)


# for parallelisation (can't pickle rv_continuous all that easily)
class _CDF(object):
    def __init__(self, spl, fudge_factor, a, b):
        self.a = a
        self.b = b
        self.spl = spl
        self.fudge_factor = fudge_factor

    def __call__(self, x):
        return self.spl.integral(self.a, x) / self.fudge_factor


class _G(object):
    def __init__(self, cdf):
        self.cdf = cdf

    def _f(self, x, qq):
        return self.cdf(x) - qq

    def __call__(self, q):
        return brentq(self._f, self.cdf.a, self.cdf.b, args=(q,), xtol=1e-4)


class ReflectSimulator(object):
    """
    Simulate a reflectivity pattern from PLATYPUS.

    Parameters
    ----------
    model : refnx.reflect.ReflectModel

    angle : float
        Angle of incidence (degrees)

    p_theta : AngularDivergence

    L2S : float
        Distance from pre-sample slit to sample

    direct_spectrum : str, ReflectNexus, h5 handle
        Contains the direct beam spectrum. Processed using ReflectNexus

    lo_wavelength : float
        smallest wavelength used from the generated neutron spectrum

    hi_wavelength : float
        longest wavelength used from the generated neutron spectrum

    dlambda : float
        Wavelength resolution expressed as a percentage. dlambda=3.3
        corresponds to using disk choppers 1+3 on *PLATYPUS*.
        (FWHM of the Gaussian approximation of a trapezoid)

    rebin : float
        Rebinning expressed as a percentage. The width of a wavelength bin is
        `rebin / 100 * lambda`. You have to multiply by 0.68 to get its
        fractional contribution to the overall resolution smearing.

    gravity : bool
        Apply gravity during simulation? Turn gravity off if `angle` is
        already an array of incident angles due to gravity.

    force_gaussian : bool
        Instead of using trapzeoidal and uniform distributions for angular
        and wavelength resolution, use a Gaussian distribution (doesn't apply
        to the rebinning contribution).

    force_uniform_wavelength : bool
        Instead of using a wavelength spectrum representative of a
        time-of-flight reflectometer generate wavelengths from a uniform
        distribution.

    only_resolution : bool
        Set to `True` if this class is only going to be used to calculate
        detailed resolution kernels.

    Notes
    -----
    Angular, chopper and rebin smearing effects are all taken into account.
    """

    def __init__(
        self,
        model,
        angle,
        p_theta,
        L2S=120.0,
        direct_spectrum=None,
        lo_wavelength=2.8,
        hi_wavelength=18,
        dlambda=3.3,
        rebin=2,
        gravity=False,
        force_gaussian=False,
        force_uniform_wavelength=False,
        only_resolution=False,
    ):
        self.model = model

        self.bkg = model.bkg.value
        self.angle = angle
        self.p_theta = p_theta
        assert isinstance(p_theta, AngularDivergence)

        # dlambda refers to the FWHM of the gaussian approximation to a uniform
        # distribution. The full width of the uniform distribution is
        # dlambda/0.68.
        self.dlambda = dlambda / 100.0
        # the rebin percentage refers to the full width of the bins. You have to
        # multiply this value by 0.68 to get the equivalent contribution to the
        # resolution function.
        self.rebin = rebin / 100.0
        self.wavelength_bins = calculate_wavelength_bins(
            lo_wavelength, hi_wavelength, rebin
        )
        bin_centre = 0.5 * (
            self.wavelength_bins[1:] + self.wavelength_bins[:-1]
        )

        # angular deviation due to gravity
        # --> no correction for gravity affecting width of angular resolution
        elevations = 0
        if gravity:
            speeds = general.wavelength_velocity(bin_centre)
            # trajectories through slits for different wavelengths
            trajectories = pm.find_trajectory(p_theta.L12 / 1000.0, 0, speeds)
            # elevation at sample
            elevations = pm.elevation(
                trajectories, speeds, (p_theta.L12 + L2S) / 1000.0
            )

        # nominal Q values
        self.q = general.q(angle - elevations, bin_centre)

        # keep a tally of the direct and reflected beam
        self.direct_beam = np.zeros((self.wavelength_bins.size - 1))
        self.reflected_beam = np.zeros((self.wavelength_bins.size - 1))

        # beam monitor counts for normalisation
        self.bmon_direct = 0
        self.bmon_reflect = 0

        self.gravity = gravity

        # wavelength generator
        self.force_uniform_wavelength = force_uniform_wavelength
        if force_uniform_wavelength:
            self.spectrum_dist = uniform(
                loc=lo_wavelength - 1, scale=hi_wavelength - lo_wavelength + 1
            )
        else:
            a = create_reflect_nexus(direct_spectrum)
            direct = False
            if isinstance(a, PlatypusNexus):
                direct = True

            q, i, di = a.process(
                normalise=False,
                normalise_bins=False,
                rebin_percent=0.5,
                lo_wavelength=max(0, lo_wavelength - 1),
                hi_wavelength=hi_wavelength + 1,
                direct=direct,
            )
            q = q.squeeze()
            i = i.squeeze()
            self.spectrum_dist = SpectrumDist(q, i)

        self.force_gaussian = force_gaussian
        self.only_resolution = only_resolution

        # angular resolution generator, based on a trapezoidal distribution
        # The slit settings are the optimised set typically used in an
        # experiment. dtheta/theta refers to the FWHM of a Gaussian
        # approximation to a trapezoid.

        # stores the q vectors contributing towards each datapoint
        self._res_kernel = {}
        self._min_samples = 0

        self.angle = angle
        self.L12 = p_theta.L12
        self.L2S = L2S
        self.div = p_theta.dtheta
        self.s1 = p_theta.d1
        self.s2 = p_theta.d2

        if force_gaussian:
            self.angular_dist = norm(scale=self.div / 2.3548)
        else:
            self.angular_dist = p_theta.rv

    def sample(self, samples, random_state=None):
        """
        Sample the beam for reflected signal.

        2400000 samples roughly corresponds to 1200 sec of *PLATYPUS* using
        dlambda=3.3 and dtheta=3.3 at angle=0.65.
        150000000 samples roughly corresponds to 3600 sec of *PLATYPUS* using
        dlambda=3.3 and dtheta=3.3 at angle=3.0.

        (The sample number <--> actual acquisition time correspondence has
         not been checked fully)

        Parameters
        ----------
        samples: int
            How many samples to run.
        random_state: {int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            If `random_state` is not specified the
            `~np.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with seed.
            If `random_state` is already a ``RandomState`` or a ``Generator``
            instance, then that object is used.
            Specify `random_state` for repeatable minimizations.
        """
        # grab a random number generator
        rng = check_random_state(random_state)

        # generate neutrons of various wavelengths
        wavelengths = self.spectrum_dist.rvs(size=samples, random_state=rng)

        # generate neutrons of different angular divergence
        angles = self.angular_dist.rvs(samples, random_state=rng) + self.angle

        # angular deviation due to gravity
        # --> no correction for gravity affecting width of angular resolution
        if self.gravity:
            speeds = general.wavelength_velocity(wavelengths)
            # trajectories through slits for different wavelengths
            trajectories = pm.find_trajectory(self.L12 / 1000.0, 0, speeds)
            # elevation at sample
            elevations = pm.elevation(
                trajectories, speeds, (self.L12 + self.L2S) / 1000.0
            )
            angles -= elevations

        # calculate Q
        q = general.q(angles, wavelengths)

        # calculate reflectivities for a neutron of a given Q.
        # the angular resolution smearing has already been done. The wavelength
        # resolution smearing follows.
        if not self.only_resolution:
            r = self.model(q, x_err=0.0)

            # accept or reject neutrons based on the reflectivity of
            # sample at a given Q.
            criterion = rng.uniform(size=samples)
            accepted = criterion < r
        else:
            accepted = np.ones_like(q, dtype=bool)

        # implement wavelength smearing from choppers. Jitter the wavelengths
        # by a uniform distribution whose full width is dlambda / 0.68.
        if self.force_gaussian:
            noise = rng.standard_normal(size=samples)
            jittered_wavelengths = wavelengths * (
                1 + self.dlambda / 2.3548 * noise
            )
        else:
            noise = rng.uniform(-0.5, 0.5, size=samples)
            jittered_wavelengths = wavelengths * (
                1 + self.dlambda / 0.68 * noise
            )

        # update reflected beam counts. Rebin smearing
        # is taken into account due to the finite size of the wavelength
        # bins.
        hist = np.histogram(
            jittered_wavelengths[accepted], self.wavelength_bins
        )
        self.reflected_beam += hist[0]
        self.bmon_reflect += float(samples)

        # update resolution kernel. If we have more than 100000 in all
        # bins skip
        if (
            len(self._res_kernel)
            and np.min([len(v) for v in self._res_kernel.values()]) > 1000000
        ):
            return

        bin_loc = np.digitize(jittered_wavelengths, self.wavelength_bins)
        for i in range(1, len(self.wavelength_bins)):
            # extract q values that fall in each wavelength bin
            q_for_bin = np.copy(q[bin_loc == i])
            q_samples_so_far = self._res_kernel.get(i - 1, np.array([]))
            updated_samples = np.concatenate((q_samples_so_far, q_for_bin))

            # no need to keep double precision for these sample arrays
            self._res_kernel[i - 1] = updated_samples.astype(np.float32)

    def sample_direct(self, samples, random_state=None):
        """
        Samples the direct beam

        Parameters
        ----------
        samples: int
            How many samples to run.
        random_state: {int, `~np.random.RandomState`, `~np.random.Generator`}, optional
            If `random_state` is not specified the
            `~np.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is used,
            seeded with seed.
            If `random_state` is already a ``RandomState`` or a ``Generator``
            instance, then that object is used.
            Specify `random_state` for repeatable minimizations.
        """
        # grab a random number generator
        rng = check_random_state(random_state)

        # generate neutrons of various wavelengths
        wavelengths = self.spectrum_dist.rvs(size=samples, random_state=rng)

        # implement wavelength smearing from choppers. Jitter the wavelengths
        # by a uniform distribution whose full width is dlambda / 0.68.
        if self.force_gaussian:
            noise = rng.standard_normal(size=samples)
            jittered_wavelengths = wavelengths * (
                1 + self.dlambda / 2.3548 * noise
            )
        else:
            noise = rng.uniform(-0.5, 0.5, size=samples)
            jittered_wavelengths = wavelengths * (
                1 + self.dlambda / 0.68 * noise
            )

        hist = np.histogram(jittered_wavelengths, self.wavelength_bins)

        self.direct_beam += hist[0]
        self.bmon_direct += float(samples)

    @property
    def resolution_kernel(self):
        histos = []
        # first histogram all the q values corresponding to a specific bin
        # this will come as shortest wavelength first, or highest Q. This
        # is because the wavelength bins are monotonic increasing.
        for v in self._res_kernel.values():
            histos.append(np.histogram(v, density=True, bins="auto"))

        # make lowest Q comes first.
        histos.reverse()

        # what is the largest number of bins?
        max_bins = np.max([len(histo[0]) for histo in histos])

        kernel = np.full((len(histos), 2, max_bins), np.nan)
        for i, histo in enumerate(histos):
            p, x = histo
            sz = len(p)
            kernel[i, 0, :sz] = 0.5 * (x[:-1] + x[1:])
            kernel[i, 1, :sz] = p

        # filter points with zero counts because error is incorrect
        mask = self.reflected_beam != 0
        mask = mask[::-1]

        kernel = kernel[mask]

        return kernel

    @property
    def reflectivity(self):
        """
        The reflectivity of the sampled system
        """
        if self.only_resolution:
            raise RuntimeError(
                "ReflectSimulator is only calculating resolution function"
            )

        rerr = np.sqrt(self.reflected_beam)
        bmon_reflect_err = np.sqrt(self.bmon_reflect)

        ierr = np.sqrt(self.direct_beam)
        bmon_direct_err = np.sqrt(self.bmon_direct)

        # reverse the resolution kernel, because that's output in sorted order
        dx = self.resolution_kernel[::-1]

        # divide reflectivity signal by bmon
        ref, rerr = ErrorProp.EPdiv(
            self.reflected_beam, rerr, self.bmon_reflect, bmon_reflect_err
        )
        # divide direct signal by bmon
        direct, ierr = ErrorProp.EPdiv(
            self.direct_beam, ierr, self.bmon_direct, bmon_direct_err
        )

        # now calculate reflectivity
        ref, rerr = ErrorProp.EPdiv(ref, rerr, direct, ierr)

        # filter points with zero counts because error is incorrect
        mask = rerr != 0

        dataset = ReflectDataset(
            data=(self.q[mask], ref[mask], rerr[mask], dx)
        )

        dataset.sort()
        # apply some counting statistics on top of dataset otherwise there will
        # be no variation at e.g. critical edge.
        # return dataset.synthesise()
        return dataset
