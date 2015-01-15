from __future__ import division
import numpy as np
import scipy
import scipy.linalg
from scipy.signal import convolve, fftconvolve
from scipy.interpolate import interp1d
from .curvefitter import CurveFitter
import refnx.util.ErrorProp as EP
import warnings
import math


try:
    import _creflect as refcalc
except ImportError:
    print('WARNING, Using slow reflectivity calculation')
    import _reflect as refcalc

# some definitions for resolution smearing
_FWHM = 2 * np.sqrt(2 * np.log(2.0))
_INTLIMIT = 3.5

def convert_coefs_to_layer_format(coefs):
    nlayers = int(coefs[0])
    w = np.zeros((nlayers + 2, 4), np.float64)
    w[0, 1: 3] = coefs[2: 4]
    w[-1, 1: 3] = coefs[4: 6]
    w[-1, 3] = coefs[7]
    if nlayers:
        w[1: -1] = np.array(coefs[8: ]).reshape(nlayers, 4)

    return w

def convert_layer_format_to_coefs(layers, scale=1, bkg=0):
    nlayers = np.size(layers, 0) - 2
    coefs = np.zeros(4 * nlayers + 8, np.float64)
    coefs[0] = nlayers
    coefs[1] = scale
    coefs[2:4] = layers[0, 1: 3]
    coefs[4: 6] = layers[-1, 1: 3]
    coefs[6] = bkg
    coefs[7] = layers[-1, 3]
    if nlayers:
        coefs[8:] = layers.flatten()[4: -4]

    return coefs

def parameter_names(coefs):
    names = ['nlayers', 'scale', 'SLDfront', 'iSLDfront', 'SLDback',
             'iSLDback', 'bkg', 'sigma_back']
    nlayers = (coefs.size - 8 / 4)
    for i in range(int(nlayers)):
        names.append('thick%d'%(i + 1))
        names.append('SLD%d'%(i + 1))
        names.append('iSLD%d'%(i + 1))
        names.append('sigma%d'%(i + 1))
    return names
    
def abeles(q, coefs, *args, **kwds):
    """
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------

    q : np.ndarray
        The qvalues required for the calculation. Q=4*Pi/lambda * sin(omega).
        Units = Angstrom**-1

    coefs : np.ndarray
        coefs[0] = number of layers, N
        coefs[1] = scale factor
        coefs[2] = SLD of fronting (/1e-6 Angstrom**-2)
        coefs[3] = iSLD of fronting (/1e-6 Angstrom**-2)
        coefs[4] = SLD of backing
        coefs[5] = iSLD of backing
        coefs[6] = background
        coefs[7] = roughness between backing and layer N

        coefs[4 * (N - 1) + 8] = thickness of layer N in Angstrom (layer 1 is
        closest to fronting)
        coefs[4 * (N - 1) + 9] = SLD of layer N
        coefs[4 * (N - 1) + 10] = iSLD of layer N
        coefs[4 * (N - 1) + 11] = roughness between layer N and N-1.


    kwds : dict, optional
        The following keys are used:

        'dqvals' - float or np.ndarray, optional
            If dqvals is a float, then a constant dQ/Q resolution smearing is
            employed.  For 5% resolution smearing supply 5.
            If `dqvals` is the same shape as q, then the array contains the
            FWHM of a Gaussian approximated resolution kernel. Point by point
            resolution smearing is employed.  Use this option if dQ/Q varies
            across your dataset.
            If `dqvals.ndim == q.ndim + 2` and
            `q.shape == dqvals[..., -3].shape` then an individual resolution
            kernel is applied to each measurement point.  This resolution kernel
            is a probability distribution function (PDF). `dqvals` will have the
            shape (qvals.shape, M, 2).  There are `M` points in the kernel.
            `dqvals[..., 0]` holds the q values for the kernel, `dqvals[..., 1]`
            gives the corresponding probability.

        'quad_order' - int, optional
            the order of the Gaussian quadrature polynomial for doing the
            resolution smearing. default = 17. Don't choose less than 13. If
            quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
            quadrature will always work, but takes a _long_ time (2 or 3 orders
            of magnitude longer). Fixed quadrature will always take a lot less
            time. BUT it won't necessarily work across all samples. For example,
            13 points may be fine for a thin layer, but will be atrocious at
            describing a multilayer with bragg peaks.
    """

    qvals = q
    quad_order = 17
    scale = coefs[1]
    bkg = coefs[6]

    if not is_proper_Abeles_input(coefs):
        raise ValueError('The size of the parameter array passed to abeles'
                         ' should be 4 * coefs[0] + 8')

    #make into form suitable for reflection calculation
    w = convert_coefs_to_layer_format(coefs)

    if 'quad_order' in kwds:
        quad_order = kwds['quad_order']

    if 'dqvals' in kwds and kwds['dqvals'] is not None:
        dqvals = kwds['dqvals']

        # constant dq/q smearing
        if type(dqvals) == float:
            return (scale * _smeared_abeles_constant(qvals, w, dqvals)) + bkg

        # point by point resolution smearing
        if dqvals.size == qvals.size:
            dqvals_flat = dqvals.flatten()
            qvals_flat = q.flatten()

            # adaptive quadrature
            if quad_order == 'ultimate':
                smeared_rvals = (scale *
                                 _smeared_abeles_adaptive(qvals_flat,
                                                          w,
                                                          dqvals_flat)
                                 + bkg)
                return smeared_rvals.reshape(q.shape)
            # fixed order quadrature
            else:
                smeared_rvals = (scale * _smeared_abeles_fixed(
                                                      qvals_flat,
                                                      w,
                                                      dqvals_flat,
                                                      quad_order=quad_order)
                                 + bkg)
                return np.reshape((smeared_rvals), q.shape)

        # resolution kernel smearing
        elif (dqvals.ndim == qvals.ndim + 2
              and dqvals.shape[0: qvals.ndim] == qvals.shape):
            #TODO may not work yet.
            qvals_for_res = dqvals[..., 0]
            # work out the reflectivity at the kernel evaluation points
            smeared_rvals = refcalc.abeles(qvals_for_res, w, scale=coefs[1],
                                           bkg=coefs[6])

            #multiply by probability
            smeared_rvals *= dqvals[..., 1]

            #now do simpson integration
            return scipy.integrate.simps(smeared_rvals, x=dqvals[..., 0])

    else:
        # no smearing
        return refcalc.abeles(q, w, scale=coefs[1], bkg=coefs[6])


def _memoize_gl(f):
    cache = {}
    def inner(n):
        if n in cache:
            return cache[n]
        else:
            result = cache[n] = f(n)
            return result
    return inner

@_memoize_gl
def gauss_legendre(n):
    """
    Calculate gaussian quadrature abscissae and weights
    Returns
    -------
    (x, w) : tuple
        The abscissae and weights for Gauss Legendre integration.
    """
    k = np.arange(1.0, n)
    a_band = np.zeros((2, n))
    a_band[1, 0: n - 1] = k / np.sqrt(4 * k * k - 1)
    x, V = scipy.linalg.eig_banded(a_band, lower=True)
    w = 2 * np.real(np.power(V[0, :], 2))
    return x, w

def _smearkernel(x, w, q, dq):
    '''
    Kernel for Gaussian Quadrature
    '''
    prefactor = 1 / np.sqrt(2 * np.pi)
    gauss = prefactor * np.exp(-0.5 * x * x)
    localq = q + x * dq / _FWHM
    return refcalc.abeles(localq, w) * gauss

def _smeared_abeles_adaptive(qvals, w, dqvals):
    # adaptive gaussian quadrature smearing
    smeared_rvals = np.zeros(qvals.size)
    warnings.simplefilter('ignore', Warning)
    for idx, val in enumerate(qvals):
        smeared_rvals[idx], err = scipy.integrate.quadrature(
            _smearkernel,
            -_INTLIMIT,
            _INTLIMIT,
            tol=2 * np.finfo(np.float64).eps,
            rtol=2 * np.finfo(np.float64).eps,
            args=(w, qvals[idx], dqvals[idx]))

    warnings.resetwarnings()
    return smeared_rvals

def _smeared_abeles_fixed(qvals, w, dqvals, quad_order=17):
    # fixed order gaussian quadrature smearing
    # get the gauss-legendre weights and abscissae
    abscissa, weights = gauss_legendre(quad_order)
    # get the normal distribution at that point
    prefactor = 1. / np.sqrt(2 * np.pi)
    gauss = lambda x: np.exp(-0.5 * x * x)
    gaussvals = prefactor * gauss(abscissa * _INTLIMIT)

    # integration between -3.5 and 3.5 sigma
    va = qvals - _INTLIMIT * dqvals / _FWHM
    vb = qvals + _INTLIMIT * dqvals / _FWHM

    va = va[:, np.newaxis]
    vb = vb[:, np.newaxis]

    qvals_for_res = ((np.atleast_2d(abscissa) *
                     (vb - va)
                     + vb + va) / 2.)
    smeared_rvals = refcalc.abeles(qvals_for_res.flatten(), w)

    smeared_rvals = np.reshape(smeared_rvals,
                               (qvals.size, abscissa.size))

    smeared_rvals *= np.atleast_2d(gaussvals * weights)
    return np.sum(smeared_rvals, 1) * _INTLIMIT

def _smeared_abeles_constant(q, w, resolution):
    # constant dq/q resolution smearing.
    resolution /= 100
    gaussnum = 51
    gaussgpoint = (gaussnum - 1) / 2

    gauss = lambda x, s: (1. / s / np.sqrt(2 * np.pi)
                          * np.exp(-0.5 * x**2 / s / s))

    lowQ = np.min(q)
    highQ = np.max(q)
    if lowQ <= 0:
        lowQ = 1e-6

    start = np.log10(lowQ) - 6 * resolution / _FWHM
    finish = np.log10(highQ * (1 + 6 * resolution / _FWHM))
    interpnum = np.round(np.abs(1 * (np.abs(start - finish))
                                / (1.7 * resolution / _FWHM / gaussgpoint)))
    xtemp = np.linspace(start, finish, interpnum)
    xlin = np.power(10., xtemp)

    gauss_x = np.linspace(-1.7 * resolution, 1.7 * resolution, gaussnum)
    gauss_y = gauss(gauss_x, resolution / _FWHM)

    rvals = refcalc.abeles(xlin, w)
    smeared_rvals = fftconvolve(rvals, gauss_y, mode='same')
    interpolator = interp1d(xlin, smeared_rvals)

    smeared_output = interpolator(q)
    # smeared_output *= np.sum(gauss_y)
    smeared_output *= gauss_x[1] - gauss_x[0]
    return smeared_output

def is_proper_Abeles_input(coefs):
    '''
    Test to see if the coefs array is suitable input for the abeles function
    '''
    if np.size(coefs, 0) != 4 * int(coefs[0]) + 8:
        return False
    return True

def sld_profile(coefs, z):

    nlayers = int(coefs[0])
    summ = np.zeros_like(z)
    summ += coefs[2]
    thick = 0

    for idx, zed in enumerate(z):
        dist = 0
        for ii in range(nlayers + 1):
            if ii == 0:
                if nlayers:
                    deltarho = -coefs[2] + coefs[9]
                    thick = 0
                    sigma = np.fabs(coefs[11])
                else:
                    sigma = np.fabs(coefs[7])
                    deltarho = -coefs[2] + coefs[4]
            elif ii == nlayers:
                SLD1 = coefs[4 * ii + 5]
                deltarho = -SLD1 + coefs[4]
                thick = np.fabs(coefs[4 * ii + 4])
                sigma = np.fabs(coefs[7])
            else:
                SLD1 = coefs[4 * ii + 5]
                SLD2 = coefs[4 * (ii + 1) + 5]
                deltarho = -SLD1 + SLD2
                thick = np.fabs(coefs[4 * ii + 4])
                sigma = np.fabs(coefs[4 * (ii + 1) + 7])

            dist += thick

            # if sigma=0 then the computer goes haywire (division by zero), so
            # say it's vanishingly small
            if sigma == 0:
                sigma += 1e-3

            #summ += deltarho * (norm.cdf((zed - dist)/sigma))
            summ[idx] += deltarho * \
                (0.5 + 0.5 * math.erf((zed - dist) / (sigma * np.sqrt(2.))))

    return summ


class ReflectivityFitter(CurveFitter):

    '''
        A sub class of refnx.analysis.fitting.CurveFitter suited for
        fitting reflectometry data.

        If you wish to fit analytic profiles you should subclass this class,
        overriding the model() method.  If you do this you should also
        override the sld_profile method of ReflectivityFitter.
    '''

    def __init__(self, xdata, ydata, parameters, edata=None, fcn_args=(),
                 fcn_kws=None, kws=None):
        '''
        Initialises the ReflectivityFitter.
        See the constructor of the CurveFitter for more details, especially the
        entries in kwds that are used.

        Parameters
        ----------
        xdata : np.ndarray
            The independent variables
        ydata : np.ndarray
            The dependent (observed) variable
        parameters : lmfit.Parameters instance
            Specifies the parameter set for the fit
        edata : np.ndarray, optional
            The measured uncertainty in the dependent variable, expressed as
            sd.  If this array is not specified, then edata is set to unity.
        fcn_args : tuple, optional
            Extra parameters for supplying to the abeles function.
        fcn_kws : dict, optional
            Extra keyword parameters for supplying to the abeles function.
            See the notes below.
        kws : dict, optional
            Keywords passed to the minimizer.

        Notes
        -----
        ReflectivityFitter uses one extra kwds entry:

        fcn_kws['transform'] : callable, optional
            If specified then this function is used to transform the data
            returned by the model method. With the signature:
            ``transformed_y_vals = transform(x_vals, y_vals)``.

        fcn_kws['dqvals'] : np.ndarray, optional
            An array containing the _FWHM of the Gaussian approximated resolution
            kernel. Has the same size as qvals.

        fcn_kws['quad_order'] : int, optional
            The order of the Gaussian quadrature polynomial for doing the
            resolution smearing. default = 17. Don't choose less than 13. If
            quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
            quadrature will always work, but takes a _long_ time (2 or 3 orders
            of magnitude longer). Fixed quadrature will always take a lot less
            time. BUT it won't necessarily work across all samples. For
            example 13 points may be fine for a thin layer, but will be
            atrocious at describing a multilayer with Bragg peaks.
        '''
        if fcn_kws is None:
            fcn_kws = {}
        if kws is None:
            minimizer_kwds = {}

        super(ReflectivityFitter, self).__init__(None,
                                                 xdata,
                                                 ydata,
                                                 parameters,
                                                 edata=edata,
                                                 fcn_args=fcn_args,
                                                 callback=self.callback,
                                                 fcn_kws=fcn_kws,
                                                 kws=minimizer_kwds)

        self.transform = None
        if 'transform' in fcn_kws:
            self.transform = fcn_kws['transform']

    def model(self, parameters):
        '''
        Calculate the theoretical model, given a set of parameters.

        Parameters
        ----------
        parameters : lmfit.parameters.Parameters instance
            Contains the parameters that are required for reflectivity
            calculation.

        Returns
        -------
        yvals : np.ndarray
            The theoretical model for the xdata, i.e.
            abeles(self.xdata, parameters, *self.args, **self.kwds)
        '''
        params = np.array([param.value for param in parameters.values()], float)
        yvals = abeles(self.xdata, params, *self.userargs, **self.userkws)

        if self.transform:
            yvals, temp = self.transform(self.xdata, yvals)

        return yvals

    def set_dq(self, res, quad_order=17):
        '''
            Sets the resolution information.

        Parameters
        ----------
        res: None, float or np.ndarray
            If `None` then there is no resolution smearing.
            If a float, e.g. 5, then dq/q smearing of 5% is applied. If res==0
            then resolution smearing is removed.
            If an np.ndarray the same length as ydata, it contains the _FWHM of
            the Gaussian approximated resolution kernel.
        quad_order: int or 'ultimate'
            The order of the Gaussian quadrature polynomial for doing the
            resolution smearing. default = 17. Don't choose less than 13. If
            quad_order == 'ultimate' then adaptive quadrature is used. Adaptive
            quadrature will always work, but takes a _long_ time (2 or 3 orders
            of magnitude longer). Fixed quadrature will always take a lot less
            time. BUT it won't necessarily work across all samples. For
            example 13 points may be fine for a thin layer, but will be
            atrocious at describing a multilayer with Bragg peaks.
        '''
        if res is None and 'dqvals' in self.userkws:
            self.userkws.pop('dqvals')
        elif type(res) is float or type(res) is int:
            if res < 0.4 and 'dqvals' in self.userkws:
                self.userkws.pop('dqvals')
            else:
                self.userkws['dqvals'] = float(res)
        elif type(res) is np.ndarray and res.shape == self.ydata.shape:
            self.userkws['dqvals'] = res

        if quad_order > 12:
            self.userkws['quad_order'] = quad_order

    def sld_profile(self, parameters, fcn_args=(), **fcn_kws):
        '''
        Calculate the SLD profile corresponding to the model parameters.

        Parameters
        ----------
        parameters : lmfit.parameters.Parameters instance

        Returns
        -------
        (z, rho_z) : tuple of np.ndarrays
            The distance from the top interface and the SLD at that point.
        '''

        params = np.asfarray(parameters.valuesdict().values())

        if 'points' in fcn_kws and fcn_kws['points'] is not None:
            points = fcn_kws['points']
            return points, sld_profile(params, points)

        if not int(params[0]):
            zstart = -5 - 4 * np.fabs(params[7])
        else:
            zstart = -5 - 4 * np.fabs(params[11])

        temp = 0
        if not int(params[0]):
            zend = 5 + 4 * np.fabs(params[7])
        else:
            for ii in xrange(int(params[0])):
                temp += np.fabs(params[4 * ii + 8])
            zend = 5 + temp + 4 * np.fabs(params[7])

        points = np.linspace(zstart, zend, num=500)

        return points, sld_profile(params, points)

    def callback(self, parameters, iteration, resid, *fcn_args, **fcn_kws):
        return True


class Transform(object):

    def __init__(self, form):
        types = ['None', 'lin', 'logY', 'YX4', 'YX2']
        self.form = None

        if form in types:
            self.form = form

    def transform(self, xdata, ydata, edata=None):
        '''
            An irreversible transform from lin R vs Q, to some other form
            form - specifies the transform
                form = None - no transform is made.
                form = 'logY' - log transform
                form = 'YX4' - YX**4 transform
                form = 'YX2' - YX**2 transform
        '''

        if edata is None:
            etemp = np.ones_like(ydata)
        else:
            etemp = edata

        if self.form == 'None':
            yt = np.copy(ydata)
            et = np.copy(etemp)
        elif self.form == 'lin':
            yt = np.copy(ydata)
            et = np.copy(etemp)
        elif self.form == 'logY':
            yt, et = EP.EPlog10(ydata, etemp)
        elif self.form == 'YX4':
            yt = ydata * np.power(xdata, 4)
            et = etemp * np.power(xdata, 4)
        elif self.form == 'YX2':
            yt = ydata * np.power(xdata, 2)
            et = etemp * np.power(xdata, 2)
        if edata is None:
            return yt, None
        else:
            return yt, et


if __name__ == '__main__':
    import timeit
    a = np.zeros((12))
    a[0] = 1.
    a[1] = 1.
    a[4] = 2.07
    a[7] = 3
    a[8] = 100
    a[9] = 3.47
    a[11] = 2

    b = np.arange(10000.)
    b /= 20000.
    b += 0.0005

    def _loop():
        abeles(b, a)

    t = timeit.Timer(stmt=_loop)
    print(t.timeit(number=1000))
