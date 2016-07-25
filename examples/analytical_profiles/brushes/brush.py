""""
Various Analytic profiles for studying polymer brushes at an interface
# Author - Tim Murdoch, University of Newcastle 2016
"""
import os.path
import numpy as np
from refnx.analysis import AnalyticalReflectivityFunction, Transform
from scipy.integrate import simps, trapz
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from lmfit import Parameter, Parameters
from scipy.interpolate import Akima1DInterpolator, InterpolatedUnivariateSpline


class Brush(AnalyticalReflectivityFunction):
    """
    Class contains parameters and to to_slab methods for fitting polymer brush
    reflectivity data using different functional forms

    Order of layers:

    superphase (Si)
    native SiO2
    interior polymer layers
    exterior polymer layers (functional form for tail)
    subphase

    """
    tail_par = []

    # Common parameters to all functional forms
    gen_par = ['scale', 'bkg',
               'SLD_super', 'SLD_sub', 'thickness_SiO2', 'SLD_SiO2',
               'roughness_SiO2', 'roughness_backing']

    def __init__(self, sld_poly, n_interior, *args, n_slices=50,
                 vol_cut=0.005, **kwds):
        """
        Parameters
        ----------
        sld_poly: float
            SLD of polymer
        n_interior: integer
            number of interior layers
        n_slices: integer
            number of layers to slice tail region into
        vol_cut: float
            volume fraction at which to cut off exponentially decaying profiles
        """
        super(Brush, self).__init__(*args, **kwds)
        self.sld_poly = sld_poly
        self.n_interior = n_interior
        self.n_slices = n_slices
        self.vol_cut = vol_cut

    def vol_fraction(self, params):
        """
        Calculates SLD profile and sets boundary between SiO2 and polymer to
        z = 0. Then calculates volume fraction profile from additive mixing of
        SLD
        Parameters
        ----------
        params: lmfit.Parameters instance
            The parameters for this analytic profile
        Returns
        -------
        z: z value for volume fraction profile
        profile: volume fraction (phi) values for volume fraction profile

        """
        # Store roughness between SiO2 and first polymer layer and temporarily
        # set to zero
        lmfit_values = params.valuesdict()
        roughness = lmfit_values['roughness_1']
        params['roughness_1'].val = 0

        # Determine extent of SLD profile and store
        z, profile = self.sld_profile(params)
        end = max(z)

        # generate points vector between SiO2 and last layer and re-evaluate
        # SLD profile
        points = np.linspace(lmfit_values['thickness_SiO2'], end, num=1001)
        z, profile = self.sld_profile(params, z=points)

        # Convert SLD profile to volume fraction and zero z values at extent of
        # SiO2 layer
        profile = (profile - lmfit_values['SLD_sub']) / \
                  (self.sld_poly - lmfit_values['SLD_sub'])
        z = z - lmfit_values['thickness_SiO2']
        # Restore roughness parameter to original value
        params['roughness_1'].val = roughness

        return z, profile

    def moment(self, params, moment=1):
        """
        Calculates the n'th moment of the volume fraction profile

        Parameters
        ----------
        params: lmfit.Parameters instance
        moment: order of moment to be calculated

        Returns
        -------
        n'th moment
        """
        points, profile = self.vol_fraction(params)
        profile *= points**moment
        val = simps(profile, points)
        area = self.vfp_area(params)
        return val / area

    def vfp_area(self, params):
        """
        Calculates integrated area of volume fraction profile
        Parameters
        ----------
        params: lmfit.Parameters instance

        Returns
        -------
        area: integrated area of volume fraction profile
        """

        # Evaluates volume fraction profile then calculates area using
        # Simpson's rule
        points, profile = self.vol_fraction(params)
        area = simps(profile, points)
        return area

    def parameter_names(self, nparams=None):
        """
        Generates list of parameter names for interior layers then returns
        concatenated list of general parameter, tail parameters and interior
        layer parameters.
        Parameters
        ----------
        nparams

        Returns
        -------
        parameter names
        """
        # TODO list the parameter names.
        int_par = ",".join(['thickness_%d,phi_%d,roughness_%d' %
                            (i + 1, i + 1, i + 1) for i in range(self.n_interior)]).split(',')

        return self.gen_par + self.tail_par + int_par


class BrushPara(Brush):
    """
    Class adds parabolic functional form for the tail (doi: 10.1002/macp.201300477)
    """
    # TODO list the exact parameter names in the class docstring
    # Parameters needed for parabolic tail
    tail_par = ['phi_init', 'thickness_tail', 'roughness_tail2int']

    def __init__(self, *args, **kwds):
        """
        Parameters
        ----------
        """
        super(BrushPara, self).__init__(*args, **kwds)

    def to_slab(self, params):
        """
        Parameters
        ----------
        params: lmfit.Parameters instance
            The parameters for this analytic profile
        Returns
        -------
        slab_model: np.ndarray
            Parameters for a slab-model reflectivity calculation
        """
        # with this model there are 6 layers, and the reflectivity calculation
        # needs to be done with 4*N + 8 = 32 variables
        lmfit_values = params.valuesdict()

        # Number of layers addition of interior Slabs, analytical Slices and SiO2
        n_layers = self.n_interior + self.n_slices + 1
        n_par = 4 * n_layers + 8

        # General model parameters
        slab_model = np.zeros((n_par,), float)
        slab_model[0] = n_layers
        slab_model[1] = lmfit_values['scale']
        slab_model[2] = lmfit_values['SLD_super']
        slab_model[4] = lmfit_values['SLD_sub']
        slab_model[6] = lmfit_values['bkg']
        slab_model[7] = lmfit_values['roughness_backing']

        # SiO2 layer
        slab_model[8] = lmfit_values['thickness_SiO2']
        slab_model[9] = lmfit_values['SLD_SiO2']
        slab_model[11] = lmfit_values['roughness_SiO2']

        # determine SLD for a given polymer volume fraction
        def overall_sld(vf1):
            return vf1 * self.sld_poly + (1 - vf1) * lmfit_values['SLD_sub']

        # Interior Layers
        for i in range(self.n_interior):
            slab_model[12 + 4*i] = lmfit_values['thickness_{}'.format(i + 1)]
            slab_model[13 + 4*i] = overall_sld(lmfit_values['phi_{}'.format(i + 1)])
            slab_model[15 + 4*i] = lmfit_values['roughness_{}'.format(i + 1)]

        # Parabola Layers
        # Determine maximum extent and divide by number of layers
        slab_thick = lmfit_values['thickness_tail'] / self.n_slices
        for i in range(self.n_slices):
            # Volume fraction (phi) evaluated at half-height of each layer
            distance = (i + 0.5) * slab_thick
            phi = (lmfit_values['phi_init']
                   * (1 - (distance / lmfit_values['thickness_tail'])**2))
            # Assign slab thickness and volume fraction
            slab_model[12 + 4 * (self.n_interior + i)] = slab_thick
            slab_model[13 + 4 * (self.n_interior + i)] = overall_sld(phi)

            # If first iteration of loop, set roughness between slab and tail,
            # else apply an arbitrary smoothing roughness
            if not i:
                slab_model[15 + 4 * (self.n_interior + i)] = \
                    lmfit_values['roughness_tail2int']
            else:
                slab_model[15 + 4 * (self.n_interior + i)] = slab_thick / 3

        return slab_model


def test_brush_para():
    # load in some previously calculated data (from IGOR) for a test
    path = os.path.dirname(os.path.abspath(__file__))
    igor_r, igor_q = np.hsplit(np.loadtxt(os.path.join(path, 'brush_para.txt')), 2)

    # Convert reflectivity to logY
    transform = Transform('logY').transform
    brush = BrushPara(0.46, 3, transform=transform, dq=0)

    # Generate parameter names
    names = brush.parameter_names()

    # these are the parameters we used in our IGOR analysis
    igor_params = {'scale': 1, 'bkg': 1e-7, 'SLD_super': 2.07,
                   'SLD_sub': 6.36, 'thickness_SiO2': 8.8,
                   'SLD_SiO2': 3.47, 'roughness_SiO2': 3.5,
                   'roughness_backing': 10, 'phi_init': 0.1,
                   'thickness_tail': 1000, 'roughness_tail2int': 4,
                   'thickness_1': 28, 'phi_1': 0.95, 'roughness_1': 2,
                   'thickness_2': 50, 'phi_2': 0.85, 'roughness_2': 2,
                   'thickness_3': 100, 'phi_3': 0.2, 'roughness_3': 20}

    # Create parameter dictionary add names and values
    P = Parameters()
    for name, val in igor_params.items():
        P.add(name, val, True)

    # Confirm reflectivity same as IGOR
    ref = brush.model(igor_q, P)
    assert_almost_equal(ref, igor_r)

    # Confirm area calculated correctly
    area = brush.vfp_area(P)
    assert_allclose(area, 155.40491, atol=1e-2)

    # Confirm average brush height (2 * first_moment) is correct
    first_moment = brush.moment(P)
    assert_allclose(2 * first_moment, 542.9683913)


class BrushGauss(Brush):
    """
    Class adds Gaussian functional form for the tail (doi: 10.1002/macp.201300477)
    """
    # TODO list the exact parameter names in the class docstring

    # Tail parameter names required for Gaussian tail
    tail_par = ['phi_init', 'thickness_tail', 'roughness_tail2int']

    def __init__(self, *args, **kwds):
        """
        Parameters
        ----------
        """
        super(BrushGauss, self).__init__(*args, **kwds)

    def to_slab(self, params):
        """
        Parameters
        ----------
        params: lmfit.Parameters instance
            The parameters for this analytic profile
        Returns
        -------
        slab_model: np.ndarray
            Parameters for a slab-model reflectivity calculation
        """
        # with this model there are 6 layers, and the reflectivity calculation
        # needs to be done with 4*N + 8 = 32 variables
        lmfit_values = params.valuesdict()

        # Interior Slabs, Analytical Slices and SiO2
        n_layers = self.n_interior + self.n_slices + 1
        n_par = 4*n_layers + 8

        # General model parameters
        slab_model = np.zeros((n_par,), float)
        slab_model[0] = n_layers
        slab_model[1] = lmfit_values['scale']
        slab_model[2] = lmfit_values['SLD_super']
        slab_model[4] = lmfit_values['SLD_sub']
        slab_model[6] = lmfit_values['bkg']
        slab_model[7] = lmfit_values['roughness_backing']

        # SiO2 layer
        slab_model[8] = lmfit_values['thickness_SiO2']
        slab_model[9] = lmfit_values['SLD_SiO2']
        slab_model[11] = lmfit_values['roughness_SiO2']

        def overall_sld(vf1):
            return vf1 * self.sld_poly + (1 - vf1) * lmfit_values['SLD_sub']

        # Interior Layers
        for i in range(self.n_interior):
            slab_model[12 + 4*i] = lmfit_values['thickness_{}'.format(i + 1)]
            slab_model[13 + 4*i] = overall_sld(lmfit_values['phi_{}'.format(i + 1)])
            slab_model[15 + 4*i] = lmfit_values['roughness_{}'.format(i + 1)]

        # Gaussian Layers
        # Cutoff_thickness occurs at distance where volume fraction
        # equal to vol_cut
        cutoff_thickness = np.sqrt(-lmfit_values['thickness_tail']**2 *
                                   np.log(self.vol_cut / lmfit_values['phi_init']))
        # Layer thickness equal to cutoff distance divided by number of slices
        slab_thick = cutoff_thickness / self.n_slices
        for i in range(self.n_slices):
            # Calculates volume fraction (phi) at half-height distance of each
            # layer then assigns appropriate values
            distance = (i + 0.5) * slab_thick
            phi = lmfit_values['phi_init'] *  \
                  np.exp(-(distance / lmfit_values['thickness_tail'])**2)
            slab_model[12 + 4 * (self.n_interior + i)] = slab_thick
            slab_model[13 + 4 * (self.n_interior + i)] = overall_sld(phi)

            # If first iteration of loop, set roughness between slab and tail,
            # else apply an arbitrary smoothing roughness
            if not i:
                slab_model[15 + 4 * (self.n_interior + i)] = lmfit_values['roughness_tail2int']
            else:
                slab_model[15 + 4 * (self.n_interior + i)] = slab_thick / 3
        return slab_model


def test_brush_gauss():
    # load in some previously calculated data (from IGOR) for a test
    path = os.path.dirname(os.path.abspath(__file__))
    igor_r, igor_q = np.hsplit(np.loadtxt(os.path.join(path, 'brush_Gauss.txt')), 2)

    # Convert reflectivity to logY
    transform = Transform('logY').transform
    brush = BrushGauss(0.46, 3, transform=transform, dq=0)

    # Generate parameter names
    names = brush.parameter_names()

    # these are the parameters we used in our IGOR analysis
    igor_params = {'scale': 1, 'bkg': 1e-7, 'SLD_super': 2.07,
                   'SLD_sub': 6.36, 'thickness_SiO2': 8.8,
                   'SLD_SiO2': 3.47, 'roughness_SiO2': 3.5,
                   'roughness_backing': 10, 'phi_init': 0.1,
                   'thickness_tail': 1000, 'roughness_tail2int': 4,
                   'thickness_1': 28, 'phi_1': 0.95, 'roughness_1': 2,
                   'thickness_2': 50, 'phi_2': 0.85, 'roughness_2': 2,
                   'thickness_3': 100, 'phi_3': 0.2, 'roughness_3': 20}

    # Create parameter dictionary add names and values
    P = Parameters()
    for name, val in igor_params.items():
        P.add(name, val, True)

    # Confirm reflectivity same as IGOR
    ref = brush.model(igor_q, P)
    assert_almost_equal(ref, igor_r)

    # Confirm area calculated correctly
    area = brush.vfp_area(P)
    assert_allclose(area, 176.087080, atol=1e-2)

    # Confirm average brush height (2 * first_moment) is correct
    first_moment = brush.moment(P)
    assert_allclose(2 * first_moment, 776.621101187)


class BrushSlabs(Brush):
    """
    Simple brush model using only slab layers
    """
    # TODO list the exact parameter names in the class docstring

    tail_par = []

    def __init__(self, *args, **kwds):
        """
        Parameters
        ----------
        """
        super(BrushSlabs, self).__init__(*args, **kwds)

    def to_slab(self, params):
        """
        Parameters
        ----------
        params: lmfit.Parameters instance
            The parameters for this analytic profile
        Returns
        -------
        slab_model: np.ndarray
            Parameters for a slab-model reflectivity calculation
        """
        # with this model there are 6 layers, and the reflectivity calculation
        # needs to be done with 4*N + 8 = 32 variables
        lmfit_values = params.valuesdict()

        # Number of layers addition of interior Slabs and SiO2 layer
        n_layers = self.n_interior + 1
        n_par = 4 * n_layers + 8

        # General model parameters
        slab_model = np.zeros((n_par,), float)
        slab_model[0] = n_layers
        slab_model[1] = lmfit_values['scale']
        slab_model[2] = lmfit_values['SLD_super']
        slab_model[4] = lmfit_values['SLD_sub']
        slab_model[6] = lmfit_values['bkg']
        slab_model[7] = lmfit_values['roughness_backing']

        # SiO2 layer
        slab_model[8] = lmfit_values['thickness_SiO2']
        slab_model[9] = lmfit_values['SLD_SiO2']
        slab_model[11] = lmfit_values['roughness_SiO2']

        # determine SLD for a given polymer volume fraction
        def overall_sld(vf1):
            return vf1 * self.sld_poly + (1 - vf1) * lmfit_values['SLD_sub']

        # Interior Layers
        for i in range(self.n_interior):
            slab_model[12 + 4*i] = lmfit_values['thickness_{}'.format(i + 1)]
            slab_model[13 + 4*i] = overall_sld(lmfit_values['phi_{}'.format(i + 1)])
            slab_model[15 + 4*i] = lmfit_values['roughness_{}'.format(i + 1)]

        return slab_model


def test_brush_slab():
    # load in some previously calculated data (from IGOR) for a test
    path = os.path.dirname(os.path.abspath(__file__))
    igor_r, igor_q = np.hsplit(np.loadtxt(os.path.join(path, 'brush_Slab.txt')), 2)

    # Convert reflectivity to logY
    transform = Transform('logY').transform
    brush = BrushSlabs(0.46, 3, transform=transform, dq=0)

    # Generate parameter names
    names = brush.parameter_names()

    # these are the parameters we used in our IGOR analysis
    igor_params = {'scale': 1, 'bkg': 1e-7, 'SLD_super': 2.07,
                   'SLD_sub': 6.36, 'thickness_SiO2': 8.8,
                   'SLD_SiO2': 3.47, 'roughness_SiO2': 3.5,
                   'roughness_backing': 10,
                   'thickness_1': 28, 'phi_1': 0.95, 'roughness_1': 2,
                   'thickness_2': 50, 'phi_2': 0.85, 'roughness_2': 2,
                   'thickness_3': 100, 'phi_3': 0.2, 'roughness_3': 20}

    # Create parameter dictionary add names and values
    P = Parameters()
    for name, val in igor_params.items():
        P.add(name, val, True)

    # Confirm reflectivity same as IGOR
    ref = brush.model(igor_q, P)
    assert_almost_equal(ref, igor_r)

    # Confirm area calculated correctly
    area = brush.vfp_area(P)
    assert_allclose(area, 88.734573085, atol=1e-2)

    # Confirm average brush height (2 * first_moment) is correct
    first_moment = brush.moment(P)
    assert_allclose(2 * first_moment, 120.01276074)


class BrushSpline(Brush):
    """
    Interior slab --> Akima spline
    """
    # TODO list the exact parameter names in the class docstring

    # Tail parameter names required for Gaussian tail
    # tail_par = ['n_nodes', thickness_akima, 'vf_akima_0', ..., 'vf_akima_n-1']

    def __init__(self, n_nodes, *args, **kwds):
        """
        Parameters
        ----------
        """
        self.tail_par = ['n_nodes', 'thickness_akima']
        self.n_nodes = n_nodes
        for i in range(n_nodes):
            self.tail_par.append('vf_nodes_%d' % i)

        super(BrushSpline, self).__init__(*args, **kwds)

    def to_slab(self, params):
        """
        Parameters
        ----------
        params: lmfit.Parameters instance
            The parameters for this analytic profile
        Returns
        -------
        slab_model: np.ndarray
            Parameters for a slab-model reflectivity calculation
        """
        lmfit_values = params.valuesdict()

        # Interior Slabs, Analytical Slices and SiO2
        n_layers = self.n_interior + self.n_slices + 1
        n_par = 4 * n_layers + 8

        # General model parameters
        slab_model = np.zeros((n_par,), float)
        slab_model[0] = n_layers
        slab_model[1] = lmfit_values['scale']
        slab_model[2] = lmfit_values['SLD_super']
        slab_model[4] = lmfit_values['SLD_sub']
        slab_model[6] = lmfit_values['bkg']
        slab_model[7] = lmfit_values['roughness_backing']

        # SiO2 layer
        slab_model[8] = lmfit_values['thickness_SiO2']
        slab_model[9] = lmfit_values['SLD_SiO2']
        slab_model[11] = lmfit_values['roughness_SiO2']

        def overall_sld(vf1):
            return vf1 * self.sld_poly + (1 - vf1) * lmfit_values['SLD_sub']

        # Interior Layers
        for i in range(self.n_interior):
            slab_model[12 + 4*i] = lmfit_values['thickness_{}'.format(i + 1)]
            slab_model[13 + 4*i] = overall_sld(lmfit_values['phi_{}'.format(i + 1)])
            slab_model[15 + 4*i] = lmfit_values['roughness_{}'.format(i + 1)]

        # akima layer
        thickness_akima = lmfit_values['thickness_akima']
        vf_nodes = np.zeros(2 + self.n_nodes)
        vf_nodes[0] = lmfit_values['phi_{}'.format(self.n_interior)]
        for i in range(self.n_nodes):
            vf_nodes[1 + i] = lmfit_values['vf_nodes_%d' % i]

        A = InterpolatedUnivariateSpline(np.linspace(0, thickness_akima, 2 + self.n_nodes),
                                         vf_nodes,
                                         k=2)

        slab_thick = thickness_akima / self.n_slices
        phi = A(np.linspace(slab_thick / 2, thickness_akima - slab_thick/2, self.n_slices))

        for i in range(self.n_slices):
            slab_model[12 + 4 * (self.n_interior + i)] = slab_thick
            slab_model[13 + 4 * (self.n_interior + i)] = overall_sld(phi[i])
            slab_model[15 + 4 * (self.n_interior + i)] = slab_thick / 4.
        return slab_model


if __name__ == "__main__":
    test_brush_para()
    test_brush_gauss()
    test_brush_slab()
