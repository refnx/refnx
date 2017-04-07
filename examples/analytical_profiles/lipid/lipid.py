"""
Various Analytic profiles for studying lipid membranes at an interface
"""

import numpy as np
from refnx.analysis import AnalyticalReflectivityFunction
from numpy.testing import assert_


class bilayer_au(AnalyticalReflectivityFunction):
    """
    Order of layers:

    superphase (Si?)
    native SiO2
    sticker layer (could be permalloy, could be Cr)
    Au layer
    inner heads
    chain region, assumed to be symmetric
    outer heads
    subphase

    This model assumes that the Area per Molecule is identical in the inner
    and outer leaflets. This model can easily be changed to remove that
    assumption.
    """

    def __init__(self, b_heads, vm_heads, b_tails, vm_tails, *args, **kwds):
        """
        Parameters
        ----------
        b_heads: float
            Sum of coherent scattering lengths of head group (Angstrom)
        vm_heads: float
            Molecular volume of head group (Angstrom**2)
        b_tails: float
            Sum of coherent scattering lengths of tail group (Angstrom)
        vm_tails: float
            Molecular volume of tail group (Angstrom**2)
        """
        super(bilayer_magnetic_au, self).__init__(*args, **kwds)
        self.b_heads = b_heads
        self.vm_heads = vm_heads
        self.b_tails = b_tails
        self.vm_tails = vm_tails

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

        slab_model = np.zeros((32,), float)
        slab_model[0] = 6
        slab_model[1] = lmfit_values['scale']
        slab_model[2] = lmfit_values['SLD_super']
        slab_model[4] = lmfit_values['SLD_sub']
        slab_model[6] = lmfit_values['bkg']
        slab_model[7] = lmfit_values['roughness_sub_head']

        # SiO2 layer
        slab_model[8] = lmfit_values['thickness_siO2']
        slab_model[9] = lmfit_values['SLD_sio2']
        slab_model[11] = lmfit_values['roughness_sio2_si']

        # sticker layer
        slab_model[12] = lmfit_values['thickness_sticker']
        slab_model[13] = lmfit_values['SLD_sticker']
        slab_model[15] = lmfit_values['roughness_sticker_siO2']

        # Au layer
        slab_model[16] = lmfit_values['thickness_au']
        slab_model[17] = lmfit_values['SLD_au']
        slab_model[19] = lmfit_values['roughness_au_sticker']

        def overall_SLD(vf1, SLD1, SLD2):
            return vf1 * SLD1 + (1 - vf1) * SLD2

        area_per_molecule = lmfit_values['area_per_molecule']

        # inner heads
        volfrac = self.vm_heads / (area_per_molecule
                                   * lmfit_values['thickness_inner_head'])
        assert_(volfrac < 1)

        slab_model[20] = lmfit_values['thickness_inner_head']
        slab_model[21] = overall_SLD(volfrac,
                                     self.b_heads / self.vm_heads,
                                     lmfit_values['SLD_sub'])
        slab_model[23] = lmfit_values['roughness_head_au']

        # tail_region
        # this region encompasses inner and outer tail region.
        volfrac = 2 * self.vm_tails / (area_per_molecule
                                       * lmfit_values['thickness_tail'])
        assert_(volfrac < 1)

        slab_model[24] = lmfit_values['thickness_tail']
        slab_model[25] = overall_SLD(volfrac,
                                     self.b_tails / self.vm_tails,
                                     lmfit_values['SLD_sub'])
        slab_model[27] = lmfit_values['roughness_tail_head']

        # outer_heads
        volfrac = self.vm_heads / (area_per_molecule
                                   * lmfit_values['thickness_outer_head'])
        assert_(volfrac < 1)
        slab_model[28] = lmfit_values['thickness_outer_head']
        slab_model[29] = overall_SLD(volfrac,
                                     self.b_heads / self.vm_heads,
                                     lmfit_values['SLD_sub'])
        slab_model[31] = lmfit_values['roughness_head_tail']

        return slab_model


    def parameter_names(self, nparams=None):
        return ['scale', 'bkg',
                'SLD_super',
                'thickness_sio2, SLD_sio2, roughness_sio2_si',
                'thickness_sticker', 'SLD_sticker', 'roughness_sticker_siO2',
                'thickness_au', 'SLD_au', 'roughness_au_sticker',
                'area_per_molecule'
                'thickness_inner_head', 'roughness_head_au',
                'thickness_tail', 'roughness_tail_head',
                'thickness_outer_head', 'roughness_head_tail',
                'SLD_sub', 'roughness_sub_head']