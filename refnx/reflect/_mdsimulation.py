from __future__ import print_function, division
import numpy as np
try:
    from Bio.PDB import PDBParser
except ImportError:
    print('biopython must be installed to use MDSimulation')
    pass
try:
    import periodictable as pt
except ImportError:
    print('periodictable must be installed to use automatically generate '
          'scattering lengths, either install periodic table or use a .lgt '
          'file.')
    pass
from refnx.analysis import Parameters
from refnx.reflect import structure, Component


class MDSimulation(Component):
    """
    Determines the scattering length density profile along the z-axis of a
    molecular dynamics output file (PDB format, unless MDAnalysis is
    installed). Lateral averaging occurs in the x-y plane.

    Parameters
    ----------
    pdbfile: str
        The path and name of the .pdb file or for which the reflectometry
        should be found.
    radiation: str
        Either 'neutron' or 'xray'. This is the type of radiation that was used
        in the experimental reflectometry measurements.
    xray_energy: float, optional
        Only required if the radiation is 'xray'. This is the energy of the
        probing X-ray radiation, in units of kilo electron Volts
    lgtfile: str, optional
        The path and name of the .lgt file (if present), which contains the
        scattering lengths of each of the atom type in the pdbfile. Currently
        the .lgt file style that is supported is a 3 column space separated txt
        file where the columns are atom_type, real_scattering_length, and
        imaginary_scattering_length respectively. If a lgtfile is not used the
        scattering lengths for the system will be determined based on the
        element type that is defined in the final column of the pdb file.
    layer_thickness: float, optional
        The thickness of the layers that the simulation cell should be sliced
        into. This will depend on the size of the particles in the simulation,
        for example a coarse-grained forcefield simulation will require thicker
        layers to stop overbinning.
    cut_off: float, optional
        The thickness (in the z-dimension) of the volume of the bottom
        simulation cell that should be ignored. This is to allow for the use of
        a vacuum gap in the simulation cell. This thickness can be determined
        by visualising the simulation or by running the box with a cut-off of 0
        and investigating the SLD profile that is produced.
    flip: bool, optional
        False if the system should be read as is, True if the simulation cell
        should be rotated through the xy-plane. This expects the neutrons or
        X-ray to be incident at the bottom of the simulation cell (where
        z=0).
    roughness: float, optional
        The fractional (of the layer thickness) roughness to be considered
        between the layers in the simulation cell.
    verbose: bool, optional
        True if you want to be notified when the pdb and lgt files have been
        read and when the sld profile is calculated.
    structure: biopython.structure, attribute
        The is the object which contains the information read in from the
        simulation trajectory.
    dimensions: float, array_like, attribute
        An array of 3 floats containing the simulation cell dimensions.
    """
    def __init__(self, pdbfile, layer_thickness=1, cut_off=5, flip=False,
                 roughness=0, verbose=False):
        self.pdbfile = pdbfile
        self.structure = None
        self.dimensions = [0, 0, 0]
        self.read_pdb()
        self.verbose = verbose
        if self.verbose:
            print('PDB file read.')
        self.av_layers = np.zeros((int(self.dimensions[2] /
                                       layer_thickness) + 1, 5))
        self.av_layers[:, 0] = layer_thickness
        self.av_layers[:, 3] = layer_thickness * roughness
        self.layers = np.array([self.av_layers, ] * len(self.structure))
        self.flip = flip
        self.cut_off = cut_off

    def assign_scattering_lengths(self, lgtfile=None, radiation='neutron',
                                  xray_energy=None):
        self.lgtfile = lgtfile
        if radiation == 'neutron':
            self.neutron = True
        else:
            if not xray_energy:
                raise ValueError('If the probing radiation is the X-ray'
                                 ' it is necessary to define an xray_energy'
                                 ' (in keV).')
            else:
                self.neutron = False
                self.xray_energy = xray_energy
        self.scatlens = {}
        self.read_lgt()
        if self.verbose:
            print('Scattering lengths found.')

    def run(self):
        self._get_sld_profile()
        if self.verbose:
            print('SLD profile determined.')

    def read_pdb(self):
        """Parse pdb file.

        Parses the pdbfile.
        """
        parser = PDBParser()
        self.structure = parser.get_structure('model', self.pdbfile)
        file = open(self.pdbfile, 'r')
        for line in file:
            if line.startswith('CRYST1') and not np.any(self.dimensions):
                line_list = line.split()
                self.dimensions[0] = float(line_list[1])
                self.dimensions[1] = float(line_list[2])
                self.dimensions[2] = float(line_list[3])
                break

    def read_lgt(self):
        """Parses .lgt.

        Parses the lgtfile.
        """
        if self.lgtfile:
            file = open(self.lgtfile, 'r')
            for i, line in enumerate(file):
                line_list = line.split()
                self.scatlens[line_list[0]] = [float(line_list[1]),
                                               float(line_list[2])]
            file.close()
        else:
            import scipy.constants as const
            cre = const.physical_constants['classical electron radius'][0]
            for atom in self.structure.get_atoms():
                if atom.name not in self.scatlens:
                    scattering_length = [0, 0]
                    if self.neutron:
                        scattering_length[0] = pt.elements.symbol(
                            atom.element).neutron.b_c
                        if pt.elements.symbol(atom.element).neutron.b_c_i:
                            inc = pt.elements.symbol(
                                atom.element).neutron.b_c_i
                        else:
                            inc = 0
                        scattering_length[1] = inc
                    else:
                        scattering_length = np.mulitply(
                            pt.elements.symbol(
                                atom.element).xray.scattering_factors(
                                    energy=12), cre)
                    self.scatlens[atom.name] = scattering_length

    def set_atom_scattering(self, name, scattering_length):
        """
        Sets the scattering length of a particular atom.

        Parameters
        ----------
        name: str
            The atom type to be set.
        scattering_length: float, array_like
            The scattering length (real and imaginary) of the atom, in units of
            10^{-6} Angstrom.
        """
        if len(scattering_length) != 2:
            raise ValueError('The scattering length must be an array of '
                             'length 2, corresponding to the real and '
                             'imaginary scattering lengths.')
        self.scatlens[name] = scattering_length

    def set_residue_scattering(self, name, scattering_length):
        """
        Sets the scattering length of a particular residue.

        Parameters
        ----------
        name: str
            The residue to be set.
        scattering_length: float, array_like
            An array of shape [N, 2], where N is the number of atoms in the
            residue. The scattering length (real and imaginary) for each of the
            atoms in the residue, in units of 10^{-6} Angstrom.
        """
        for residue in self.structure.get_residues():
            if residue.resname == name:
                if len(residue) != len(scattering_length):
                    raise ValueError('The scattering length must be an array '
                                     'with shape [N, 2], where N is the '
                                     'number of atoms in the residue.')
                for k, atom in enumerate(residue.get_atoms()):
                    self.scatlens[atom.name] = scattering_length[k]

    def _get_sld_profile(self):
        """Calculate SLD profile.

        This will calculate the time-averaged SLD profile for the simulation
        trajectory. This is achieved by summing the scattering lengths for each
        of the atoms found in a given layer (of defined thickness). The total
        scattering length is converted to a density by division by the volume
        of the layer. This is completed for each timestep and the average
        taken.
        """
        structure = self.structure
        # loop through each timestep in the simulation trajectory
        for k, models in enumerate(structure):
            # loop across all atoms in the current timestep
            for atom in models.get_atoms():
                # assign scattering length based on atom type, if there is a
                # lgtfile use this, if not use periodictable
                scattering_length = self.scatlens[atom.name]
                # with the system split into a series of layer, select the
                # appropriate layer based on the atom's z coordinate
                layer_choose = int(atom.coord[2] /
                                   self.layers[k, 0, 0])
                # add the real and imaginary scattering lengths to this layer
                self.layers[k, layer_choose, 1] += scattering_length[0]
                self.layers[k, layer_choose, 2] += scattering_length[1]
        # get a scattering length density
        self.layers[:, :, 1] /= (self.dimensions[0] * self.dimensions[1] *
                                 self.layers[0, 0, 0])
        self.layers[:, :, 2] /= (self.dimensions[0] * self.dimensions[1] *
                                 self.layers[0, 0, 0])
        if self.flip:
            self.layers = self.layers[:, ::-1, :]
        # cut off layers as requested from the end/bottom of the cell
        # the + 1 is to account for the + 1 on line 91
        layers_to_cut = int(self.cut_off / self.layers[0, 0, 0]) + 1
        self.layers = self.layers[:, :-layers_to_cut, :]
        # get the time-averaged scattering length density profile
        self.av_layers = np.average(self.layers, axis=0)

    def sld_profile(self, z=None):
        """
        Calculates an SLD profile, as a function of distance through the
        interface.

        Parameters
        ----------
        z : float
            Interfacial distance (Angstrom) measured from interface between the
            fronting medium and the first layer.

        Returns
        -------
        sld : float
            Scattering length density / 1e-6 Angstrom**-2

        Notes
        -----
        This can be called in vectorised fashion.
        """
        slabs = self.slabs

        return structure.sld_profile(slabs, z)

    @property
    def slabs(self):
        return self.av_layers

    @property
    def parameters(self):
        p = Parameters(name='traj: {}, lgt: {}'.format(self.pdbfile,
                                                       self.lgtfile))
        return p
