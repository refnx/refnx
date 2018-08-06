import MDAnalysis as mda
import numpy as np
from refnx.analysis import Parameters
from refnx.reflect import structure


class Simulation:
    """
    Determines the scattering length density profile from a simulation output
    file.

    Parameters
    ----------
    pdbfile: str
        The path and name of the .pdb file
    lgtfile: str
        The path and name of the .lgt file (which contains the scattering
        lengths of each of the atom type in the pdbfile). Currently the .lgt
        file style that is supported is a 3 column space separated txt file
        where the columns are atom_type, real_scattering_length, and
        imaginary_scattering_length respectively.
    xray: bool, optional
          True if the scattering length of the particles should be scaled by
          the classical radius of an electron.
    layer_thickness: float
        The thickness of the layers that the simulation cell should be sliced
        into.
    cut_off: float
        The size of the simulation cell that should be ignored from the bottom.
        This is to allow for the use of a vacuum gap at the bottom of the cell.
    flip: bool, optional
        False if the system should be read as is, true is the simulation cell
        should be rotated through they-plane -- note that this treats the first
        side that the neutron or X-ray interacts with as that at z=0.
    roughness: float, optional
        The fractional (of the layer thickness) roughness to be considered
        between the layers.
    verbose: bool, optional
        True if you want to be notified when the pdb and lgt files have been
        read and when the sld profile is calculated.
    """
    def __init__(self, pdbfile=None, lgtfile=None, xray=False,
                 layer_thickness=1, cut_off=5, flip=False, roughness=0,
                 verbose=False):
        self.pdbfile = pdbfile
        self.lgtfile = lgtfile
        self.u = None
        self.xray = xray
        self.scatlens = {}
        self.read_pdb()
        if verbose:
            print('PDB file read.')
        self.read_lgt()
        if verbose:
            print('LGT file read.')
        self.av_layers = np.zeros((int(self.u.dimensions[2] /
                                       layer_thickness) -
                                   cut_off, 5))
        self.layers = np.zeros((self.av_layers.shape[0],
                                self.av_layers.shape[1],
                                len(self.u.trajectory)))
        self.layers[:, 0, :] = layer_thickness
        self.layers[:, 3, :] = layer_thickness * roughness
        self.layers[:, 4, :] = 0
        self.av_layers[:, 0] = layer_thickness
        self.av_layers[:, 3] = layer_thickness * roughness
        self.av_layers[:, 4] = 0
        self.flip = flip
        self.cut_off = cut_off
        self.get_sld_profile()
        if verbose:
            print('SLD profile determined.')

    def read_pdb(self):
        """Parse pdb file.

        This reads the pdb file into memory.
        """
        self.u = mda.Universe(self.pdbfile)

    def read_lgt(self):
        """Parses .lgt.

        Parses the lgtfile. If no lgtfile is defined falass will help the user
        to build one by working through the
        atom types in the pdb file and requesting input of the real and
        imaginary scattering lengths. This will also
        occur if a atom type if found in the pdbfile but not in the given lgts
        file. falass will write the lgtfile
        to disk if atom types do not feature in the given lgtfile or one is
        written from scratch.
        """
        if self.lgtfile:
            file = open(self.lgtfile, 'r')
            for i, line in enumerate(file):
                line_list = line.split()
                i = 1
                if self.xray:
                    i *= 2.817940
                self.scatlens[line_list[0]] = [float(line_list[1]) * i,
                                               float(line_list[2]) * i]
            file.close()
        else:
            raise ValueError("No lgtfile has been defined.")

    def get_sld_profile(self):
        """Calculate SLD profile.

        This will calculate the SLD profile for each of the timesteps. This is
        achieved by summing the scattering lengths for each of the atoms found
        in a given layer (of defined thickness). The total scattering length is
        converted to a density by division by the volume of the layer.
        """
        u = self.u
        k = 0
        for ts in u.trajectory:
            for atom in range(0, len(u.atoms)):
                analysis_layers = self.layers.shape[0] * self.layers[0, 0, k]
                if u.atoms[atom].position[2] < analysis_layers:
                    bin_choose = int(u.atoms[atom].position[2] /
                                     self.layers[0, 0, k])
                    self.layers[bin_choose, 1, k] += \
                        self.scatlens[u.atoms[atom].name][0]
                    self.layers[bin_choose, 2, k] += \
                        self.scatlens[u.atoms[atom].name][1]
            self.layers[:, 1, k] /= (u.dimensions[0] * u.dimensions[1] *
                                     self.layers[0, 0, k])
            self.layers[:, 2, k] /= (u.dimensions[0] * u.dimensions[1] *
                                     self.layers[0, 0, k])
            k += 1
        if self.flip:
            self.layers = self.layers[::-1, :, :]
        self.av_layers = np.average(self.layers, axis=2)

    def sld_profile(self, z=None):
        """
        Calculates an SLD profile, as a function of distance through the
        interface.s

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
