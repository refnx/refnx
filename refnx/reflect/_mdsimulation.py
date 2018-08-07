from __future__ import print_function, division
import numpy as np
try:
    import MDAnalysis as mda
except ImportError:
    pass
try:
    import periodictable as pt
except ImportError:
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
    pdbfile: str or MDAnalysis.Universe
        The path and name of the .pdb file or the MDAnalysis.Universe object
        for which the reflectometry should be found.
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
    u: Universe or MDAnalysis.Universe
        The is the object which contains the information read in from the
        simulation trajectory.
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
    """
    def __init__(self, pdbfile, radiation='neutron', xray_energy=None,
                 lgtfile=None, layer_thickness=1, cut_off=5, flip=False,
                 roughness=0, verbose=False):
        self.pdbfile = pdbfile
        self.lgtfile = lgtfile
        self.u = None
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
        self.read_pdb()
        if verbose:
            print('PDB file read.')
        if lgtfile:
            self.read_lgt()
            if verbose:
                print('LGT file read.')
        self.av_layers = np.zeros((int(self.u.dimensions[2] /
                                       layer_thickness) + 1, 5))
        self.av_layers[:, 0] = layer_thickness
        self.av_layers[:, 3] = layer_thickness * roughness
        self.layers = np.array([self.av_layers, ] * len(self.u.trajectory))
        self.flip = flip
        self.cut_off = cut_off
        self._get_sld_profile()
        if verbose:
            print('SLD profile determined.')

    def read_pdb(self):
        """Parse pdb file.

        Parses the pdbfile.
        """
        try:
            import MDAnalysis as mda
        except ImportError:
            self.u = Universe(self.pdbfile)
        else:
            if isinstance(self.pdbfile, mda.Universe):
                self.u = self.pdbfile
            else:
                self.u = mda.Universe(self.pdbfile)

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
            raise ValueError("No lgtfile has been defined.")

    def _get_sld_profile(self):
        """Calculate SLD profile.

        This will calculate the time-averaged SLD profile for the simulation
        trajectory. This is achieved by summing the scattering lengths for each
        of the atoms found in a given layer (of defined thickness). The total
        scattering length is converted to a density by division by the volume
        of the layer. This is completed for each timestep and the average
        taken.
        """
        u = self.u
        # loop through each timestep in the simulation trajectory
        for k, ts in enumerate(u.trajectory):
            if isinstance(ts[0], AtomClass):
                atoms = ts
            elif isinstance(u, mda.Universe):
                atoms = u.atoms
            # loop across all atoms in the current timestep
            for atom in range(0, len(atoms)):
                # assign scattering length based on atom type, if there is a
                # lgtfile use this, if not use periodictable
                if self.scatlens:
                    scattering_length = self.scatlens[atoms[atom].name]
                else:
                    atom_type = self.u.atoms[atom].type
                    if self.neutron:
                        scattering_length = pt.elements.symbol(
                            atom_type).neutron.scattering()[0][0:2]
                    else:
                        scattering_length = pt.elements.symbol(
                            atom_type).xray.sld(
                                energy=self.xray_energy)
                # with the system split into a series of layer, select the
                # appropriate layer based on the atom's z coordinate
                layer_choose = int(atoms[atom].position[2] /
                                   self.layers[k, 0, 0])
                # add the real and imaginary scattering lengths to this layer
                self.layers[k, layer_choose, 1] += scattering_length[0]
                self.layers[k, layer_choose, 2] += scattering_length[1]
        # get a scattering length density
        self.layers[:, :, 1] /= (u.dimensions[0] * u.dimensions[1] *
                                 self.layers[0, 0, 0])
        self.layers[:, :, 2] /= (u.dimensions[0] * u.dimensions[1] *
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


class Universe:
    """
    This is a custom class to emulate the parsing of simulations by MDAnalysis
    such that the Simulation class will operate both with and without
    MDAnalysis.

    Parameters
    ----------
    pdbfile: str
        The path and name of the .pdb file
    trajectory: ndarray
        An array of arrays of type AtomClass
    dimensions: float, array_like
        An array of length 3 giving the cell dimensions for the simulation.
    """
    def __init__(self, pdbfile):
        self.pdbfile = pdbfile
        self.trajectory = np.array([])
        self.dimensions = np.zeros((3))
        self.read_pdb()

    def read_pdb(self):
        file = open(self.pdbfile, 'r')
        count_timesteps = 0
        atoms = np.array([], dtype=AtomClass)
        for line in file:
            if line[0:6] == 'CRYST1' and not np.any(self.dimensions):
                line_list = line.split()
                self.dimensions[0] = line_list[1]
                self.dimensions[1] = line_list[2]
                self.dimensions[2] = line_list[3]
            if line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
                atoms = np.append(atoms,
                                  AtomClass(line[12:16].strip(),
                                            [float(line[30:38]),
                                             float(line[38:46]),
                                             float(line[46:54])],
                                            line[76:78]))
            if line[0:6] == 'ENDMDL':
                self.trajectory = np.append(self.trajectory, atoms)
                atoms = np.array([], dtype=AtomClass)
                count_timesteps += 1
        self.trajectory = self.trajectory.reshape(count_timesteps,
                                                  int(self.trajectory.size /
                                                      count_timesteps))


class AtomClass():
    """
    This is a custom class to emulate the parsing of atoms by MDAnalysis such
    that the Simulation class will operate both with and without MDAnalysis.

    Parameters
    ----------
    position: float, array_like
        An array of length 3 containing information about the position of the
        particle.
    name: str
        The atom type for the given particle.
    """
    def __init__(self, name, position, type):
        self.position = position
        self.name = name
        self.type = type
