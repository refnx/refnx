import os.path
import os

import numpy as np

from numpy.testing import (assert_almost_equal, assert_equal)

from refnx.reflect import MDSimulation, AtomClass


class TestSimulation(object):
    def test_init(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        lgtfile = os.path.join(pth, 'mdsim_test.lgt')
        sim = MDSimulation(pdbfile, lgtfile=lgtfile, flip=True)
        assert_equal(sim.layers.shape, [6, 10, 5])
        assert_equal(sim.av_layers.shape, [10, 5])
        assert_almost_equal(sim.layers[:, :, 0], np.ones((6, 10)))
        assert_almost_equal(sim.layers[:, :, 3], np.zeros((6, 10)))
        assert_almost_equal(sim.layers[:, :, 4], np.zeros((6, 10)))
        assert_equal(sim.flip, True)
        assert_almost_equal(sim.cut_off, 5)

    def test_read_pdb(self):
        try:
            import MDAnalysis as mda
        except ImportError:
            pass
        else:
            pth = os.path.dirname(os.path.abspath(__file__))
            pdbfile = os.path.join(pth, 'mdsim_test.pdb')
            lgtfile = os.path.join(pth, 'mdsim_test.lgt')
            sim = MDSimulation(pdbfile, lgtfile=lgtfile, flip=True)
            a = np.arange(0, 10, 2)
            assert_equal(len(sim.u.trajectory), 6)
            for ts in sim.u.trajectory:
                if isinstance(ts[0], AtomClass):
                    atoms = ts
                elif isinstance(sim.u, mda.Universe):
                    atoms = sim.u.atoms
                assert_equal(len(atoms), 5)
                for atom in range(0, len(atoms)):
                    assert_almost_equal(atoms[atom].position[0], a[atom])
                    assert_almost_equal(atoms[atom].position[1], a[atom])
                    assert_almost_equal(atoms[atom].position[2], a[atom])

    def test_read_mda(self):
        try:
            import MDAnalysis as mda
        except ImportError:
            pass
        else:
            pth = os.path.dirname(os.path.abspath(__file__))
            pdbfile = os.path.join(pth, 'mdsim_test.pdb')
            lgtfile = os.path.join(pth, 'mdsim_test.lgt')
            sim = MDSimulation(pdbfile, lgtfile=lgtfile, flip=True)
            a = np.arange(0, 10, 2)
            b = ['C1', 'C2', 'C3', 'N4', 'C5']
            assert_equal(len(sim.u.trajectory), 6)
            assert_equal(sim.u.atoms.names, b)
            for ts in sim.u.trajectory:
                assert_equal(len(ts), 5)
                assert_almost_equal(ts.positions[:, 0], a)
                assert_almost_equal(ts.positions[:, 1], a)
                assert_almost_equal(ts.positions[:, 2], a)

    def test_read_lgt(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        lgtfile = os.path.join(pth, 'mdsim_test.lgt')
        sim = MDSimulation(pdbfile, lgtfile=lgtfile, flip=True)
        a = [26.659, 26.659, 26.659, 9.36, 19.998]
        b = np.zeros((5))
        c = ['C1', 'C2', 'C3', 'N4', 'C5']
        for i in range(0, len(c)):
            assert_almost_equal(sim.scatlens[c[i]], [a[i], b[i]])

    def test_get_sld_profile(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        lgtfile = os.path.join(pth, 'mdsim_test.lgt')
        sim = MDSimulation(pdbfile, lgtfile=lgtfile, flip=False)
        a = np.ones(10)
        b = np.array([26.659, 0, 26.659, 0, 26.659, 0, 9.36, 0, 19.998, 0])
        b /= 100
        c = np.zeros(10)
        for i in range(0, sim.layers.shape[2]):
            assert_almost_equal(sim.layers[i, :, 0], a)
            assert_almost_equal(sim.layers[i, :, 1], b)
            assert_almost_equal(sim.layers[i, :, 2], c)
            assert_almost_equal(sim.layers[i, :, 3], c)
            assert_almost_equal(sim.layers[i, :, 4], c)
        assert_almost_equal(sim.av_layers[:, 0], a)
        assert_almost_equal(sim.av_layers[:, 1], b)
        assert_almost_equal(sim.av_layers[:, 2], c)
        assert_almost_equal(sim.av_layers[:, 3], c)
        assert_almost_equal(sim.av_layers[:, 4], c)
