import os.path
import os

import numpy as np

from numpy.testing import (assert_almost_equal, assert_equal)

from refnx.reflect import MDSimulation


class TestSimulation(object):
    def test_init(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        lgtfile = os.path.join(pth, 'mdsim_test.lgt')
        sim = MDSimulation(pdbfile, lgtfile=lgtfile, flip=True)
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [16, 5])
        assert_almost_equal(sim.layers[:, :, 0], np.ones((6, 16)))
        assert_almost_equal(sim.layers[:, :, 3], np.zeros((6, 16)))
        assert_almost_equal(sim.layers[:, :, 4], np.zeros((6, 16)))
        assert_equal(sim.flip, True)
        assert_almost_equal(sim.cut_off, 5)

    def test_read_pdb(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'mdsim_test.pdb')
        lgtfile = os.path.join(pth, 'mdsim_test.lgt')
        sim = MDSimulation(pdbfile, lgtfile=lgtfile, flip=True)
        a = np.arange(0, 10, 2)
        assert_equal(len(sim.structure), 6)
        for models in sim.structure:
            for k, atom in enumerate(models.get_atoms()):
                assert_almost_equal(atom.coord[0], a[k])
                assert_almost_equal(atom.coord[1], a[k])
                assert_almost_equal(atom.coord[2], a[k])
            assert_equal(k, 4)

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
        assert_equal(sim.layers.shape, [6, 16, 5])
        assert_equal(sim.av_layers.shape, [16, 5])
        sim.run()
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
        assert_equal(sim.layers.shape, [6, 10, 5])
        assert_equal(sim.av_layers.shape, [10, 5])
