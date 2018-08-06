import os.path
import os

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal)

from refnx.reflect import Simulation


class TestSimulation(object):
    def test_init(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'sim_test.pdb')
        lgtfile = os.path.join(pth, 'sim_test.lgt')
        sim = Simulation(pdbfile, lgtfile=lgtfile, flip=True)
        assert_equal(sim.layers.shape, [10, 5, 6])
        assert_equal(sim.av_layers.shape, [10, 5])
        assert_almost_equal(sim.layers[:, 0, :], np.ones((10, 6)))
        assert_almost_equal(sim.layers[:, 3, :], np.zeros((10, 6)))
        assert_almost_equal(sim.layers[:, 4, :], np.zeros((10, 6)))
        assert_equal(sim.flip, True)
        assert_almost_equal(sim.cut_off, 5)
        assert_equal(sim.xray, False)

    def test_read_pdb(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'sim_test.pdb')
        lgtfile = os.path.join(pth, 'sim_test.lgt')
        sim = Simulation(pdbfile, lgtfile=lgtfile, flip=True)
        a = np.arange(0, 10, 2)
        b = ['C1', 'C2', 'C3', 'N4', 'C5']
        assert_equal(len(sim.u.trajectory), 6)
        assert_equal(sim.u.atoms.names, b)
        for ts in sim.u.trajectory:
            assert_equal(len(sim.u.atoms), 5)
            assert_almost_equal(sim.u.atoms.positions[:, 0], a)
            assert_almost_equal(sim.u.atoms.positions[:, 1], a)
            assert_almost_equal(sim.u.atoms.positions[:, 2], a)

    def test_read_lgt(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'sim_test.pdb')
        lgtfile = os.path.join(pth, 'sim_test.lgt')
        sim = Simulation(pdbfile, lgtfile=lgtfile, flip=True)
        a = [26.659, 26.659, 26.659, 9.36, 19.998]
        b = np.zeros((5))
        c = ['C1', 'C2', 'C3', 'N4', 'C5']
        for i in range(0, len(c)):
            assert_almost_equal(sim.scatlens[c[i]], [a[i], b[i]])

    def test_get_sld_profile(self):
        pth = os.path.dirname(os.path.abspath(__file__))
        pdbfile = os.path.join(pth, 'sim_test.pdb')
        lgtfile = os.path.join(pth, 'sim_test.lgt')
        sim = Simulation(pdbfile, lgtfile=lgtfile, flip=True)
        a = np.ones(10)
        b = np.array([0, 19.998, 0, 9.36, 0, 26.659, 0, 26.659, 0, 26.659])
        b /= 100
        c = np.zeros(10)
        for i in range(0, sim.layers.shape[2]):
            assert_almost_equal(sim.layers[:, 0, i], a)
            assert_almost_equal(sim.layers[:, 1, i], b)
            assert_almost_equal(sim.layers[:, 2, i], c)
            assert_almost_equal(sim.layers[:, 3, i], c)
            assert_almost_equal(sim.layers[:, 4, i], c)
        assert_almost_equal(sim.av_layers[:, 0], a)
        assert_almost_equal(sim.av_layers[:, 1], b)
        assert_almost_equal(sim.av_layers[:, 2], c)
        assert_almost_equal(sim.av_layers[:, 3], c)
        assert_almost_equal(sim.av_layers[:, 4], c)
