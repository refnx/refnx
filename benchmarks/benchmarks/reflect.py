import os.path
import numpy as np

from .common import Benchmark

from refnx.analysis import CurveFitter, Objective, Parameter
import refnx.reflect
from refnx.reflect._creflect import abeles as c_abeles
from refnx.reflect._reflect import abeles
from refnx.reflect import SLD, Slab, Structure, ReflectModel, reflectivity
from refnx.dataset import ReflectDataset as RD


class Abeles(Benchmark):
    def setup(self):
        self.q = np.linspace(0.005, 0.5, 50000)
        self.layers = np.array([[0, 2.07, 0, 3],
                                [50, 3.47, 0.0001, 4],
                                [200, -0.5, 1e-5, 5],
                                [50, 1, 0, 3],
                                [0, 6.36, 0, 3]])
        self.repeat=20
        self.number=10

    def time_cabeles(self):
        c_abeles(self.q, self.layers)

    def time_abeles(self):
        abeles(self.q, self.layers)

    def time_reflectivity(self):
        reflectivity(self.q, self.layers)


class Reflect(Benchmark):
    timeout = 120.
    repeat = 2

    def setup(self):
        pth = os.path.dirname(os.path.abspath(refnx.reflect.__file__))
        e361 = RD(os.path.join(pth, 'test', 'e361r.txt'))

        sio2 = SLD(3.47, name='SiO2')
        si = SLD(2.07, name='Si')
        d2o = SLD(6.36, name='D2O')
        polymer = SLD(1, name='polymer')

        # e361 is an older dataset, but well characterised
        structure361 = si | sio2(10, 4) | polymer(200, 3) | d2o(0, 3)
        model361 = ReflectModel(structure361, bkg=2e-5)

        model361.scale.vary = True
        model361.bkg.vary = True
        model361.scale.range(0.1, 2)
        model361.bkg.range(0, 5e-5)
        model361.dq = 5.

        # d2o
        structure361[-1].sld.real.vary = True
        structure361[-1].sld.real.range(6, 6.36)

        structure361[1].thick.vary = True
        structure361[1].thick.range(5, 20)
        structure361[2].thick.vary = True
        structure361[2].thick.range(100, 220)

        structure361[2].sld.real.vary = True
        structure361[2].sld.real.range(0.2, 1.5)

        # e361.x_err = None
        objective = Objective(model361,
                              e361)
        self.fitter = CurveFitter(objective, nwalkers=200)
        self.fitter.initialise('jitter')

    def time_reflect_emcee(self):
        self.fitter.sampler.run_mcmc(self.fitter._state, 50)
