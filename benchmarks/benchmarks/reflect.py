import os.path
import numpy as np
import pickle

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
        self.repeat = 20
        self.number = 10

    def time_cabeles(self):
        c_abeles(self.q, self.layers)

    def time_abeles(self):
        abeles(self.q, self.layers)

    def time_reflectivity_constant_dq_q(self):
        reflectivity(self.q, self.layers)

    def time_reflectivity_pointwise_dq(self):
        reflectivity(self.q, self.layers, dq=0.05 * self.q)


class Reflect(Benchmark):
    timeout = 120.
    # repeat = 2

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

        self.p = structure361[1].thick
        structure361[1].thick.vary = True
        structure361[1].thick.range(5, 20)
        structure361[2].thick.vary = True
        structure361[2].thick.range(100, 220)

        structure361[2].sld.real.vary = True
        structure361[2].sld.real.range(0.2, 1.5)

        self.structure361 = structure361
        self.model361 = model361

        # e361.x_err = None
        self.objective = Objective(self.model361,
                                   e361)
        self.fitter = CurveFitter(self.objective, nwalkers=200)
        self.fitter.initialise('jitter')

    def time_reflect_emcee(self):
        # test how fast the emcee sampler runs in serial mode
        self.fitter.sampler.run_mcmc(self.fitter._state, 30)

    def time_reflect_sampling_parallel(self):
        # discrepancies in different runs may be because of different numbers
        # of processors
        self.model361.threads = 1
        self.fitter.sample(30, pool=-1)

    def time_pickle_objective(self):
        # time taken to pickle an objective
        s = pickle.dumps(self.objective)
        pickle.loads(s)

    def time_pickle_model(self):
        # time taken to pickle a model
        s = pickle.dumps(self.model361)
        pickle.loads(s)

    def time_pickle_model(self):
        # time taken to pickle a parameter
        s = pickle.dumps(self.p)
        pickle.loads(s)

    def time_structure_slabs(self):
        self.structure361.slabs()
