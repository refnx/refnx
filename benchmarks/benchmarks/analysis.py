import os.path
import numpy as np

from .common import Benchmark

from refnx.analysis import CurveFitter, Objective, Parameter, Model
from refnx.dataset import Data1D


def line(x, params, *args, **kwds):
    p_arr = np.array(params)
    return p_arr[0] + x * p_arr[1]


class curvefitter(Benchmark):
    repeat = 3

    def setup(self):
        # Reproducible results!
        np.random.seed(123)

        m_true = -0.9594
        b_true = 4.294
        f_true = 0.534
        m_ls = -1.1040757010910947
        b_ls = 5.4405552502319505

        # Generate some synthetic data from the model.
        N = 50
        x = np.sort(10 * np.random.rand(N))
        y_err = 0.1 + 0.5 * np.random.rand(N)
        y = m_true * x + b_true
        y += np.abs(f_true * y) * np.random.randn(N)
        y += y_err * np.random.randn(N)

        data = Data1D(data=(x, y, y_err))

        p = Parameter(b_ls, 'b', vary=True, bounds=(-100, 100))
        p |= Parameter(m_ls, 'm', vary=True, bounds=(-100, 100))

        model = Model(p, fitfunc=line)
        objective = Objective(model, data)
        self.mcfitter = CurveFitter(objective)
        self.mcfitter.initialise('prior')

    def time_sampler(self):
        # to get an idea of how fast the actual sampling is.
        # i.e. the overhead of objective.lnprob, objective.lnprior, etc
        self.mcfitter.sampler.run_mcmc(self.mcfitter._lastpos, 100)
