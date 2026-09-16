# Copyright (c) 2011-2026 A. R. J. Nelson, Australian Nuclear Science and Technology Organisation
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from refnx.analysis import Parameter, Parameters
from refnx._lib._cutil import c_flatten, c_unique
from refnx._lib.util import flatten, unique

from .common import Benchmark


def create(items, params, rng):
    for i in range(items):
        if rng.uniform() < 0.2:
            f = Parameters()
            _p = create(items - i, f, rng)
            params.append(_p)
        else:
            _p = Parameter(1)
            params.append(_p)
    return params


class Util(Benchmark):
    def setup(self):
        rng = np.random.default_rng(1892908)
        p = Parameters()
        self.p = create(20, p, rng)
        self.flattened = flatten(self.p)

    def time_c_flatten(self):
        list(c_flatten(self.p))

    def time_flatten(self):
        list(flatten(self.p))

    def time_unique(self):
        list(unique(self.flattened))

    def time_c_unique(self):
        list(c_unique(self.flattened))
