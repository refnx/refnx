#!/bin/bash

"""
Using refnx in a highly parallelised environment using mpi.

You'll need to install:
- refnx
- numpy
- cython
- schwimmbad
- mpi4py
- ptemcee

Usage
-----
mpiexec -n 4 python mpi_parallelisation.py
"""

# Start off by importing necessary packages
import sys
import os.path

import refnx
from schwimmbad import MPIPool

from refnx.reflect import SLD, Slab, ReflectModel
from refnx.dataset import ReflectDataset
from refnx.analysis import (Objective, CurveFitter, Transform)


def setup():
    # load the data.
    DATASET_NAME = os.path.join(refnx.__path__[0],
                                'analysis',
                                'test',
                                'c_PLP0011859_q.txt')

    # load the data
    data = ReflectDataset(DATASET_NAME)

    # the materials we're using
    si = SLD(2.07, name='Si')
    sio2 = SLD(3.47, name='SiO2')
    film = SLD(2, name='film')
    d2o = SLD(6.36, name='d2o')

    structure = si | sio2(30, 3) | film(250, 3) | d2o(0, 3)
    structure[1].thick.setp(vary=True, bounds=(15., 50.))
    structure[1].rough.setp(vary=True, bounds=(1., 6.))
    structure[2].thick.setp(vary=True, bounds=(200, 300))
    structure[2].sld.real.setp(vary=True, bounds=(0.1, 3))
    structure[2].rough.setp(vary=True, bounds=(1, 6))

    model = ReflectModel(structure, bkg=9e-6, scale=1.)
    model.bkg.setp(vary=True, bounds=(1e-8, 1e-5))
    model.scale.setp(vary=True, bounds=(0.9, 1.1))
    model.threads = 1
    # fit on a logR scale, but use weighting
    objective = Objective(model, data, transform=Transform('logY'),
                          use_weights=True)

    return objective


if __name__ == "__main__":
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        # buffering so the program doesn't try to write to the file
        # constantly
        with open('steps.chain', 'w', buffering=500000) as f:
            objective = setup()
            # Create the fitter and fit
            fitter = CurveFitter(objective, nwalkers=300, ntemps=15)
            fitter.initialise('prior')
            fitter.fit('differential_evolution')
            # thin by 10 so we have a smaller filesize
            fitter.sample(100, pool=pool, f=f, verbose=False, nthin=10);
            f.flush()
