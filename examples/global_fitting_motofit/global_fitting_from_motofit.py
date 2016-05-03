#!/usr/bin/env python
"""
global_fitting_from_motofit.py [OPTIONS] [-- ARGS]

Sets and executes up a global fitting environment supplied by Motofit in IGOR.

Examples::

    $ python global_fitting_from_motofit.py global_pilot_file.txt

"""
import sys
import numbers
import time
import numpy as np
from refnx.analysis import (CurveFitter, ReflectivityFitFunction, GlobalFitter,
                            to_parameters, Transform, values, names)
from refnx.dataset import ReflectDataset
from argparse import ArgumentParser


def global_fitter_setup(global_pilot_file, dqvals=5.0):
    with open(global_pilot_file, 'r') as f:
        data_files = f.readline().split()
        pilot_files = f.readline().split()

    constraints = np.loadtxt(global_pilot_file, skiprows=2, dtype=int)

    # open the datafiles
    datasets = []
    for data_file in data_files:
        dataset = ReflectDataset(data_file)
        datasets.append(dataset)

    # deal with the individual pilot files
    parameters = []
    for pilot_file in pilot_files:
        pars = np.loadtxt(pilot_file, skiprows=4)

        # lets just assume for now that the data has resolution info
        # and that we're doing a slab model.
        pv = pars[:, 0][:]
        bounds = [(b[0], b[1]) for b in pars[:, 2:4]]
        varies = (pars[:, 1].astype(int) == 0)

        # workout bounds, and account for the fact that MotofitMPI
        # doesn't set bounds for parameters that are fixed
        bounds = []
        for idx in range(np.size(pv)):
            if not varies[idx]:
                bounds.append((0, 2 * pv[idx]))
            else:
                bounds.append(pars[idx, 2:4])

        P = to_parameters(pv,
                          varies=varies,
                          bounds=bounds)
        parameters.append(P)

    # now create CurveFitting instances
    T = Transform('logY')
    fitters = []

    for parameter, dataset in zip(parameters, datasets):
        t_data_y, t_data_ysd = T.transform(dataset.x,
                                           dataset.y,
                                           dataset.y_sd)

        if isinstance(dqvals, numbers.Real):
            _dqvals = float(dqvals)
        else:
            _dqvals = dataset.x_sd

        c = CurveFitter(ReflectivityFitFunction(T.transform),
                        (dataset.x, t_data_y, t_data_ysd),
                        parameter,
                        fcn_kws={'dqvals': _dqvals})
        fitters.append(c)

    # create globalfitter
    # setup constraints
    unique, indices = np.unique(constraints, return_index=True)

    # TODO assertions for checking linkage integrity

    n_datasets = len(datasets)
    max_params = max([len(parameter) for parameter in parameters])

    def is_unique(row, col):
        ravelled_idx = row * n_datasets + col
        return ravelled_idx in indices

    cons = []
    for col in range(n_datasets):
        for row in range(len(parameters[col])):
            if constraints[row, col] == -1 or is_unique(row, col):
                continue
            # so it's not unique, but which parameter does it depend on?
            # find location of master parameter
            master = np.extract(unique == constraints[row, col], indices)[0]
            m_col = master % n_datasets
            m_row = (master - m_col) // n_datasets
            constraint = 'd%u:p%u = d%u:p%u' %(col, row, m_col, m_row)
            cons.append(constraint)

            # also have to rejig the bounds because MotoMPI doesn't
            # set bounds for those that aren't unique. But this is bad for
            # lmfit because it'll clip them.
            par = fitters[col].params['p%u' % row]
            m_par = fitters[m_col].params['p%u' % m_row]
            par.min = m_par.min
            par.max = m_par.max

    global_fitter = GlobalFitter(fitters, constraints=cons)

    # # update the constraints
    # global_fitter.params.update_constraints()

    return global_fitter


def main(argv):
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("global_pilot_file", help="The name of the global pilot"
                                                  " file")
    parser.add_argument("--walkers", "-w", type=int, default=100,
                        help="How many MCMC walkers?")
    parser.add_argument("--steps", "-s", type=int, default=2000,
                        help="How many MCMC steps?")
    parser.add_argument("--burn", "-b", type=int, default=500,
                        help="How many initial MCMC steps do you want to burn?")
    parser.add_argument("--thin", "-t", type=int, default=20,
                        help="Thins the chain by accepting 1 in every 'thin'")
    parser.add_argument("--qres", "-q", type=float, default=5.0,
                        help="Constant dq/q resolution")
    parser.add_argument("--pointqres", "-p", action="store_true", default=False,
                        help="Use point by point resolution smearing (default False)")
    parser.add_argument("--output", "-o", type=str, default='iterations',
                        help="Output file for MCMC chain")
    parser.add_argument("--nprocesses", "-n", type=int, default=4,
                        help="How many processes for parallelisation?")

    args = parser.parse_args(argv)

    # set up global fitting
    if args.pointqres:
        dqvals = None
    else:
        dqvals = args.qres

    if args.nprocesses < 1:
        args.nprocesses = 1

    if args.thin < 1:
        args.thin = 1

    if args.burn < 1:
        args.burn = 1

    global_fitter = global_fitter_setup(args.global_pilot_file,
                                        dqvals=dqvals)

    # do the sampling
    reuse_sampler = False
    n_remaining = args.steps
    done = 0

    sys.stdout.write("Starting MCMC\n")
    sys.stdout.write("-------------\n")
    start = time.time()

    while n_remaining > 0:
        todo = min(100, n_remaining)
        res = global_fitter.emcee(nwalkers=args.walkers,
                                  steps=todo,
                                  burn=args.burn,
                                  thin=args.thin,
                                  workers=args.nprocesses,
                                  reuse_sampler=reuse_sampler)
        n_remaining -= todo
        reuse_sampler = True
        done += todo
        sys.stdout.write("%u iterations in %g seconds\n"
                         % (done, time.time() - start))

    # the flatchain is what we're interested in.
    # make an output array
    # hopefully the chain has been burned and thinned enough.
    output = np.zeros((np.size(res.flatchain, 0), len(res.params)))
    gen = pgen(res.params, res.flatchain)
    for row in output:
        pars = next(gen)
        row[:] = values(pars)[:]

    np.savetxt(args.output, output, header=' '.join(names(res.params)))


def pgen(parameters, flatchain, idx=None):
    # generator for all the different parameters from a flatchain.
    if idx is None:
        idx = range(np.size(flatchain, 0))
    for i in idx:
        vec = flatchain.iloc[i]
        for var_name in flatchain.columns:
            parameters[var_name].value = flatchain.iloc[i][var_name]
        yield parameters

if __name__ == "__main__":
    main(argv=sys.argv[1:])
