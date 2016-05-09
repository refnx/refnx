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
from argparse import ArgumentParser

import numpy as np
from refnx.analysis import (CurveFitter, ReflectivityFitFunction, GlobalFitter,
                            to_parameters, Transform, values, names)
from refnx.dataset import ReflectDataset
from lmfit.printfuncs import fit_report


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

        c = CurveFitter(ReflectivityFitFunction(T.transform, parallel=True),
                        (dataset.x, t_data_y, t_data_ysd),
                        parameter,
                        fcn_kws={'dqvals': _dqvals})
        fitters.append(c)

    # create globalfitter
    # setup constraints
    unique, indices = np.unique(constraints, return_index=True)

    # TODO assertions for checking linkage integrity

    n_datasets = len(datasets)

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
            constraint = 'd%u:p%u = d%u:p%u' % (col, row, m_col, m_row)
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


def MCMC_analysis(global_pilot_file, walkers=100, steps=2000, burn=500,
                  thin=20, dqvals=5.0, pos=None, output=None,
                  workers=1, chunk=50):

    global_fitter = global_fitter_setup(global_pilot_file,
                                        dqvals=dqvals)

    # do the sampling
    n_remaining = steps
    done = 0

    sys.stdout.write("Starting MCMC\n")
    sys.stdout.write("-------------\n")
    start = time.time()

    while n_remaining > 0:
        todo = min(chunk, n_remaining)

        if done > burn:
            _burn = burn
        if (done - burn) > thin:
            _thin = thin

        res = global_fitter.emcee(nwalkers=args.walkers,
                                  steps=todo,
                                  burn=_burn,
                                  thin=_thin,
                                  workers=workers,
                                  pos=pos)
        n_remaining -= todo
        done += todo
        pos = res.chain

        # write the results so far. You should only get anything if the
        # number of steps is > args.burn.
        _write_results(arg.output, res)

        # write raw chain in npy format
        np.save(args.chain_output, res.chain)

        sys.stderr.write(
            "{0:^7} steps, {1:^7} seconds\n".format(done, time.time() - start)
                         )

    sys.stderr.write("\nFinished MCMC\n")
    sys.stderr.write("-------------\n")
    sys.stderr.write(fit_report(res.params))


def main(argv):
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("global_pilot_file",
                        help="The name of the global pilot file")
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
    parser.add_argument("--chain_input", "-i", type=str,
                        help="Initialise/restart emcee with this RAW chain. This file"
                             " is a numpy array (.npy) that would've originally been"
                             "saved by the --chain_output option.")
    parser.add_argument("--chain_output", "-c", default='chain.npy', type=str,
                        help="Specify filename for unthinned, unburnt RAW"
                             " chain. The file is saved as a numpy (.npy)"
                             " array. You can use this file if you'd like to do"
                             " the burn/thin procedure yourself. You can also use"
                             " this file to restart the sampling. The array has"
                             " shape (walkers, steps, walkers, dims), where dims"
                             " represents the number of parameters you are"
                             " varying.")
    parser.add_argument("--output", "-o", type=str, default='iterations',
                        help="Output file for burnt and thinned MCMC chain")
    parser.add_argument("--nprocesses", "-n", type=int, default=1,
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
        sys.stdout.write("Can't have thin < 1, setting to 1.\n")
        args.thin = 1

    if (args.burn < 0) or (args.burn > args.steps):
        sys.stdout.write("Can't burn < 0 or burn > steps, setting to 1.\n")
        args.burn = 1

    global_fitter = global_fitter_setup(args.global_pilot_file,
                                        dqvals=dqvals)

    pos = None
    if args.chain_input is not None:
        pos = np.load(args.chain_input)

    # do the sampling
    chunk_size = 50
    n_remaining = args.steps
    done = 0

    sys.stdout.write("Starting MCMC\n")
    sys.stdout.write("-------------\n")
    start = time.time()

    burn = 0
    thin = 1
    while n_remaining > 0:
        todo = min(chunk_size, n_remaining)

        if done > args.burn:
            burn = args.burn
        if (done - args.burn) > thin:
            thin = args.thin

        res = global_fitter.emcee(nwalkers=args.walkers,
                                  steps=todo,
                                  burn=burn,
                                  thin=thin,
                                  workers=args.nprocesses,
                                  pos=pos)
        n_remaining -= todo
        done += todo
        pos = res.chain

        # write the results so far. You should only get anything if the
        # number of steps is > args.burn.
        _write_results(arg.output, res)

        # write raw chain in npy format
        np.save(args.chain_output, res.chain)

        sys.stdout.write(
            "{0:^7} steps, {1:^7} seconds\n".format(done, time.time() - start)
                         )

    sys.stdout.write("\nFinished MCMC\n")
    sys.stdout.write("-------------\n")
    sys.stdout.write(fit_report(res.params))


def _write_results(f, emcee_result):
    # the flatchain is what we're interested in.
    # make an output array
    # hopefully the chain has been burned and thinned enough.
    output = np.zeros((np.size(emcee_result.flatchain, 0),
                       len(emcee_result.params)))
    gen = pgen(emcee_result.params, emcee_result.flatchain)
    for row in output:
        pars = next(gen)
        row[:] = values(pars)[:]

    np.savetxt(f, output, header=' '.join(names(emcee_result.params)))


def pgen(parameters, flatchain, idx=None):
    # generator for all the different parameters from a flatchain.
    if idx is None:
        idx = range(np.size(flatchain, 0))
    for i in idx:
        for var_name in flatchain.columns:
            parameters[var_name].value = flatchain.iloc[i][var_name]
        yield parameters

if __name__ == "__main__":
    main(argv=sys.argv[1:])
