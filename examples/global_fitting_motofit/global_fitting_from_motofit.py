#!/usr/bin/env python
"""
global_fitting_from_motofit.py [OPTIONS] [-- ARGS]

Sets and executes up a global fitting environment supplied by Motofit in IGOR.

Examples::

    $ python global_fitting_from_motofit.py global_pilot_file

"""
import sys
import numbers
import time
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
from refnx.analysis import (CurveFitter, ReflectivityFitFunction, GlobalFitter,
                            to_parameters, Transform, values, names)
from refnx.dataset import ReflectDataset


def global_fitter_setup(global_pilot_file, dqvals=5.0):
    # Parse the global_fitter setup from Igor.
    # TODO deal with user generated non-slab models.
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
        t_data_y, t_data_yerr = T.transform(dataset.x,
                                            dataset.y,
                                            dataset.y_err)

        if isinstance(dqvals, numbers.Real):
            _dqvals = float(dqvals)
        else:
            _dqvals = dataset.x_err

        c = CurveFitter(ReflectivityFitFunction(T.transform, workers=True),
                        (dataset.x, t_data_y, t_data_yerr),
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
        for row, val in enumerate(parameters[col]):
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


def _mcmc(args, global_fitter):
    # sample via Markov Chain Monte Carlo
    pos = None
    if args.chain_input is not None:
        pos = np.load(args.chain_input)

    # do the sampling
    chunk_size = 50
    n_remaining = args.steps
    done = 0

    sys.stdout.write("----------------------\n")
    sys.stdout.write("Starting MCMC\n")
    sys.stdout.write("----------------------\n")
    start = time.time()

    reuse_sampler = False
    while n_remaining > 0:
        todo = min(chunk_size, n_remaining)

        res = global_fitter.emcee(nwalkers=args.walkers,
                                  steps=todo,
                                  ntemps=args.ntemps,
                                  burn=0,
                                  thin=1,
                                  workers=args.nprocesses,
                                  pos=pos,
                                  reuse_sampler=reuse_sampler)
        reuse_sampler = True
        n_remaining -= todo
        done += todo
        pos = res.chain

        # write raw chain in npy format. It is unburnt and unthinned
        np.save(args.chain_output, res.chain)

        sys.stdout.write(
            "{0:^7} steps, {1:^7} seconds\n".format(done, time.time() - start)
                         )

    # thin and burn the chain.
    chain = res.chain[..., args.burn::args.thin, :]
    res.chain = np.copy(chain)

    return res


def __resample_mc_iterator(args):
    global_fitter, seed = args
    gf = deepcopy(global_fitter)
    np.random.seed(seed)
    res = gf._resample_mc(1, 'differential_evolution')
    return res.mc


def _resample_mc(args, global_fitter):
    # do the sampling by Resampling Monte Carlo

    sys.stdout.write("----------------------\n")
    sys.stdout.write("Starting resampling MC\n")
    sys.stdout.write("----------------------\n")
    start = time.time()

    # do a single fit first
    output = global_fitter.fit('differential_evolution')

    chunksize = 5 * args.nprocesses
    remaining = args.steps
    done = 0
    mcs = []

    with Pool(args.nprocesses) as pool:
        while remaining > 0:
            todo = min(remaining, chunksize)

            # seeding the random number generator
            seeds = range(done, done + todo)
            gfs = [global_fitter] * todo

            results = pool.map(__resample_mc_iterator, zip(gfs, seeds))

            mcs.append(results)
            done += todo
            remaining -= todo

            sys.stdout.write("{0:^7} steps, {1:^7} seconds\n".format(done, time.time() - start))

    mc = np.squeeze(np.vstack(mcs))

    quantiles = np.percentile(mc, [15.87, 50, 84.13], axis=0)

    params = output.params
    for i, var_name in enumerate(output.var_names):
        std_l, median, std_u = quantiles[:, i]
        params[var_name].value = median
        params[var_name].stderr = 0.5 * (std_u - std_l)
        params[var_name].correl = {}

    params.update_constraints()

    # work out correlation coefficients
    corrcoefs = np.corrcoef(mc.T)

    for i, var_name in enumerate(output.var_names):
        for j, var_name2 in enumerate(output.var_names):
            if i != j:
                output.params[var_name].correl[var_name2] = corrcoefs[i, j]

    output.mc = mc
    output.chain = mc
    output.errorbars = True
    output.nvarys = len(output.var_names)

    return output


def main(argv):
    parser = ArgumentParser(usage=__doc__.lstrip())
    parser.add_argument("global_pilot_file",
                        help="The name of the global pilot file")
    parser.add_argument("--walkers", "-w", type=int, default=100,
                        help="How many MCMC walkers? Default=100")
    parser.add_argument("--steps", "-s", type=int, default=2000,
                        help="How many MCMC steps? Default=2000")
    parser.add_argument("--ntemps", '-T', type=int, default=1,
                        help="How many parallel tempering temperatures? "
                             "Default=1")
    parser.add_argument("--burn", "-b", type=int, default=500,
                        help="How many initial MCMC steps do you want to "
                             "burn? Default=500")
    parser.add_argument("--thin", "-t", type=int, default=20,
                        help="Thins the chain by accepting 1 in every 'thin'. "
                             "Default=20")
    parser.add_argument("--resample", "-r", default=False, action='store_true',
                        help="Don't do Markov Chain Monte Carlo, do resampling "
                             "MC instead. All MCMC parameters are ignored if "
                             "this option is specified.")
    parser.add_argument("--qres", "-q", type=float, default=5.0,
                        help="Constant dq/q resolution. Default=5")
    parser.add_argument("--pointqres", "-p", action="store_true",
                        default=False,
                        help="Use point by point resolution smearing. "
                             "Default=False")
    parser.add_argument("--chain_input", "-i", type=str,
                        help="Initialise/restart emcee with this RAW chain. "
                             "This file is a numpy array (.npy) that would've "
                             "originally been saved by the --chain_output "
                             "option.")
    parser.add_argument("--chain_output", "-c", default='raw_chain.npy',
                        type=str,
                        help="Specify filename for unthinned, unburnt RAW "
                             "chain. The file is saved as a numpy (.npy) "
                             "array. You can use this file if you'd like to do "
                             "the burn/thin procedure yourself. You can also"
                             " use this file to restart the sampling. The array"
                             " has shape (walkers, steps, dims), where dims "
                             "represents the number of parameters you are "
                             "varying. If ntemps is > 1 then the array has "
                             "shape (ntemps, walkers, steps, dims). "
                             "Default=raw_chain.npy")
    parser.add_argument("--output", "-o", type=str, default='iterations',
                        help="Output file for burnt and thinned MCMC chain. "
                             "This is only written once the sampling has "
                             "finished")
    parser.add_argument("--nprocesses", "-n", type=int, default=1,
                        help="How many processes for parallelisation? Default=1")

    args = parser.parse_args(argv)

    # set up global fitting
    if args.pointqres:
        dqvals = None
    else:
        dqvals = args.qres

    if args.nprocesses < 1:
        args.nprocesses = 1

    global_fitter = global_fitter_setup(args.global_pilot_file,
                                        dqvals=dqvals)

    # do the Monte Carlo
    if not args.resample:
        # By Markov Chain Monte Carlo
        if args.thin < 1:
            sys.stdout.write("Can't have thin < 1, setting 'thin' to 1.\n")
            args.thin = 1
        if (args.burn < 0) or (args.burn > args.steps):
            sys.stdout.write("Can't burn < 0 or burn > steps, setting 'burn' to 1.\n")
            args.burn = 1

        res = _mcmc(args, global_fitter)
    else:
        # By resampling
        res = _resample_mc(args, global_fitter)

    # write the iterations.
    _write_results(args.output, res)

    sys.stdout.write("\nFinished MCMC\n")
    sys.stdout.write("-------------\n")
    sys.stdout.write(fit_report(res.params))
    sys.stdout.write("\n-----------------------------------------------------\n")


def filter_dependent_params(params, output):
    # filters dependent parameters from the MonteCarlo output
    # i.e. reject all parameters where expr is not None.
    independent = []
    for idx, param in enumerate(params):
        if param.expr is not None:
            independent.append(idx)

    arr = np.zeros((np.size(output, 0), idx))
    arr = output[:, idx]
    return arr


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
