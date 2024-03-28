import operator
import numpy as np

from refnx.analysis import GlobalObjective, Objective
from refnx.analysis.parameter import (
    constraint_tree,
    build_constraint_from_tree,
    Constant,
)
from refnx._lib import flatten
import refnx


_imports = r"""#!/usr/bin/env python

'''
Script exported by refnx for analysing NR/XRR data.

To get help:

    python mcmc.py -h

You will need to install the following packages in a Python interpreter:
  - refnx
  - numpy
  - cython
  - matplotlib
  - scipy

If you wish to run the script on a cluster making use of MPI you'll need to
install:
  - mpi4py
  - schwimmbad

The usage then is:

    mpiexec -n 4 python mcmc.py <other options>
'''

import sys
import time
import argparse
from multiprocessing import Pool
import operator
import numpy as np
from numpy import array
try:
    import tqdm
    import schwimmbad
except ImportError:
    pass

from refnx.analysis import Objective, GlobalObjective, Transform, CurveFitter
from refnx.analysis import Parameter, Parameters, Interval, process_chain
from refnx.analysis import load_chain
from refnx.analysis.parameter import Constant, build_constraint_from_tree
from refnx.analysis.parameter import _BinaryOp, _UnaryOp
from refnx.dataset import ReflectDataset, Data1D

from refnx.reflect import Slab, SLD, Structure, Stack
from refnx.reflect import ReflectModel, LipidLeaflet, MixedReflectModel, Spline
from refnx._lib import flatten

import refnx
# Script created by refnx version: {version}
"""


_main = r"""

def structure_plot(obj, samples=0):
    # plot sld profiles
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if isinstance(obj, GlobalObjective):
        if samples > 0:
            savedparams = np.array(obj.parameters)
            for pvec in obj.parameters.pgen(ngen=samples):
                obj.setp(pvec)
                for o in obj.objectives:
                    if hasattr(o.model, 'structure'):
                        ax.plot(*o.model.structure.sld_profile(),
                                color="k", alpha=0.01)

            # put back saved_params
            obj.setp(savedparams)

        for o in obj.objectives:
            if hasattr(o.model, 'structure'):
                ax.plot(*o.model.structure.sld_profile(), zorder=20)

        ax.set_ylabel('SLD / $10^{-6}\\AA^{-2}$')
        ax.set_xlabel("z / $\\AA$")

    elif isinstance(obj, Objective) and hasattr(obj.model, 'structure'):
        fig, ax = obj.model.structure.plot(samples=samples)

    fig.savefig('steps_sld.png', dpi=1000)


def main(args):
    nwalkers = args.walkers
    nthin = args.thin
    nsteps = args.steps
    ntemps = args.temps
    nplot = args.plot
    nburn = args.burn
    cfile = args.chain
    verbose = args.verbose
    pargs = ()

    try:
        # we may want to use MPI to parallelise
        import schwimmbad
        if args.mpi:
            pool_klass = schwimmbad.MPIPool
    except ImportError:
        pass

    # not wanting MPI, so just use multiprocessing.Pool to parallelise
    if not args.mpi:
        pool_klass = Pool
        pargs = (args.n_cores,)

    with pool_klass(*pargs) as workers:
        # necessary when using MPI.
        if args.mpi and (not workers.is_master()):
            workers.wait()
            sys.exit(0)

        start_time = time.time()

        obj = objective()

        # turn off pthread'ing of reflectivity calculation if MPI. Otherwise
        # the reflectivity calculation will want to spread out over all the
        # available processors.
        if args.mpi:
            _objectives = [obj]
            if isinstance(obj, GlobalObjective):
                _objectives = obj.objectives

            for o in _objectives:
                o.model.threads = 1

        # Create the fitter and fit
        fitter = CurveFitter(obj, nwalkers=nwalkers, ntemps=ntemps)

        if nsteps:
            if cfile:
                # Initialise the walkers with a pre-existing chain
                chain = load_chain(cfile)
                fitter.initialise(chain)
            else:
                # Initialise the walkers by doing a fit, then using the
                # covariance. The workers kwd is only present in scipy >1.2
                fitter.fit('differential_evolution', workers=workers.map)
                fitter.initialise('covar')

            # Buffering is there so the chain file is not written to
            # continuously
            with open('steps.chain', 'w', buffering=500000) as f:
                res = fitter.sample(nsteps, pool=workers.map, f=f,
                                    verbose=verbose, nthin=nthin);
                f.flush()
            process_chain(obj, fitter.chain, nburn=nburn)
        else:
            # the workers kwd is only present in scipy >1.2
            fitter.fit('differential_evolution', workers=workers.map)

        print(str(obj))
        print('\n')
        print(f"Duration (s): {time.time() - start_time}")

        try:
            # create graphs of reflectivity and SLD profiles
            import matplotlib
            import matplotlib.pyplot as plt
            matplotlib.use('agg')

            fig, ax = obj.plot(samples=nplot)
            ax.set_ylabel('R')
            ax.set_xlabel("Q / $\\AA$")
            fig.savefig('steps.png', dpi=1000)

            structure_plot(obj, samples=nplot)

            # corner plot
            fig = obj.corner()
            fig.savefig('steps_corner.png')

            # plot the Autocorrelation function of the chain
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(fitter.acf())
            ax.set_ylabel('autocorrelation')
            ax.set_xlabel('step')
            fig.savefig('steps-autocorrelation.png')
        except ImportError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--walkers', help='number of emcee walkers',
                        type=int, default=200)
    parser.add_argument('-t', '--thin', help='factor to thin chain by',
                        type=int, default=1)
    parser.add_argument('-s', '--steps', help=("number of thinned MCMC steps"
                                               " to save"),
                        type=int, default=1000)
    parser.add_argument('-b', '--burn', help=("number of initial MCMC steps"
                                               " to discard"),
                        type=int, default=0)
    parser.add_argument('-n', '--temps', help=("number of parallel tempering"
                                               " temperatures (requires the"
                                               " ptemcee package)"),
                        type=int, default=-1)
    parser.add_argument('-p', '--plot', help=("create plots of the MCMC"
                                              " using 'plot' samples"),
                        type=int, default=0),
    parser.add_argument('-c', '--chain', help='initialise chain from file',
                        type=str, default='')
    parser.add_argument('-v', '--verbose', help="Displays a progress bar while"
                        " sampling",
                        dest='verbose', action='store_true',  default=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=-1,
                       type=int, help=("Number of processes (uses"
                       "multiprocessing)."))
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    if args.n_cores == -1:
        args.n_cores = None
    main(args)
"""


def code_fragment(objective):
    code = []
    code.append(_imports.format(version=refnx.version.version))

    if isinstance(objective, GlobalObjective):
        _objectives = objective.objectives
    elif isinstance(objective, Objective):
        _objectives = [objective]

    code.append("def objective():")
    tab = "    "

    if len(_objectives) == 1:
        code.append(tab + objective_fragment(0, _objectives[0]))
    else:
        global_objective = [tab]
        global_objective.append("objective_0 = GlobalObjective([")

        for i, o in enumerate(_objectives):
            code.append(tab + objective_fragment(i + 1, o))
            global_objective.append(f"objective_{i + 1}, ")

        global_objective.append("])")

        code.append("".join(global_objective))

    code.extend(_calculate_constraints(0, objective))

    code.append(tab + "return objective_0")

    # add main to make the script executable
    code.append(_main)
    code_str = "\n".join(code)

    try:
        from black import format_str, FileMode

        code_str = format_str(code_str, mode=FileMode())
    except ImportError:
        pass
    finally:
        return code_str


def objective_fragment(i, objective):
    return f"objective_{i} = {objective!r}"


operators = {
    operator.add: "operator.add",
    operator.sub: "operator.sub",
    operator.mul: "operator.mul",
    operator.truediv: "operator.truediv",
    operator.floordiv: "operator.floordiv",
    np.power: "np.power",
    operator.pow: "operator.pow",
    operator.mod: "operator.mod",
    operator.neg: "operator.neg",
    operator.abs: "operator.abs",
    np.sin: "np.sin",
    np.tan: "np.tan",
    np.cos: "np.cos",
    np.arcsin: "np.arcsin",
    np.arctan: "np.arctan",
    np.arccos: "np.arccos",
    np.log10: "np.log10",
    np.log: "np.log",
    np.sqrt: "np.sqrt",
    np.exp: "np.exp",
}


def _calculate_constraints(i, objective):
    # builds constraints strings for an objective which has a local variable
    # name of objective_{i}
    all_pars = list(flatten(objective.parameters))
    var_pars = objective.varying_parameters()

    non_var_pars = [p for p in all_pars if p not in var_pars]

    # now get parameters with constraints
    con_pars = [par for par in non_var_pars if par.constraint is not None]
    tab = "    "

    constrain_strings = [
        tab + f"parameters = list(flatten(objective_{i}.parameters))"
    ]
    for con_par in con_pars:
        idx = all_pars.index(con_par)
        con_tree = constraint_tree(con_par.constraint)
        for j, v in enumerate(con_tree):
            if v in operators:
                con_tree[j] = operators[v]
            elif v in all_pars:
                con_tree[j] = f"parameters[{all_pars.index(v)}]"
            else:
                con_tree[j] = repr(v)
        s = ", ".join(con_tree)
        constraint = f"build_constraint_from_tree([{s}])"
        item = tab + f"parameters[{idx}].constraint = {constraint}"

        constrain_strings.append(item)

    return constrain_strings
