import operator
import numpy as np

from refnx.analysis import GlobalObjective, Objective
from refnx.analysis.parameter import (constraint_tree,
                                      build_constraint_from_tree,
                                      Constant)
from refnx._lib import flatten
import refnx


_imports = ("""import argparse
from multiprocessing import Pool
import operator
import numpy as np
from numpy import array
try:
    import tqdm
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
""")


_main = (r"""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--walkers', help='number of emcee walkers',
                        type=int, default=300)
    parser.add_argument('-t', '--thin', help='factor to thin chain by',
                        type=int, default=10)
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

    args = parser.parse_args()
    nwalkers = args.walkers
    nthin = args.thin
    nsteps = args.steps
    ntemps = args.temps
    nplot = args.plot
    nburn = args.burn
    cfile = args.chain

    with Pool() as workers:
        obj = objective()
        # Create the fitter and fit
        fitter = CurveFitter(obj, nwalkers=nwalkers, ntemps=ntemps)

        if nsteps:
            if cfile:
                chain = load_chain(cfile)
                fitter.initialise(chain)
            else:
                # the workers kwd is only present in scipy >1.2
                fitter.fit('differential_evolution', workers=workers.map)
                fitter.initialise('covar')

            with open('steps.chain', 'w', buffering=500000) as f:
                res = fitter.sample(nsteps, pool=workers, f=f, verbose=False,
                                    nthin=nthin);
                f.flush()
            process_chain(obj, fitter.chain, nburn=nburn)
        else:
            # the workers kwd is only present in scipy >1.2
            fitter.fit('differential_evolution', workers=workers.map)

        print(str(obj))

    try:
        fig, ax = obj.plot(samples=nplot)
        ax.set_ylabel('R')
        ax.set_xlabel("Q / $\\AA$")
        fig.savefig('steps.png', dpi=1000)

        structure_plot(obj, samples=nplot)

        # corner plot
        fig = obj.corner()
        fig.savefig('steps_corner.png')

    except ImportError:
        pass
""")


def code_fragment(objective):
    code = []
    code.append(_imports.format(version=refnx.version.version))

    if isinstance(objective, GlobalObjective):
        _objectives = objective.objectives
    elif isinstance(objective, Objective):
        _objectives = [objective]

    code.append('def objective():')
    tab = '    '

    if len(_objectives) == 1:
        code.append(tab + objective_fragment(0, _objectives[0]))
    else:
        global_objective = [tab]
        global_objective.append('objective_0 = GlobalObjective([')

        for i, o in enumerate(_objectives):
            code.append(tab + objective_fragment(i + 1, o))
            global_objective.append('objective_{0}, '.format(i + 1))

        global_objective.append('])')

        code.append(''.join(global_objective))

    code.extend(_calculate_constraints(0, objective))

    code.append(tab + 'return objective_0')

    # add main to make the script executable
    code.append(_main)
    return '\n'.join(code)


def objective_fragment(i, objective):
    return 'objective_{0} = {1}'.format(i, repr(objective))


operators = {operator.add: 'operator.add', operator.sub: 'operator.sub',
             operator.mul: 'operator.mul',
             operator.truediv: 'operator.truediv',
             operator.floordiv: 'operator.floordiv', np.power: 'np.power',
             operator.pow: 'operator.pow', operator.mod: 'operator.mod',
             operator.neg: 'operator.neg', operator.abs: 'operator.abs',
             np.sin: 'np.sin', np.tan: 'np.tan', np.cos: 'np.cos',
             np.arcsin: 'np.arcsin', np.arctan: 'np.arctan',
             np.arccos: 'np.arccos', np.log10: 'np.log10', np.log: 'np.log',
             np.sqrt: 'np.sqrt', np.exp: 'np.exp'}


def _calculate_constraints(i, objective):
    # builds constraints strings for an objective which has a local variable
    # name of objective_{i}
    all_pars = list(flatten(objective.parameters))
    var_pars = objective.varying_parameters()

    non_var_pars = [p for p in all_pars if p not in var_pars]

    # now get parameters with constraints
    con_pars = [par for par in non_var_pars if par.constraint is not None]
    tab = '    '

    constrain_strings = [tab + "parameters = list(flatten("
                         "objective_{}.parameters))".format(i)]
    for con_par in con_pars:
        idx = all_pars.index(con_par)
        con_tree = constraint_tree(con_par.constraint)
        for j, v in enumerate(con_tree):
            if v in operators:
                con_tree[j] = operators[v]
            elif v in all_pars:
                con_tree[j] = 'parameters[{}]'.format(all_pars.index(v))
            else:
                con_tree[j] = repr(v)
        s = ', '.join(con_tree)
        constraint = "build_constraint_from_tree([" + s + "])"
        item = tab + "parameters[{}].constraint = {}".format(idx, constraint)

        constrain_strings.append(item)

    return constrain_strings
