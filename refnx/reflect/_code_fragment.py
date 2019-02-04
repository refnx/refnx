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
from refnx.analysis import Parameter, Parameters, Interval
from refnx.analysis.parameter import Constant, build_constraint_from_tree
from refnx.analysis.parameter import _BinaryOp, _UnaryOp
from refnx.dataset import ReflectDataset, Data1D

from refnx.reflect import Slab, SLD, Structure
from refnx.reflect import ReflectModel, LipidLeaflet, MixedReflectModel
from refnx._lib import flatten

import refnx
# Script created by refnx version: {version}
""")


_main = (
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--walkers', help='number of emcee walkers',
                        type=int, default=300)
    parser.add_argument('-t', '--thin', help='factor to thin chain by',
                        type=int, default=10)
    parser.add_argument('-s', '--steps', help=("number of thinned MCMC steps"
                                               " to save"),
                        type=int, default=1000)
    parser.add_argument('-o', '--output', help='file to save chain to',
                        type=str)

    args = parser.parse_args()
    nwalkers = args.walkers
    nthin = args.thin
    nsteps = args.steps

    with open('steps.chain', 'w', buffering=500000) as f, Pool() as workers:
        obj = objective()
        # Create the fitter and fit
        fitter = CurveFitter(obj, nwalkers=nwalkers)
        fitter.initialise('prior')

        # the workers kwd is only present in scipy >1.2
        fitter.fit('differential_evolution', workers=workers.map)

        res = fitter.sample(nsteps, pool=workers, f=f, verbose=False,
                            nthin=nthin);
        print(str(obj))
        f.flush()

""")


def code_fragment(objective):
    code = []
    code.append(_imports.format(version=refnx.version.version))

    if isinstance(objective, GlobalObjective):
        _objectives = [objective.objectives]
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
            code.append(objective_fragment(i + 1, o))
            global_objective.append('objective_{0}, '.format(i))

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
