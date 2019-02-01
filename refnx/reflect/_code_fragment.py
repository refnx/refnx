import operator
import numpy as np

from refnx.analysis import GlobalObjective, Objective
from refnx.analysis.parameter import (constraint_tree,
                                      build_constraint_from_tree,
                                      Constant)
from refnx._lib import flatten
import refnx


def code_fragment(objective):
    code = []

    code.append('import operator')
    code.append('import numpy as np')
    code.append('from numpy import array')

    code.append("from refnx.analysis import Objective, GlobalObjective,"
                " Transform")
    code.append("from refnx.analysis import Parameter, Parameters, Interval")
    code.append("from refnx.analysis.parameter import Constant,"
                " build_constraint_from_tree, _BinaryOp, _UnaryOp")
    code.append("from refnx.dataset import ReflectDataset")

    code.append('from refnx.reflect import Slab, SLD, Structure')
    code.append('from refnx.reflect import ReflectModel, LipidLeaflet')
    code.append('from refnx._lib import flatten')

    code.append('import refnx')
    code.append("# Script created by refnx"
                " version: {}".format(refnx.version.version))
    code.append('print(refnx.version.version)')

    if isinstance(objective, GlobalObjective):
        _objectives = [objective.objectives]
    elif isinstance(objective, Objective):
        _objectives = [objective]

    if len(_objectives) == 1:
        code.append(objective_fragment(0, _objectives[0]))
    else:
        global_objective = []
        global_objective.append('objective_0 = GlobalObjective([')

        for i, o in enumerate(_objectives):
            code.append(objective_fragment(i + 1, o))
            global_objective.append('objective_{0}, '.format(i))

        global_objective.append('])')

        code.append(''.join(global_objective))

    code.extend(_calculate_constraints(0, objective))

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

    constrain_strings = ["parameters = list(flatten("
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
        item = "parameters[{}].constraint = {}".format(idx, constraint)

        constrain_strings.append(item)

    return constrain_strings
