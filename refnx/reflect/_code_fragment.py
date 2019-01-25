import numpy as np

from refnx.analysis import GlobalObjective, Objective
import refnx


def code_fragment(objective):
    code = []

    code.append('import numpy as np')
    code.append('from numpy import array')

    code.append("from refnx.analysis import Objective, GlobalObjective")
    code.append("from refnx.analysis import Parameter, Parameters, Interval")
    code.append("from refnx.dataset import ReflectDataset")

    code.append('from refnx.reflect import Slab, SLD, Structure')
    code.append('from refnx.reflect import ReflectModel')

    code.append('import refnx')
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
            global_objective.append('objective_{0},'.format(i))

        global_objective.append('])')

        code.append(''.join(global_objective))

    return '\n'.join(code)


def objective_fragment(i, objective):
    return 'objective_{0} = {1}'.format(i, repr(objective))
