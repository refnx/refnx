import numpy as np

from refnx.analysis import GlobalObjective, Objective
import refnx


def code_fragment(objective):
    code = []

    code.append('import numpy as np')
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

    # start by creating the dataset
    if len(_objectives) == 1:
        code.append('objective = {0}'.format(repr(o)))
    else:
        global_objective = []
        global_objective.append('objective = GlobalObjective([')

        for i, o in enumerate(_objectives):
            code.append('objective_{0} = {1}'.format(i, repr(o)))
            global_objective.append('objective_{0},'.format(i))

        global_objective.append('])')

        code.append(''.join(global_objective))

    return '\n'.join(code)
