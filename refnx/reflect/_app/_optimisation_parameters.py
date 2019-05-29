import os.path
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSlot
import numpy as np


UI_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ui')


class OptimisationParameterView(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super(OptimisationParameterView, self).__init__(parent)

        self.ui = uic.loadUi(
            os.path.join(UI_LOCATION,
                         'optimisation.ui'),
            self)

    def parameters(self, method):
        """
        Parameters
        ----------
        method : {'differential_evolution'}
            The fit method

        Returns
        -------
        opt_par : dict
            Options for fitting algorithm.
        """
        kws = {}
        if method == 'DE':
            kws['strategy'] = self.ui.de_strategy.currentText()
            kws['maxiter'] = self.ui.de_maxiter.value()
            kws['popsize'] = self.ui.de_popsize.value()
            kws['tol'] = self.ui.de_tol.value()
            kws['atol'] = self.ui.de_atol.value()
            kws['init'] = self.ui.de_initialisation.currentText()
            kws['recombination'] = self.ui.de_recombination.value()
            mutation_lb = self.ui.de_mutation_lb.value()
            mutation_ub = self.ui.de_mutation_ub.value()
            kws['mutation'] = (min(mutation_lb, mutation_ub),
                               max(mutation_lb, mutation_ub))
            target = self.ui.de_target.currentText()
            if target == 'log-posterior':
                target = 'nlpost'
            else:
                target = 'nll'
            kws['target'] = target

        return kws
