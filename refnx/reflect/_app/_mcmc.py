"""
Deals with GUI aspects of MCMC
"""
import os

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from refnx.analysis import (load_chain, process_chain, autocorrelation_chain,
                            integrated_time)

pth = os.path.dirname(os.path.abspath(__file__))
UI_LOCATION = os.path.join(pth, 'ui')
ProcessMCMCDialogUI = uic.loadUiType(os.path.join(UI_LOCATION,
                                             'process_mcmc.ui'))[0]


class ProcessMCMCDialog(QtWidgets.QDialog, ProcessMCMCDialogUI):
    def __init__(self, objective, chain, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        self.objective = objective

        if chain is None:
            # TODO open file dialogue for chain.
            pass

        self.chain = chain

        if len(chain.shape) == 3:
            steps, walkers, varys = chain.shape
            self.chain_size.setText(
                'steps: {}, walkers: {}, varys: {}'.format(
                    steps, walkers, varys))
        else:
            steps, temps, walkers, varys = chain.shape
            self.chain_size.setText(
                'steps: {}, temps: {}, walkers: {}, varys: {}'.format(
                steps, temps, walkers, varys))

        self.total_samples.setText(
            'Total samples: {}'.format(steps * walkers))

        acfs = autocorrelation_chain(self.chain)
        time = integrated_time(acfs, tol=1, quiet=True)
        self.autocorrelation_time.setText(
            'Estimated Autocorrelation Time: {}'.format(time))
