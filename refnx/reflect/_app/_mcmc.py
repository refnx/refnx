"""
Deals with GUI aspects of MCMC
"""
import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from refnx.analysis import (load_chain, process_chain, autocorrelation_chain,
                            integrated_time, GlobalObjective, Objective)
from refnx.reflect import Structure

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

        self.burn.setMaximum(steps - 1)
        self.thin.setMaximum(steps - 1)

        acfs = autocorrelation_chain(self.chain)
        time = integrated_time(acfs, tol=1, quiet=True)
        self.autocorrelation_time.setText(
            'Estimated Autocorrelation Time: {}'.format(time))

    @QtCore.pyqtSlot(int)
    def on_burn_valueChanged(self, val):
        self.recalculate()

    @QtCore.pyqtSlot(int)
    def on_thin_valueChanged(self, val):
        self.recalculate()

    def recalculate(self):
        nthin = self.thin.value()
        nburn = self.burn.value()

        lchain = self.chain[nburn::nthin]

        if len(lchain.shape) == 3:
            steps, walkers, varys = lchain.shape
        else:
            steps, temps, walkers, varys = lchain.shape

        self.total_samples.setText(
            'Total samples: {}'.format(steps * walkers))

        acfs = autocorrelation_chain(lchain)
        time = integrated_time(acfs, tol=1, quiet=True)
        self.autocorrelation_time.setText(
            'Estimated Autocorrelation Time: {}'.format(time))

    @QtCore.pyqtSlot()
    def on_buttonBox_accepted(self):
        nthin = self.thin.value()
        nburn = self.burn.value()

        process_chain(self.objective, self.chain, nburn=nburn, nthin=nthin)

        _plots(self.objective, 200)


def _plots(obj, nplot=0):
    # create graphs of reflectivity and SLD profiles
    import matplotlib
    import matplotlib.pyplot as plt

    fig, ax = obj.plot(samples=nplot)
    ax.set_ylabel('R')
    ax.set_xlabel("Q / $\\AA$")
    fig.savefig('steps.png', dpi=1000)
    plt.close(fig)

    # corner plot
    # fig = obj.corner()
    # fig.savefig('steps_corner.png')

    # plot sld profiles
    if isinstance(obj, GlobalObjective):
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        if nplot > 0:
            savedparams = np.array(obj.parameters)
            for pvec in obj.parameters.pgen(ngen=samples):
                obj.setp(pvec)
                for o in obj.objectives:
                    if hasattr(o.model, 'structure'):
                        ax2.plot(*o.model.structure.sld_profile(),
                                color="k", alpha=0.01)

            # put back saved_params
            obj.setp(savedparams)

        for o in obj.objectives:
            if hasattr(o.model, 'structure'):
                ax2.plot(*o.model.structure.sld_profile(), zorder=20)

        ax2.set_ylabel('SLD / $10^{-6}\\AA^{-2}$')
        ax2.set_xlabel("z / $\\AA$")

    elif isinstance(obj, Objective) and hasattr(obj.model, 'structure'):
        fig2, ax2 = obj.model.structure.plot(samples=nplot)
    fig2.savefig('steps_sld.png', dpi=1000)
    plt.close(fig2)
