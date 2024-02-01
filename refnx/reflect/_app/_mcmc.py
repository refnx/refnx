"""
Deals with GUI aspects of MCMC
"""

from pathlib import Path

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.compat import getopenfilename

from refnx.analysis import (
    load_chain,
    process_chain,
    autocorrelation_chain,
    integrated_time,
    GlobalObjective,
    Objective,
)
from refnx.reflect import Structure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure

pth = Path(__file__).absolute().parent
UI_LOCATION = pth / "ui"


class SampleMCMCDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(UI_LOCATION / "sample_mcmc.ui", self)


class ProcessMCMCDialog(QtWidgets.QDialog):
    def __init__(self, objective, chain, folder=None, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(UI_LOCATION / "process_mcmc.ui", self)

        self.objective = objective
        self.folder = folder
        self.chain = chain

        if folder is None:
            self.folder = Path.cwd()

        if self.chain is None:
            model_file_name, ok = getopenfilename(self, "Select chain file")
            if not ok:
                return
            self.folder = Path(model_file_name).parent
            try:
                self.chain = load_chain(model_file_name)
            except Exception as e:
                # a chain load will go wrong quite often I'd expect
                self.chain = None
                print(repr(e))
                return

        if len(self.chain.shape) == 3:
            steps, walkers, varys = self.chain.shape
            self.chain_size.setText(
                f"steps: {steps}, walkers: {walkers}, varys: {varys}"
            )
        else:
            steps, temps, walkers, varys = self.chain.shape
            self.chain_size.setText(
                f"steps: {steps}, temps: {temps}, "
                f"walkers: {walkers}, varys: {varys}"
            )

        self.total_samples.setText(f"Total samples: {steps * walkers}")

        self.burn.setMaximum(steps - 1)
        self.thin.setMaximum(steps - 1)

        acfs = autocorrelation_chain(self.chain)
        time = integrated_time(acfs, tol=1, quiet=True)
        self.autocorrelation_time.setText(
            f"Estimated Autocorrelation Time: {time}"
        )

    @QtCore.Slot(int)
    def on_burn_valueChanged(self, val):
        self.recalculate()

    @QtCore.Slot(int)
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

        self.total_samples.setText(f"Total samples: {steps * walkers}")

        acfs = autocorrelation_chain(lchain)
        time = integrated_time(acfs, tol=1, quiet=True)
        self.autocorrelation_time.setText(
            f"Estimated Autocorrelation Time: {time}"
        )
        self.nplot.setMaximum(steps * walkers)

    @QtCore.Slot()
    def on_buttonBox_accepted(self):
        nthin = self.thin.value()
        nburn = self.burn.value()
        _process_chain(
            self.objective, self.chain, nburn, nthin, folder=self.folder
        )


def _process_chain(objective, chain, nburn, nthin, folder=None):
    # processes the chain for the ProcessMCMCDialog
    if folder is None:
        folder = Path.cwd()

    process_chain(objective, chain, nburn=nburn, nthin=nthin)

    # plot the Autocorrelation function of the chain
    acfs = autocorrelation_chain(chain, nburn=nburn, nthin=nthin)

    fig = Figure()
    FigureCanvas(fig)

    ax = fig.add_subplot(111)
    ax.plot(acfs)
    ax.set_ylabel("autocorrelation")
    ax.set_xlabel("step")
    fig.savefig(folder / "steps-autocorrelation.png")


def _plots(obj, nplot=0, folder=None):
    # create graphs of reflectivity and SLD profiles
    # steps.png, steps_corner.png, steps_sld.png, steps_param_{i}.png
    # Also writes out arrays:
    #      steps.npy, steps_sld.npy
    if folder is None:
        folder = Path.cwd()

    fig = Figure()
    FigureCanvas(fig)

    _, ax = obj.plot(samples=nplot, fig=fig)
    ax.set_ylabel("R")
    ax.set_xlabel("Q / $\\AA$")
    fig.savefig(folder / "steps.png", dpi=1000)

    rng = np.random.default_rng(1)

    # write out a file with the reflectivities
    g = obj._generate_generative_mcmc(ngen=nplot, random_state=rng)
    arr = np.vstack([a for a in g])
    np.save(folder / "steps.npy", arr)

    # corner plot
    try:
        import corner

        kwds = {}
        var_pars = obj.varying_parameters()
        chain = np.array([par.chain for par in var_pars])
        labels = [par.name for par in var_pars]
        chain = chain.reshape(len(chain), -1).T
        kwds["labels"] = labels
        kwds["quantiles"] = [0.16, 0.5, 0.84]
        fig2 = corner.corner(chain, **kwds)

        fig2.savefig(folder / "steps_corner.png")
    except ImportError:
        pass

    # plot sld profiles
    if isinstance(obj, GlobalObjective):
        fig3 = Figure()
        FigureCanvas(fig3)
        ax3 = fig3.add_subplot(111)

        if nplot > 0:
            for idx, o in enumerate(obj.objectives):
                if hasattr(o.model, "structure"):
                    o.model.structure.plot(samples=nplot, fig=fig3)

                # write out all the data for the SLD profiles
                rng = np.random.default_rng(1)
                g = o.model.structure._generate_sld_profile_mcmc(
                    samples=nplot, random_state=rng
                )
                arr = np.vstack([a for a in g])
                np.save(folder / f"steps_sld_{idx}.npy", arr)

        for o in obj.objectives:
            if hasattr(o.model, "structure"):
                ax3.plot(*o.model.structure.sld_profile(), zorder=20)

        ax3.set_ylabel("SLD / $10^{-6}\\AA^{-2}$")
        ax3.set_xlabel("z / $\\AA$")

    elif isinstance(obj, Objective) and hasattr(obj.model, "structure"):
        fig3 = Figure()
        FigureCanvas(fig3)
        fig3, ax3 = obj.model.structure.plot(samples=nplot, fig=fig3)

        # write out all the data for the SLD profiles
        rng = np.random.default_rng(1)
        g = obj.model.structure._generate_sld_profile_mcmc(
            samples=nplot, random_state=rng
        )
        arr = np.vstack([a for a in g])
        np.save(folder / "steps_sld.npy", arr)

    fig3.savefig(folder / "steps_sld.png", dpi=1000)

    # plot the chains so one can see when parameters reach
    # 'equilibrium values'
    for i, vp in enumerate(obj.varying_parameters()):
        label = vp.name
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_ylabel(label)

        for j in range(100):
            ax.plot(vp.chain[:, j].flat)

        fig.savefig(folder / f"steps_param_{i}.png")
