import pickle
import numpy as np

from refnx.reflect import MixedReflectModel
from .graphproperties import GraphProperties


class DataObject:
    """
    Everything to do with the lifecycle of a dataset.
    """

    # remember how the object was visualised
    _requiredgraphproperties = {
        "lw": float,
        "label": str,
        "linestyle": str,
        "fillstyle": str,
        "marker": str,
        "markersize": float,
        "markeredgecolor": str,
        "markerfacecolor": str,
        "zorder": int,
        "color": str,
        "visible": bool,
    }

    def __init__(self, dataset, constantdq_q=True):
        self.dataset = dataset
        self.name = dataset.name
        self.constantdq_q = constantdq_q

        self._objective = None
        self._model = None

        self.graph_properties = GraphProperties()

        # the evaluator is the Curvefitting object that was used to calculate a
        # fit. The curvefitter object should contain parameters, etc.
        # It should only be overwritten if a fit has been carried out.
        self.curvefitter = None

    @property
    def generative(self):
        if self.model is not None:
            x_err = self.dataset.x_err
            if self.constantdq_q:
                x_err = None
            return self.model(self.dataset.x, x_err=x_err)
        else:
            return None

    @property
    def sld_profile(self):
        try:
            if isinstance(self.model, MixedReflectModel):
                return self.model.structures[0].sld_profile(max_delta_z=1.0)
            else:
                return self.model.structure.sld_profile(max_delta_z=1.0)
        except AttributeError:
            # if self.model is None, or if self.model doesn't have a structure
            # (e.g. MixedReflectModel)
            return None

    @property
    def objective(self):
        return self._objective

    @objective.setter
    def objective(self, objective):
        self._objective = objective

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def save_fit(self, filename):
        fit = self.generative
        if self.model is not None:
            with open(filename, "wb+") as f:
                if fit is not None:
                    np.savetxt(f, np.column_stack((self.dataset.x, fit)))

    def save_model(self, filename):
        if self.model is not None:
            with open(filename, "wb+") as f:
                pickle.dump(self.model, f)

    def refresh(self):
        self.dataset.refresh()
