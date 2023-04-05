import os.path
from copy import deepcopy
import pickle
import os
import sys
import time
import csv
from multiprocessing import get_context

import numpy as np
import scipy
import matplotlib
import periodictable

from qtpy.compat import getopenfilename, getopenfilenames, getsavefilename
from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtCore import Qt

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import matplotlib.artist as artist
import matplotlib.lines as lines


from .SLD_calculator_view import SLDcalculatorView
from .datastore import DataStore
from .treeview_gui_model import (
    TreeModel,
    Node,
    DatasetNode,
    DataObjectNode,
    ComponentNode,
    StructureNode,
    PropertyNode,
    ReflectModelNode,
    ParNode,
    TreeFilter,
    find_data_object,
    SlabNode,
    StackNode,
)
from ._lipid_leaflet import LipidLeafletDialog
from ._optimisation_parameters import OptimisationParameterView
from ._spline import SplineDialog
from ._mcmc import ProcessMCMCDialog, SampleMCMCDialog, _plots, _process_chain

import refnx
from refnx.analysis import (
    CurveFitter,
    Objective,
    Transform,
    GlobalObjective,
    Parameter,
)
from refnx.reflect import (
    SLD,
    ReflectModel,
    Slab,
    Stack,
    Structure,
    MixedReflectModel,
)
from refnx.dataset import Data1D
from refnx.reflect._code_fragment import code_fragment
from refnx._lib import unique, flatten, MapWrapper


# matplotlib.use('QtAgg')
UI_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


class MotofitMainWindow(QtWidgets.QMainWindow):
    """
    Main View window for Motofit
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # load the GUI from the ui file
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, "motofit.ui"), self)

        self.error_handler = QtWidgets.QErrorMessage()

        #######################################################################
        # Everything ending in 'model' refers to a QtAbstract<x>Model.  These
        # are the basis for GUI elements.  They also contain data.
        data_container = DataStore()

        # a flag that is used by update_gui_model to decide whether to redraw
        # object graphs in response to the treeModel being changed
        self._hold_updating = False

        # set up tree view
        self.treeModel = TreeModel(data_container)

        # the filter controls what rows are presented in the treeView
        self.treeFilter = TreeFilter(self.treeModel)
        self.treeFilter.setSourceModel(self.treeModel)
        self.ui.treeView.setModel(self.treeFilter)

        self.ui.treeView.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )

        # context menu for the treeView
        self.context_menu = OpenMenu(self.ui.treeView)

        self.ui.treeView.customContextMenuRequested.connect(self.context_menu)
        self.context_menu.add_to_fit_action.triggered.connect(
            self.on_add_to_fit_button_clicked
        )
        self.context_menu.remove_from_fit_action.triggered.connect(
            self.on_remove_from_fit_action
        )
        self.context_menu.link_action.triggered.connect(self.link_action)
        self.context_menu.link_equivalent_action.triggered.connect(
            self.link_equivalent_action
        )
        self.context_menu.unlink_action.triggered.connect(self.unlink_action)
        self.context_menu.copy_from_action.triggered.connect(
            self.copy_from_action
        )
        self.context_menu.add_mixed_area.triggered.connect(
            self.add_mixed_area_action
        )
        self.context_menu.remove_mixed_area.triggered.connect(
            self.remove_mixed_area_action
        )
        self.actionLink_Selected.triggered.connect(self.link_action)
        self.actionUnlink_selected_Parameters.triggered.connect(
            self.unlink_action
        )
        self.actionLink_Equivalent_Parameters.triggered.connect(
            self.link_equivalent_action
        )

        self.treeModel.dataChanged.connect(self.tree_model_data_changed)
        self.treeModel.rowsRemoved.connect(self.tree_model_structure_changed)
        self.treeModel.rowsMoved.connect(self.tree_model_structure_changed)
        self.treeModel.rowsInserted.connect(self.tree_model_structure_changed)

        # list view for datasets being fitted
        self.currently_fitting_model = CurrentlyFitting(self)
        self.ui.currently_fitting.setModel(self.currently_fitting_model)
        #######################################################################

        # attach the reflectivity graphs and the SLD profiles
        self.attach_graphs_to_gui()

        # holds miscellaneous information on program settings
        self.settings = ProgramSettings()
        self.settings.current_dataset_name = "theoretical"

        theoretical = data_container["theoretical"]
        model = theoretical.model
        dataset = theoretical.dataset
        resolution = self.settings.resolution / 100.0
        transform = Transform(self.settings.transformdata)
        fit = model(dataset.x, x_err=dataset.x * resolution)
        fit, _ = transform(dataset.x, fit)
        sld = model.structure.sld_profile()

        graph_properties = theoretical.graph_properties
        line = self.reflectivitygraphs.axes[0].plot(
            dataset.x,
            fit,
            color="r",
            linestyle="-",
            lw=1,
            label="theoretical",
        )[0]
        line.set_pickradius(5.0)
        graph_properties["ax_fit"] = line

        graph_properties["ax_sld_profile"] = self.sldgraphs.axes[0].plot(
            sld[0], sld[1], linestyle="-", color="r"
        )[0]

        self.restore_settings()

        self.sample_mcmc_dialog = SampleMCMCDialog(self)
        self.spline_dialog = SplineDialog(self)
        self.sld_calculator = SLDcalculatorView(self)
        self.lipid_leaflet_dialog = LipidLeafletDialog(self)
        self.optimisation_parameters = OptimisationParameterView(self)
        self.data_object_selector = DataObjectSelectorDialog(self)
        self.data_object_selector.addItems(["theoretical"])

        self.ui.treeView.setColumnWidth(0, 200)
        h = self.ui.treeView.header()
        h.setMinimumSectionSize(100)
        # h.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)

        # redirect stdout to a console window
        console = EmittingStream()
        sys.stdout = console
        console.textWritten.connect(self.writeTextToConsole)

        print("Session started at:", time.asctime(time.localtime(time.time())))

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    @QtCore.Slot(QtGui.QDropEvent)
    def dropEvent(self, event):
        m = event.mimeData()
        urls = m.urls()

        # convert url's to files
        urls_as_files = [url.toLocalFile() for url in urls]
        self._load_files_different_types(urls_as_files)

    def _load_files_different_types(self, urls_as_files):
        # if you've only got one url then try and load it as an experiment
        if len(urls_as_files) == 1:
            try:
                self._restore_state(urls_as_files[0])
                return
            except Exception:
                pass

        # then try and load urls as data files
        try:
            loaded_names = self.load_data(urls_as_files)
        except Exception as e:
            print(e)
            loaded_names = []

        # then try and load the remainder as models
        remainder_urls = set(urls_as_files).difference(set(loaded_names))
        for url in remainder_urls:
            try:
                self.load_model(url)
                continue
            except Exception:
                pass

    @QtCore.Slot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event):
        m = event.mimeData()
        if m.hasUrls():
            event.acceptProposedAction()

    def writeTextToConsole(self, text):
        self.ui.console_text_edit.moveCursor(
            QtGui.QTextCursor.MoveOperation.End
        )
        self.ui.console_text_edit.insertPlainText(text)

    def _saveState(self, experiment_file_name):
        state = {}
        self.settings.experiment_file_name = experiment_file_name
        state["datastore"] = self.treeModel.datastore
        state["history"] = self.ui.console_text_edit.toPlainText()
        state["settings"] = self.settings
        state["refnx.version"] = refnx.version.version

        fit_list = self.currently_fitting_model
        state["currently_fitting"] = fit_list.datasets
        state["requirements.txt"] = self.requirements()

        with open(os.path.join(experiment_file_name), "wb") as f:
            pickle.dump(state, f, -1)

        self.setWindowTitle("Motofit - " + experiment_file_name)

    @QtCore.Slot()
    def on_actionSave_File_triggered(self):
        if os.path.isfile(self.settings.experiment_file_name):
            self._saveState(self.settings.experiment_file_name)
        else:
            self.on_actionSave_File_As_triggered()

    @QtCore.Slot()
    def on_actionSave_File_As_triggered(self):
        experiment_file_name, ok = getsavefilename(
            self, "Save experiment as:", "experiment.mtft"
        )

        if not ok:
            return

        path, ext = os.path.splitext(experiment_file_name)
        if ext != ".mtft":
            experiment_file_name = path + ".mtft"

        self._saveState(experiment_file_name)

    def _restore_state(self, experiment_file_name):
        with open(experiment_file_name, "rb") as f:
            state = pickle.load(f)

        if not state:
            print("Couldn't load experiment")
            return

        # we've successfully loaded a pickle, but it may not be a MTFT file
        # do some rudimentary checks
        if (
            isinstance(state, dict)
            and "history" in state
            and "datastore" in state
        ):
            pass
        else:
            raise ValueError("Not an experiment file")

        # remove and re-add datasets onto the GUI.
        self.remove_graphs_from_gui()
        self.attach_graphs_to_gui()

        try:
            self.ui.console_text_edit.setPlainText(state["history"])
            self.settings = state["settings"]
            self.settings.experiment_file_name = experiment_file_name
            self.restore_settings()
        except KeyError as e:
            print(repr(e))
            print("\n-------------------------------------------------------")
            print("\nThese are the packages used when the file was saved:\n")
            print(state.get("requirements.txt", ""))
            return

        try:
            self.treeModel._data = state["datastore"]

            # amend the internal state to compensate for mtft files saved in
            # older versions missing attributes saved in later versions.
            self.compensate_older_versions()

            self.treeModel.rebuild()

            ds = [d for d in self.treeModel.datastore]
            self.add_data_objects_to_graphs(ds)
            self.update_gui_model(ds)
            # self.reflectivitygraphs.draw()
        except Exception as e:
            version = state.get("refnx.version", "N/A")
            msg(
                "Failed to load experiment. It may have been saved in a"
                " previous refnx version ({}). Please use that version to"
                " continue with analysis, refnx will now"
                " close.".format(version)
            )
            raise e

        try:
            while self.data_object_selector.data_objects.count():
                self.data_object_selector.data_objects.takeItem(0)

            self.data_object_selector.addItems(self.treeModel.datastore.names)

            self.currently_fitting_model = CurrentlyFitting(self)
            fit_list = self.currently_fitting_model
            self.ui.currently_fitting.setModel(fit_list)
            fit_list.addItems(state["currently_fitting"])

        except KeyError as e:
            print(repr(e))
            return

    def restore_settings(self):
        """
        applies the program settings to the GUI
        """
        title = "Motofit"
        if len(self.settings.experiment_file_name):
            title += " - " + self.settings.experiment_file_name
        self.setWindowTitle(title)

        self.select_fitting_algorithm(self.settings.fitting_algorithm)
        self.ui.use_errors_checkbox.setChecked(self.settings.useerrors)
        self.settransformoption(self.settings.transformdata)

    def compensate_older_versions(self):
        """
        Amends the internal state of the program to add attributes that may
        be missing in experiment files that were saved in earlier versions of
        the GUI.
        """
        # add interfaces attribute to all Components (added in v0.1.8)
        # another way of doing it would be to use
        # `self.__dict__.get('_interfaces', None)` in the interfaces property,
        # but that is slower.

        # add bounds._logprob attribute to all parameter bounds (added in
        # v0.1.9)

        # iterate through structures if model is a MixedReflectModel.
        # v0.1.11

        # added ReflectModel.dq_type
        # v0.1.12

        # added Parameter._stderr
        # v0.1.13

        # removes picker key from graphproperties (matplotlib 3.4.0 causes
        # issues)

        # adds Parameter.units attribute
        # v0.1.21

        # try and compensate for energy dispersive machinery. This won't be
        # able to compensate for all possible fails.
        # v0.1.27

        from refnx.analysis.bounds import Interval

        for do in self.treeModel.datastore:
            model = do.model
            model.dq_type = "pointwise"

            if not hasattr(model, "_q_offset"):
                model._q_offset = Parameter(0.0, name="q_offset")

            if isinstance(model, MixedReflectModel):
                strcs = model.structures
            else:
                strcs = [model.structure]

            for s in strcs:
                if not hasattr(s, "wavelength"):
                    s.wavelength = None

                for component in s:
                    if isinstance(component, Slab) and isinstance(
                        component.sld, SLD
                    ):
                        component.sld.dispersive = False

                    if not hasattr(component, "_interfaces"):
                        component._interfaces = None

                parameters = model.parameters
                for parameter in flatten(parameters):
                    # parameter._stderr was added in 9b96ecx9
                    # check that a parameter has it
                    if not hasattr(parameter, "_stderr"):
                        v = None
                        # check to see if a parameter.stderr already existed
                        if (
                            hasattr(parameter, "stderr")
                            and parameter.stderr is not None
                        ):
                            v = parameter.stderr
                        parameter._stderr = v

                    if not hasattr(parameter, "units"):
                        parameter.units = None

                    bnd = parameter.bounds
                    if isinstance(bnd, Interval) and not hasattr(
                        bnd, "_logprob"
                    ):
                        bnd._logprob = 0
                        bnd._set_bounds(bnd.lb, bnd.ub)

                # pop picker attribute from graphproperties
                gp = do.graph_properties
                for line in [
                    "data_properties",
                    "fit_properties",
                    "sld_profile_properties",
                ]:
                    gp[line].pop("picker", None)

    def apply_settings_to_params(self, params):
        for key in self.settings.__dict__:
            params[key] = self.settings[key]

    @QtCore.Slot()
    def on_actionLoad_File_triggered(self):
        experimentFileName, ok = getopenfilename(
            self,
            caption="Select Experiment File",
            filters="Experiment Files (*.mtft)",
        )
        if not ok:
            return

        try:
            self._restore_state(experimentFileName)
        except Exception:
            pass

    def load_data(self, files):
        """
        Loads datasets into the app

        Parameters
        ----------
        files : sequence
            Sequence of `str` specifying filenames to load the datasets from

        Returns
        -------
        fnames : sequence
            Sequence of `str` containing the filenames of the successfully
            loaded datasets.
        """
        datastore = self.treeModel.datastore

        existing_data_objects = datastore.names

        data_objects = []
        fnames = []
        for file in files:
            if os.path.isfile(file):
                try:
                    data_object = self.treeModel.load_data(file)
                    if data_object is not None:
                        data_objects.append(data_object)
                        fnames.append(file)
                except Exception:
                    continue

        loaded_data_objects = [
            data_object.name for data_object in data_objects
        ]
        new_names = [
            n for n in loaded_data_objects if n not in existing_data_objects
        ]

        # give a newly loaded data object a simple model. You don't want to do
        # this for an object that's already been loaded.
        for name in new_names:
            fronting = SLD(0, name="fronting")
            sio2 = SLD(3.47, name="1")
            backing = SLD(2.07, name="backing")
            s = fronting() | sio2(15, 3) | backing(0, 3)
            data_object_node = self.treeModel.data_object_node(name)
            model = ReflectModel(s)
            model.name = name
            data_object_node.set_reflect_model(model)

        # for the intersection of loaded and old, refresh the plot.
        refresh_names = [
            n for n in existing_data_objects if n in loaded_data_objects
        ]

        refresh_data_objects = [datastore[name] for name in refresh_names]
        self.redraw_data_object_graphs(refresh_data_objects)

        # for totally new, then add to graphs
        new_data_objects = [datastore[name] for name in new_names]
        self.add_data_objects_to_graphs(new_data_objects)

        self.calculate_chi2(data_objects)

        # add newly loads to the data object selector dialogue
        self.data_object_selector.addItems(new_names)
        return fnames

    @QtCore.Slot()
    def on_actionLoad_Data_triggered(self):
        """
        you load data
        """
        files = getopenfilenames(self, caption="Select Reflectivity Files")

        if files:
            self.load_data(files[0])

    @QtCore.Slot()
    def on_actionRemove_Data_triggered(self):
        """
        you remove data
        """
        # retrieve data_objects that need to be removed
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle("Select datasets to remove")
        ok = self.data_object_selector.exec()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]
        if "theoretical" in names:
            names.pop(names.index("theoretical"))

        fit_list = self.currently_fitting_model

        for which_dataset in names:
            # remove from list of datasets to be fitted, if present
            if which_dataset in fit_list.datasets:
                fit_list.removeItems([fit_list.datasets.index(which_dataset)])

            self.reflectivitygraphs.remove_trace(datastore[which_dataset])
            self.sldgraphs.remove_trace(datastore[which_dataset])
            self.treeModel.remove_data_object(which_dataset)

            # remove from data object selector
            self.data_object_selector.removeItem(which_dataset)

    @QtCore.Slot()
    def on_actionSave_Fit_triggered(self):
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle("Select fits to save")
        ok = self.data_object_selector.exec()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if dialog.exec():
            folder = dialog.selectedFiles()
            for name in names:
                datastore[name].save_fit(
                    os.path.join(folder[0], "fit_" + name + ".dat")
                )

    @QtCore.Slot()
    def on_actionSave_SLD_Curve_triggered(self):
        # saves an SLD curve as a text file
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle("Select SLD plots to save")
        ok = self.data_object_selector.exec()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if dialog.exec():
            folder = dialog.selectedFiles()
            for name in names:
                data_object = datastore[name]
                if data_object.sld_profile is not None:
                    # it may be None if it's a mixed area model
                    sld_file_name = os.path.join(
                        folder[0], "sld_" + name + ".dat"
                    )
                    sld_curve = np.array(data_object.sld_profile).T
                    np.savetxt(sld_file_name, sld_curve)

    def load_model(self, model_file_name):
        with open(model_file_name, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, ReflectModel):
            msg("The pkl file you were loading was not a ReflectModel")
            return

        data_object_node = self.treeModel.data_object_node(model.name)
        if data_object_node is not None:
            data_object_node.set_reflect_model(model)
        else:
            # there is not dataset with that name, put the model over the
            # theoretical one
            data_object_node = self.treeModel.data_object_node("theoretical")
            data_object_node.set_reflect_model(model)
        self.update_gui_model([data_object_node.data_object])

    @QtCore.Slot()
    def on_actionLoad_Model_triggered(self):
        # load a model from a pickle file
        model_file_name, ok = getopenfilename(self, "Select Model File")
        if not ok:
            return
        self.load_model(model_file_name)

    @QtCore.Slot()
    def on_actionSave_Model_triggered(self):
        # save a model to a pickle file
        # which model are you saving?
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle("Select models to save")
        ok = self.data_object_selector.exec()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]

        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setWindowTitle("Where do you want to save the models?")
        if dialog.exec():
            folder = dialog.selectedFiles()

            for name in names:
                fname = os.path.join(folder[0], "coef_" + name + ".pkl")
                model = datastore[name]
                model.save_model(fname)

    @QtCore.Slot()
    def on_actionProcess_MCMC_triggered(self):
        """
        Process an MCMC chain. Can only do against current fitting setup though
        """
        names_to_fit = self.currently_fitting_model.datasets
        # retrieve data_objects
        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]
        objective = self.create_objective(data_objects)

        try:
            dialog = ProcessMCMCDialog(objective, None, parent=self)
            if dialog.chain is None:
                return
            dialog.exec()
            print(str(objective))
            _plots(objective, nplot=dialog.nplot.value(), folder=dialog.folder)
        except Exception as e:
            print(repr(e))
            msg(
                "MCMC processing went wrong. The MCMC chain can only be"
                " processed against the fitting setup that created it."
            )

    @QtCore.Slot()
    def on_actionExport_parameters_triggered(self):
        # save all parameter values to a text file
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle("Select parameters to export")
        ok = self.data_object_selector.exec()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]

        suggested_name = os.path.join(os.getcwd(), "coefficients.csv")
        fname, ok = getsavefilename(
            self, "Exported file name:", suggested_name
        )
        if not ok:
            return

        with open(fname, "w", newline="") as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=",",
                quotechar="|",
                quoting=csv.QUOTE_MINIMAL,
            )
            for name in names:
                model = datastore[name].model
                writer.writerow([name])
                writer.writerow([p.name for p in flatten(model.parameters)])
                writer.writerow([p.value for p in flatten(model.parameters)])
                writer.writerow([p.stderr for p in flatten(model.parameters)])

    @QtCore.Slot()
    def on_actionExport_Code_Fragment_triggered(self):
        # exports an executable script for the current fitting system.
        # this script can be used for e.g. MCMC sampling.
        names_to_fit = self.currently_fitting_model.datasets

        if not names_to_fit:
            return

        # retrieve data_objects
        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]

        objective = self.create_objective(data_objects)
        code = code_fragment(objective)

        suggested_name = os.path.join(os.getcwd(), "mcmc.py")
        modelFileName, ok = getsavefilename(
            self, "Save code fragment as:", suggested_name
        )
        if not ok:
            return

        try:
            with open(modelFileName, "w") as f:
                f.write(code)
        except Exception as e:
            print(e)

    def select_fitting_algorithm(self, method):
        meth = {
            "LM": self.ui.actionLevenberg_Marquardt,
            "MCMC": self.ui.actionMCMC,
            "L-BFGS-B": self.ui.actionL_BFGS_B,
            "SHGO": self.ui.actionSHGO,
            "dual_annealing": self.ui.actionDual_Annealing,
            "DE": self.ui.actionDifferential_Evolution,
        }
        self.settings.fitting_algorithm = method
        meth[method].setChecked(True)
        meth.pop(method)
        for k, v in meth.items():
            v.setChecked(False)

    @QtCore.Slot()
    def on_actionDifferential_Evolution_triggered(self):
        self.select_fitting_algorithm("DE")

    @QtCore.Slot()
    def on_actionMCMC_triggered(self):
        self.select_fitting_algorithm("MCMC")

    @QtCore.Slot()
    def on_actionDual_Annealing_triggered(self):
        self.select_fitting_algorithm("dual_annealing")

    @QtCore.Slot()
    def on_actionSHGO_triggered(self):
        self.select_fitting_algorithm("SHGO")

    @QtCore.Slot()
    def on_actionLevenberg_Marquardt_triggered(self):
        self.select_fitting_algorithm("LM")

    @QtCore.Slot()
    def on_actionL_BFGS_B_triggered(self):
        self.select_fitting_algorithm("L-BFGS-B")

    def change_Q_range(self, qmin, qmax, numpnts):
        data_object_node = self.treeModel.data_object_node("theoretical")

        theoretical = data_object_node._data
        dataset = theoretical.dataset

        new_x = np.linspace(qmin, qmax, numpnts)
        new_y = np.zeros_like(new_x) * np.nan
        dataset.data = (new_x, new_y)
        data_object_node.set_dataset(dataset)
        self.update_gui_model([theoretical])

    @QtCore.Slot()
    def on_actionChange_Q_range_triggered(self):
        datastore = self.treeModel.datastore
        theoretical = datastore["theoretical"]
        qmin = min(theoretical.dataset.x)
        qmax = max(theoretical.dataset.x)
        numpnts = len(theoretical.dataset)

        dvalidator = QtGui.QDoubleValidator(-2.0e-308, 2.0e308, 6)

        qrangeGUI = uic.loadUi(os.path.join(UI_LOCATION, "qrangedialog.ui"))
        qrangeGUI.numpnts.setValue(numpnts)
        qrangeGUI.qmin.setValidator(dvalidator)
        qrangeGUI.qmax.setValidator(dvalidator)
        qrangeGUI.qmin.setText(str(qmin))
        qrangeGUI.qmax.setText(str(qmax))

        ok = qrangeGUI.exec()
        if ok:
            self.change_Q_range(
                float(qrangeGUI.qmin.text()),
                float(qrangeGUI.qmax.text()),
                qrangeGUI.numpnts.value(),
            )

    @QtCore.Slot()
    def on_actionTake_Snapshot_triggered(self):
        snapshotname, ok = QtWidgets.QInputDialog.getText(
            self, "Take a snapshot", "snapshot name"
        )
        if not ok:
            return

        # is the snapshot already present?
        datastore = self.treeModel.datastore
        snapshot_exists = datastore[snapshotname]
        data_object = self.treeModel.snapshot(snapshotname)

        if snapshot_exists is not None:
            self.redraw_data_object_graphs([data_object])
        else:
            self.add_data_objects_to_graphs([data_object])

        # add newly loads to the data object selector dialogue
        self.data_object_selector.addItems([snapshotname])

    @QtCore.Slot()
    def on_actionResolution_smearing_triggered(self):
        currentVal = self.settings.quad_order
        value, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Resolution Smearing",
            "Number of points for Gaussian Quadrature",
            currentVal,
            17,
        )
        if not ok:
            return
        self.settings.quad_order = value
        self.update_gui_model([])

    @QtCore.Slot()
    def on_actionBatch_Fit_triggered(self):
        datastore = self.treeModel.datastore
        if len(datastore) < 2:
            return msg("You have no loaded datasets")

        alg = self.settings.fitting_algorithm
        if alg == "MCMC":
            return msg("It's not possible to do MCMC in batch fitting mode")

        # need to retrieve the theoretical data_object because we're going to
        # use its model.
        theoretical = datastore["theoretical"]

        self.data_object_selector.setWindowTitle(
            "Select datasets to batch fit (using the theoretical model)"
        )
        ok = self.data_object_selector.exec()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]

        # iterate and fit over all the selected datasets, but first copy the
        # model from the theoretical model because it's unlikely you're going
        # to setup all the individual models first.
        progress = QtWidgets.QProgressDialog(
            "Batch fitting progress", "Stop", 0, len(names), parent=self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(True)
        progress.setValue(0)
        progress.show()

        gui_update_list = []
        # turn of graph/treeview updating during fitting, it takes up a lot of
        # resources. Just do it all at once at the end.
        self._hold_updating = True
        last_time = time.time()
        completed = 0
        try:
            for name in names:
                if progress.wasCanceled():
                    raise StopIteration()

                data_object = datastore[name]
                if data_object.name == "theoretical":
                    continue
                new_model = deepcopy(theoretical.model)
                new_model.name = name
                data_object_node = self.treeModel.data_object_node(name)
                data_object_node.set_reflect_model(new_model)
                successfully_fitted = self.fit_data_objects([data_object])
                if successfully_fitted:
                    gui_update_list.append(successfully_fitted)

                completed += 1
                # update progress bar every 3 secs
                if (time.time() - last_time) > 3.0:
                    progress.setValue(completed)
                    last_time = time.time()

        except StopIteration:
            pass
        finally:
            self._hold_updating = False
            gui_update_list = list(flatten(gui_update_list))
            self.update_gui_fitted_data_objects(gui_update_list)
            progress.close()

    @QtCore.Slot()
    def on_actionRefresh_Data_triggered(self):
        """
        you are refreshing existing datasets
        """
        try:
            self.treeModel.refresh()
            self.redraw_data_object_graphs(None, all=True)
        except FileNotFoundError:
            print(
                "FileNotFoundError: one or more datafiles is no longer in"
                "their original location"
            )
            msg(
                "FileNotFoundError: one or more datafiles is no longer in"
                "their original location"
            )

    @QtCore.Slot()
    def on_actionlogY_vs_X_triggered(self):
        self.settransformoption("logY")

    @QtCore.Slot()
    def on_actionY_vs_X_triggered(self):
        self.settransformoption("lin")

    @QtCore.Slot()
    def on_actionYX4_vs_X_triggered(self):
        self.settransformoption("YX4")

    @QtCore.Slot()
    def on_actionYX2_vs_X_triggered(self):
        self.settransformoption("YX2")

    def settransformoption(self, transform):
        self.ui.actionlogY_vs_X.setChecked(False)
        self.ui.actionY_vs_X.setChecked(False)
        self.ui.actionYX4_vs_X.setChecked(False)
        self.ui.actionYX2_vs_X.setChecked(False)
        if transform is None:
            self.ui.actionY_vs_X.setChecked(True)
            transform = "lin"
        if transform == "lin":
            self.ui.actionY_vs_X.setChecked(True)
        elif transform == "logY":
            self.ui.actionlogY_vs_X.setChecked(True)
        elif transform == "YX4":
            self.ui.actionYX4_vs_X.setChecked(True)
        elif transform == "YX2":
            self.ui.actionYX2_vs_X.setChecked(True)
        self.settings.transformdata = transform

        self.redraw_data_object_graphs(None, all=True)

        # need to relimit graphs and display on a log scale if the transform
        # has changed
        self.reflectivitygraphs.axes[0].autoscale(
            axis="both", tight=False, enable=True
        )
        self.reflectivitygraphs.axes[0].relim()
        if transform in ["lin", "YX2"]:
            self.reflectivitygraphs.axes[0].set_yscale("log")
        else:
            self.reflectivitygraphs.axes[0].set_yscale("linear")
        self.reflectivitygraphs.draw()

    @QtCore.Slot()
    def on_actionAbout_triggered(self):
        aboutui = uic.loadUi(os.path.join(UI_LOCATION, "about.ui"))

        licence_dir = os.path.join(UI_LOCATION, "licences")
        licences = os.listdir(licence_dir)
        licences.remove("about")

        text = [refnx.version.version]
        with open(
            os.path.join(licence_dir, "about"),
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            text.append("".join(f.readlines()))

        for licence in licences:
            fname = os.path.join(licence_dir, licence)
            with open(fname, "r", encoding="utf-8") as f:
                text.append("".join(f.readlines()))

        display_text = "\n_______________________________________\n".join(text)
        aboutui.textBrowser.setText(display_text)
        aboutui.exec()

    @QtCore.Slot()
    def on_actiondocumentation_triggered(self):
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl("https://refnx.readthedocs.io/en/latest/")
        )

    @QtCore.Slot()
    def on_actionPython_Packages_triggered(self):
        aboutui = uic.loadUi(os.path.join(UI_LOCATION, "about.ui"))
        text = self.requirements()
        aboutui.textBrowser.setText(text)
        aboutui.exec()

    def requirements(self):
        # returns a string of the packages used in the GUI Python environment
        pkgs = [refnx, np, scipy, matplotlib, periodictable]
        versions = []
        for pkg in pkgs:
            versions.append(f"{pkg.__name__}=={pkg.__version__}")

        return "\n".join(versions)

    @QtCore.Slot()
    def on_actionAutoscale_graph_triggered(self):
        self.reflectivitygraphs.autoscale()

    @QtCore.Slot()
    def on_actionSLD_calculator_triggered(self):
        self.sld_calculator.show()

    @QtCore.Slot()
    def on_actionOptimisation_parameters_triggered(self):
        self.optimisation_parameters.show()

    @QtCore.Slot()
    def on_actionLipid_browser_triggered(self):
        self.lipid_leaflet_dialog.show()

    @QtCore.Slot()
    def on_add_layer_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return msg(
                "Select a single row within a Structure to insert a"
                " new Component."
            )

        index = selected_indices[0]

        # from filter to model
        index = self.mapToSource(index)

        if not index.isValid():
            return

        item = index.internalPointer()
        hierarchy = item.hierarchy()
        # reverse so that we see bottom up. We want to see if something is a
        # Component first, then whether it's a StackNode.
        hierarchy.reverse()

        # the row you selected was within the component list
        _component = [
            i
            for i in hierarchy
            if (isinstance(i, ComponentNode) or isinstance(i, StackNode))
        ]
        if not _component:
            return msg(
                "Select a single location within a Structure to insert"
                " a new Component."
            )

        # work out which component you have.
        component = _component[0]
        host = component.parent()
        idx = component.row()
        if isinstance(host, StructureNode) and idx == len(host._data) - 1:
            return msg(
                "You can't append a layer after the backing medium,"
                " select a previous layer"
            )

        # what type of component shall we add?
        comp_type = ["Slab", "LipidLeaflet", "Spline", "Stack"]
        which_type, ok = QtWidgets.QInputDialog.getItem(
            self,
            "What Component type did you want to add?",
            "",
            comp_type,
            editable=False,
        )
        if not ok:
            return

        if which_type == "Slab":
            c = _default_slab(parent=self)
        elif which_type == "LipidLeaflet":
            self.lipid_leaflet_dialog.hide()
            ok = self.lipid_leaflet_dialog.exec()
            if not ok:
                return
            c = self.lipid_leaflet_dialog.component()
        elif which_type == "Spline":
            # if isinstance(host, StackNode):
            #     msg("Can't add Splines to a Stack")
            #     return

            ok = self.spline_dialog.exec()
            if not ok:
                return
            c = self.spline_dialog.component()
        elif which_type == "Stack":
            s = _default_slab()
            c = Stack(components=[s], name="Stack")

        host.insert_component(idx + 1, c)

    @QtCore.Slot()
    def on_remove_layer_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return msg("Select a single row within a Structure to remove.")

        index = selected_indices[0]
        if not index.isValid():
            return

        # from filter to model
        index = self.mapToSource(index)

        item = index.internalPointer()
        hierarchy = item.hierarchy()
        # reverse so that we see bottom up. We want to see if something is a
        # Component first, then whether it's a StackNode.
        hierarchy.reverse()

        # the row you selected was within the component list
        _component = [
            i
            for i in hierarchy
            if (isinstance(i, ComponentNode) or isinstance(i, StackNode))
        ]
        if not _component:
            return msg(
                "Select a single Component within a Structure to" " remove"
            )

        # work out which component you have.
        component = _component[0]
        host = component.parent()
        idx = component.row()
        if isinstance(host, StructureNode) and idx in [0, len(host._data) - 1]:
            return msg("You can't remove the fronting or backing media")

        # TODO: unlink any parameters that might depend on this component?

        # all checking done, remove a layer
        host.remove_component(idx)

    @QtCore.Slot()
    def on_auto_limits_button_clicked(self):
        names_to_fit = self.currently_fitting_model.datasets

        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]
        # auto set the limits for the theoretical model, because it's used as
        # a springboard for the batch fit.
        data_objects.append(datastore["theoretical"])

        for data_object in data_objects:
            # retrive data_object node from the treeModel
            node = self.treeModel.data_object_node(data_object.name)
            descendants = node.descendants()
            # filter out all the parameter nodes
            par_nodes = [n for n in descendants if isinstance(n, ParNode)]
            var_par_nodes = [n for n in par_nodes if n.parameter.vary]

            for par in var_par_nodes:
                p = par.parameter
                val = p.value
                bounds = p.bounds
                if val < 0:
                    bounds.ub = 0
                    bounds.lb = 2 * val
                else:
                    bounds.lb = 0
                    bounds.ub = 2 * val

                parent, row = par.parent(), par.row()
                idx1 = self.treeModel.index(row, 3, parent.index)
                idx2 = self.treeModel.index(row, 4, parent.index)
                self.treeModel.dataChanged.emit(idx1, idx2)

    @QtCore.Slot()
    def on_add_to_fit_button_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        fit_list = self.currently_fitting_model

        to_be_added = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)

            data_object_node = find_data_object(index)
            if not data_object_node:
                continue
            # see if it's already in the list
            name = data_object_node.data_object.name
            to_be_added.append(name)

        fit_list.addItems(unique(to_be_added))

    @QtCore.Slot()
    def on_remove_from_fit_button_clicked(self):
        # work out what datasets are selected in the listwidget
        # remove all those that are selected
        fit_list = self.currently_fitting_model
        selected_items = self.ui.currently_fitting.selectedIndexes()

        rows = [i.row() for i in selected_items]
        fit_list.removeItems(rows)

    def on_remove_from_fit_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()
        fit_list = self.currently_fitting_model

        to_remove = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)

            data_object_node = find_data_object(index)
            if not data_object_node:
                continue
            # see if it's already in the list
            name = data_object_node.data_object.name
            if name in fit_list.datasets:
                to_remove.append(fit_list.datasets.index(name))

        fit_list.removeItems(list(unique(to_remove)))

    @QtCore.Slot()
    def on_do_fit_button_clicked(self):
        """
        you should do a fit
        """
        names_to_fit = self.currently_fitting_model.datasets

        if not names_to_fit:
            return msg("Please add datasets to fit.")

        # retrieve data_objects
        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]

        successfully_fitted = self.fit_data_objects(data_objects)
        if successfully_fitted:
            self.update_gui_fitted_data_objects(successfully_fitted)

    def create_objective(self, data_objects):
        """
        Creates an Objective for a list of DataObject.
        """

        # performs a global fit to the list of data_objects
        t = Transform(self.settings.transformdata)
        useerrors = self.settings.useerrors

        # make objectives
        objectives = []
        for data_object in data_objects:
            # this is if the user requested constant dq/q. If that's the case
            # then make a new dataset to hide the resolution information
            dataset_t = data_object.dataset
            if data_object.constantdq_q and dataset_t.x_err is not None:
                dataset_t = Data1D(data=dataset_t)
                dataset_t.x_err = None

            # can't have negative points if fitting as logR vs Q
            if self.settings.transformdata == "logY":
                dataset_t = self.filter_neg_reflectivity(dataset_t)

            objective = Objective(
                data_object.model,
                dataset_t,
                name=data_object.name,
                transform=t,
                use_weights=useerrors,
            )
            objectives.append(objective)

        if len(objectives) == 1:
            objective = objectives[0]
        else:
            objective = GlobalObjective(objectives)

        return objective

    def fit_data_objects(
        self, data_objects, alg=None, opt_kws=None, mcmc_kws=None
    ):
        """
        Simultaneously fits a sequence of datasets

        Parameters
        ----------
        data_objects: list of DataObjects

        alg: str
            One of the fitting algorithms. If None then queries the GUI state
            for the algorithm to use.

        opt_kws: dict
            Specify options for the optimisation

        mcmc_kws: dict
            Specify options for the MCMC sampling

        Returns
        -------
        fitted_dataobjects : list of DataObjects
            A list of successfully fitted datasets
        """

        objective = self.create_objective(data_objects)

        # figure out how many varying parameters
        vp = objective.varying_parameters()
        if not vp:
            return msg("No parameters are being varied.")

        methods = {
            "DE": "differential_evolution",
            "LM": "least_squares",
            "L-BFGS-B": "L-BFGS-B",
            "dual_annealing": "dual_annealing",
            "SHGO": "shgo",
            "MCMC": "MCMC",
        }

        if alg not in methods:
            alg = self.settings.fitting_algorithm

        # obtain optimisation parameters (maxiter, etc)
        if opt_kws is None:
            kws = self.optimisation_parameters.parameters(alg)
        else:
            kws = {}.update(opt_kws)

        if methods[alg] != "MCMC":
            fitter = CurveFitter(objective)
            progress = ProgressCallback(self, objective=objective)

            if alg == "L-BFGS-B":
                maxiter = kws.pop("maxiter")
                kws["options"] = {"maxiter": maxiter}

            if alg != "LM":
                progress.show()
                kws["callback"] = progress.callback

            if sys.stderr is None:
                # for pythonw, sys.stderr = None
                kws["verbose"] = False

            try:
                # workers is added to differential evolution in scipy 1.2
                if alg == "DE":
                    with MapWrapper(-1) as workers:
                        kws["workers"] = workers
                        kws["updating"] = "deferred"
                        fitter.fit(method=methods[alg], **kws)
                else:
                    fitter.fit(method=methods[alg], **kws)

                print(str(objective))
            except StopIteration as e:
                # user probably aborted the fit
                # but it's still worth creating a fit curve, so don't return
                # in this catch block
                # xk should be the best fit so far.
                text = e.args[0]
                xk = e.args[1]

                msg(repr(text))
                print(repr(text))

                objective.setp(xk)
                print(objective)
            except Exception as e:
                # Typically shown when sensible limits weren't provided
                msg(repr(e))
                progress.close()
                return []

            progress.close()
        else:
            if mcmc_kws is None:
                ok = self.sample_mcmc_dialog.exec()
                if not ok:
                    return []
                nwalkers = self.sample_mcmc_dialog.walkers.value()
                init = self.sample_mcmc_dialog.init.currentText()
                nsteps = self.sample_mcmc_dialog.steps.value()
                nthin = self.sample_mcmc_dialog.thin.value()
                ntemps = self.sample_mcmc_dialog.temps.value()
                verbose = True

                folder_dialog = QtWidgets.QFileDialog(
                    parent=self, caption="Select location to save MCMC output"
                )
                folder_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
                # folder_dialog.setWindowModality(Qt.WindowModality.WindowModal)
                if folder_dialog.exec():
                    folder = folder_dialog.selectedFiles()[0]
                else:
                    return []
            else:
                nwalkers = mcmc_kws.get("nwalkers", 200)
                init = mcmc_kws.get("init", "jitter")
                nsteps = mcmc_kws.get("nsteps", 100)
                nthin = mcmc_kws.get("nthin", 1)
                ntemps = mcmc_kws.get("ntemps", -1)
                folder = mcmc_kws.get("folder", ".")
                nplot = mcmc_kws.get("nplot", 200)
                nburn = mcmc_kws.get("nburn", 0)
                verbose = mcmc_kws.get("verbose", True)

            if ntemps in [-1, 0, 1]:
                ntemps = -1

            verbose = verbose and sys.stderr is not None

            fitter = CurveFitter(objective, ntemps=ntemps, nwalkers=nwalkers)

            try:
                fitter.initialise(pos=init)
                progress = QtWidgets.QProgressDialog(
                    "MCMC progress", "Abort", 0, nsteps, parent=self
                )
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setAutoClose(True)
                progress.setValue(0)
                progress.show()

                # only want to update every few seconds - updating the progress
                # bar can chew processor cycles.
                completed = [0]
                last_time = [time.time()]

                def callback(coords, logprob):
                    completed[0] += 1
                    if (time.time() - last_time[0]) > 2:
                        last_time[0] = time.time()
                        val = completed[0]

                        progress.setValue(val)
                        if progress.wasCanceled():
                            raise StopIteration("Sampling aborted")

                with open(
                    os.path.join(folder, "steps.chain"), "w"
                ) as f, get_context().Pool() as workers:
                    fitter.sample(
                        nsteps,
                        f=f,
                        verbose=verbose,
                        nthin=nthin,
                        callback=callback,
                        pool=workers.map,
                    )

            except StopIteration:
                pass
            except Exception as e:
                progress.close()
                msg(repr(e))
                print(repr(e))
                return []
            progress.close()

            # process the samples
            def close(dialog=None):
                if hasattr(dialog, "close"):
                    dialog.close()

            try:
                if mcmc_kws is None:
                    dialog = ProcessMCMCDialog(
                        objective, fitter.chain, folder=folder, parent=self
                    )
                    dialog.exec()
                    close(dialog)
                    nplot = dialog.nplot.value()
                else:
                    dialog = None
                    _process_chain(
                        objective, fitter.chain, nburn, nthin, folder=folder
                    )

                # create MCMC graphs
                _plots(objective, nplot=nplot, folder=folder)
            except Exception as e:
                close(dialog)
                msg(repr(e))
                print(repr(e))
                return []

            print(str(objective))

        return data_objects

    def update_gui_fitted_data_objects(self, data_objects):
        """
        Updates the GUI for a sequence of successfully fitted DataObject.

        The data tree is updated first with new model values, then chi2 is
        updated for all the datasets. Finally the graphs for the datasets
        are updated.
        This method is separated out from `update_gui_model`, as it is changing
        values in the tree model.

        Parameters
        ----------
        data_objects: list of DataObject
        """

        # mark models as having been updated.
        # prevent the GUI from updating whilst we change all the values.
        # This means we have to update the graphs separately at the end of
        # this method.
        self._hold_updating = True

        for data_object in data_objects:
            # retrive data_object node from the treeModel
            node = self.treeModel.data_object_node(data_object.name)
            descendants = node.descendants()
            # filter out all the parameter nodes
            par_nodes = [n for n in descendants if isinstance(n, ParNode)]
            var_par_nodes = [n for n in par_nodes if n.parameter.vary]

            for par in var_par_nodes:
                # for some reason the view doesn't know how to display float64
                # coerce to a float
                if par.parameter.stderr is not None:
                    par.parameter.stderr = float(par.parameter.stderr)
                    parent, row = par.parent(), par.row()
                    idx1 = self.treeModel.index(row, 1, parent.index)
                    idx2 = self.treeModel.index(row, 2, parent.index)
                    self.treeModel.dataChanged.emit(idx1, idx2)

        # re-enable the GUI updating whilst we change all the values
        self._hold_updating = False

        self.add_data_objects_to_graphs(data_objects)
        # calculates chi2 and redraws generative model
        self.update_gui_model(data_objects)

    @QtCore.Slot(int)
    def on_only_fitted_stateChanged(self, arg_1):
        """
        only display fitted parameters
        """
        self.treeFilter.only_fitted = arg_1
        self.treeFilter._fitted_datasets = (
            self.currently_fitting_model.datasets
        )
        self.treeFilter.invalidateFilter()

    @QtCore.Slot(int)
    def on_use_errors_checkbox_stateChanged(self, arg_1):
        """
        want to weight by error bars, recalculate chi2
        """

        if arg_1:
            use = True
        else:
            use = False

        self.settings.useerrors = use
        self.update_gui_model([])

    def mapToSource(self, index):
        # converts a model index in the filter model to the original model
        return self.treeFilter.mapToSource(index)

    @QtCore.Slot(QtCore.QModelIndex)
    def on_treeView_clicked(self, index):
        index = self.mapToSource(index)
        if not index.isValid():
            return None

        item = index.internalPointer()
        if not isinstance(item, ParNode):
            return

        self.currentCell = {}

        par = item.parameter
        val = par.value

        if val < 0:
            lowlim = 2 * val
            hilim = 0
        else:
            lowlim = 0
            hilim = 2 * val

        self.currentCell["item"] = item
        self.currentCell["val"] = val
        self.currentCell["lowlim"] = lowlim
        self.currentCell["hilim"] = hilim
        self.currentCell["readyToChange"] = True

    @QtCore.Slot(int)
    def on_paramsSlider_valueChanged(self, arg_1):
        # short circuit if the treeview hasn't been clicked yet
        if not hasattr(self, "currentCell"):
            return

        c = self.currentCell
        item = c["item"]

        if not c["readyToChange"]:
            return

        val = c["lowlim"] + (arg_1 / 1000.0) * np.fabs(
            c["lowlim"] - c["hilim"]
        )

        item.parameter.value = val

        # get the index for the change
        row = item.index.row()
        # who is the parent
        parent_index = item.parent().index
        index = self.treeModel.index(row, 1, parent_index)
        self.treeModel.dataChanged.emit(
            index, index, [Qt.ItemDataRole.EditRole]
        )

    @QtCore.Slot()
    def on_paramsSlider_sliderReleased(self):
        try:
            self.currentCell["readyToChange"] = False
            self.ui.paramsSlider.setValue(499)
            item = self.currentCell["item"]

            val = item.parameter.value

            if val < 0:
                low_lim = 2 * val
                hi_lim = 0
            else:
                low_lim = 0
                hi_lim = 2 * val

            self.currentCell["val"] = val
            self.currentCell["lowlim"] = low_lim
            self.currentCell["hilim"] = hi_lim
            self.currentCell["readyToChange"] = True

        except (ValueError, AttributeError, KeyError):
            return

        # for some reason linked parameters don't update well when the slider
        # moves. Setting focus to the treeview and back to the slider seems
        # to make the update happen. One could issue more dataChanged signals
        # from sliderValue changed, but I'd have to figure out all the other
        # nodes
        self.ui.treeView.setFocus(Qt.FocusReason.OtherFocusReason)
        self.ui.paramsSlider.setFocus(Qt.FocusReason.OtherFocusReason)

    def link_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()
        par_nodes_to_link = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)
            item = index.internalPointer()
            if isinstance(item, ParNode):
                par_nodes_to_link.append(item)

        # get unique nodes
        par_nodes_to_link = list(unique(par_nodes_to_link))

        if len(par_nodes_to_link) < 2:
            return

        mpn = par_nodes_to_link[0]
        mp = mpn.parameter

        # if the master parameter already has a constraint, then it
        # has to be removed, otherwise recursion occurs
        if mp.constraint is not None:
            mp.constraint = None
            idx = self.treeModel.index(mpn.row(), 1, mpn.parent().index)
            idx1 = self.treeModel.index(mpn.row(), 5, mpn.parent().index)
            self.treeModel.dataChanged.emit(idx, idx1)

        for par in par_nodes_to_link[1:]:
            par.parameter.constraint = mp
            idx = self.treeModel.index(par.row(), 1, par.parent().index)
            idx1 = self.treeModel.index(par.row(), 5, par.parent().index)
            self.treeModel.dataChanged.emit(idx, idx1)

        self.ui.paramsSlider.setFocus(Qt.FocusReason.OtherFocusReason)
        self.ui.treeView.setFocus(Qt.FocusReason.OtherFocusReason)

    def link_equivalent_action(self):
        # link equivalent parameters across a whole range of datasets.
        # the datasets all need to have the same structure for this to work.

        # retrieve data_objects that need to be linked
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle(
            "Select equivalent datasets" " to link"
        )
        ok = self.data_object_selector.exec()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]
        if "theoretical" in names:
            names.pop(names.index("theoretical"))

        # these are the data objects that you want to link across
        data_objects = [datastore[name] for name in names]

        # these are the nodes selected in the tree view. We're going to link
        # these nodes together, to equivalent nodes on the other data objects
        selected_indices = self.ui.treeView.selectedIndexes()
        par_nodes_to_link = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)
            item = index.internalPointer()
            if isinstance(item, ParNode):
                par_nodes_to_link.append(item)

        # get unique nodes
        par_nodes_to_link = list(unique(par_nodes_to_link))

        def is_same_structure(objs):
            # Now check that the models are roughly the same
            models = [data_object.model for data_object in objs]
            structures = [m.structure for m in models]
            ncomponents = [len(s) for s in structures]
            parameters = [list(flatten(m.parameters)) for m in models]
            nparams = [len(p) for p in parameters]

            if len(set(ncomponents)) == 1 and len(set(nparams)) == 1:
                return True
            return False

        extra_pars = []
        # retrieve equivalent parameters on other datasets
        for node in par_nodes_to_link:
            mstr_obj_node = find_data_object(node.index)
            # check that all data_objects have the same structure as the
            # selected parameter
            if not is_same_structure(
                [mstr_obj_node.data_object] + data_objects
            ):
                return msg(
                    "All models must have equivalent structural"
                    " components and the same number of parameters for"
                    " equivalent linking to be available, no linking"
                    " has been done."
                )

            row_indices = node.row_indices()

            for data_object in data_objects:
                # retrieve the similar parameter from row indices
                row_indices[1] = self.treeModel.data_object_row(
                    data_object.name
                )
                # now identify where the similar parameter is
                pn = self.treeModel.node_from_row_indices(row_indices)
                extra_pars.append(pn)

        par_nodes_to_link.extend(extra_pars)
        par_nodes_to_link = list(unique(par_nodes_to_link))

        mpn = par_nodes_to_link[0]
        mp = mpn.parameter

        # if the master parameter already has a constraint, then it
        # has to be removed, otherwise recursion occurs
        if mp.constraint is not None:
            mp.constraint = None
            idx = self.treeModel.index(mpn.row(), 1, mpn.parent().index)
            idx1 = self.treeModel.index(mpn.row(), 5, mpn.parent().index)
            self.treeModel.dataChanged.emit(idx, idx1)

        for par in par_nodes_to_link[1:]:
            par.parameter.constraint = mp
            idx = self.treeModel.index(par.row(), 1, par.parent().index)
            idx1 = self.treeModel.index(par.row(), 5, par.parent().index)
            self.treeModel.dataChanged.emit(idx, idx1)

        self.ui.paramsSlider.setFocus(Qt.FocusReason.OtherFocusReason)
        self.ui.treeView.setFocus(Qt.FocusReason.OtherFocusReason)

    def unlink_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()
        par_nodes_to_unlink = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)
            item = index.internalPointer()
            if isinstance(item, ParNode):
                par_nodes_to_unlink.append(item)

        # get unique nodes
        par_nodes_to_unlink = list(unique(par_nodes_to_unlink))

        for par in par_nodes_to_unlink:
            par.parameter.constraint = None
            idx = self.treeModel.index(par.row(), 2, par.parent().index)
            idx1 = self.treeModel.index(par.row(), 5, par.parent().index)
            self.treeModel.dataChanged.emit(idx, idx1)

        # a trick to make the treeView repaint
        self.ui.paramsSlider.setFocus(Qt.OtherFocusReason)
        self.ui.treeView.setFocus(Qt.OtherFocusReason)

    def copy_from_action(self):
        # whose model did you want to use?
        datastore = self.treeModel.datastore

        model_names = datastore.names
        which_model, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Which model did you want to copy?",
            "model",
            model_names,
            editable=False,
        )

        if not ok:
            return

        selected_indices = self.ui.treeView.selectedIndexes()
        source_model = datastore[which_model].model

        nodes = []
        for index in selected_indices:
            # from filter to model
            index = self.mapToSource(index)
            item = index.internalPointer()
            nodes.append(item.hierarchy())

        # filter out the data_object_nodes
        data_object_nodes = [
            n for n in flatten(nodes) if isinstance(n, DataObjectNode)
        ]

        # get unique data object nodes, these are the data objects whose
        # model you want to overwrite
        data_object_nodes = list(unique(data_object_nodes))

        # now set the model for all those data object nodes
        # _hold_updating is set so that if a lot of things change in the
        # tree structure, the recalculation of reflectivity curves triggered
        # by set_reflect_model doesn't cause an enormous slowdown. It should
        # be sufficient to do a single update at the end.
        self._hold_updating = True
        try:
            for don in data_object_nodes:
                new_model = deepcopy(source_model)
                new_model.name = don.data_object.name
                don.set_reflect_model(new_model)
        finally:
            do = [node.data_object for node in data_object_nodes]

            self._hold_updating = False
            self.update_gui_model(do)

    @QtCore.Slot()
    def add_mixed_area_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return

        index = selected_indices[0]

        # from filter to model
        index = self.mapToSource(index)
        if not index.isValid():
            return

        item = index.internalPointer()
        data_object_node = find_data_object(index)
        reflect_model_node = data_object_node.child(1)
        structures = reflect_model_node.structures

        if isinstance(item, StructureNode):
            copied_structure = deepcopy(item._data)
        else:
            copied_structure = deepcopy(structures[0])

        # add it to the list of structures, at the END
        reflect_model_node.insert_structure(
            len(structures) + 3, copied_structure
        )

    @QtCore.Slot()
    def remove_mixed_area_action(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return

        index = selected_indices[0]

        # from filter to model
        index = self.mapToSource(index)
        if not index.isValid():
            return

        item = index.internalPointer()
        if not isinstance(item, StructureNode):
            return msg("Please select a single Structure to remove")

        data_object_node = find_data_object(index)
        reflect_model_node = data_object_node.child(1)
        if len(reflect_model_node.structures) == 1:
            return msg(
                "Your model only contains a single Structure, removal"
                " not possible"
            )

        reflect_model_node.remove_structure(item.row())

    def remove_graphs_from_gui(self):
        """
        Removes reflectivity and SLD graphs from GUI
        """
        tb = self.sldgraphs.mpl_toolbar
        self.ui.gridLayout_4.removeWidget(tb)
        tb.deleteLater()
        del tb

        self.sldgraphs.mpl_toolbar = None
        self.ui.gridLayout_4.removeWidget(self.sldgraphs)
        self.sldgraphs.deleteLater()
        del self.sldgraphs

        tb = self.reflectivitygraphs.mpl_toolbar
        self.ui.gridLayout_5.removeWidget(tb)
        tb.deleteLater()
        del tb

        self.reflectivitygraphs.mpl_toolbar = None
        self.ui.gridLayout_5.removeWidget(self.reflectivitygraphs)
        self.reflectivitygraphs.deleteLater()
        del self.reflectivitygraphs

    def attach_graphs_to_gui(self):
        """
        Used to add reflectivity and SLD graphs to GUI
        """
        self.sldgraphs = MySLDGraphs(self.ui.sld)
        self.ui.gridLayout_4.addWidget(self.sldgraphs)

        self.reflectivitygraphs = MyReflectivityGraphs(self.ui.reflectivity)
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs)

        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs.mpl_toolbar)
        self.ui.gridLayout_4.addWidget(self.sldgraphs.mpl_toolbar)

    def redraw_data_object_graphs(
        self, data_objects, all=False, transform=True
    ):
        """
        Asks the graphs to be redrawn, delegating generative calculation to
        reflectivitygraphs.redraw_data_objects

        Parameters
        ----------
        data_objects: list
            all the datasets to be redrawn
        all: bool
            redraw all the datasets
        transform: bool or curvefitter.Transform
            Transform the data in the dataset for visualisation.
            False - don't transform
            True - transform the data using the program default
            refnx.objective.Transform - use a custom transform
        """
        datastore = self.treeModel.datastore
        if all:
            data_objects = [data_object for data_object in datastore]

        if callable(transform):
            t = transform
        elif transform:
            t = Transform(self.settings.transformdata)
        else:
            t = None

        self.reflectivitygraphs.redraw_data_objects(data_objects, transform=t)
        self.sldgraphs.redraw_data_objects(data_objects)

    def tree_model_structure_changed(self, parent, first, last):
        # called when you've removed or added layers to the model
        if not parent.isValid():
            return

        node = parent.internalPointer()

        # if you're not adding/removing Components don't respond.
        # if not isinstance(node, StructureNode):
        #     return

        # find out which data_object / model we're adjusting
        hierarchy = node.hierarchy()
        for n in hierarchy:
            if isinstance(n, DataObjectNode):
                self.clear_data_object_uncertainties([n.data_object])
                self.update_gui_model([n.data_object])

    def clear_data_object_uncertainties(self, data_objects):
        if self._hold_updating:
            return

        for data_object in data_objects:
            # retrive data_object node from the treeModel
            node = self.treeModel.data_object_node(data_object.name)
            descendants = node.descendants()
            # filter out all the parameter nodes
            par_nodes = [n for n in descendants if isinstance(n, ParNode)]
            for par in par_nodes:
                p = par.parameter
                p.stderr = None

                parent, row = par.parent(), par.row()
                idx1 = self.treeModel.index(row, 2, parent.index)
                idx2 = self.treeModel.index(row, 2, parent.index)
                self.treeModel.dataChanged.emit(idx1, idx2)

    def tree_model_data_changed(
        self, top_left, bottom_right, role=Qt.ItemDataRole.EditRole
    ):
        # if you've just changed whether you want to hold or vary a parameter
        # there is no need to update the reflectivity plots
        if not top_left.isValid():
            return

        # enumerations of the roles are at:
        # https://doc.qt.io/qt-5/qt.html#ItemDataRole-enum

        # find out which data_object / model we're adjusting
        node = top_left.internalPointer()
        # row = top_left.row()
        col = top_left.column()

        if node is None:
            return

        if (
            len(role)
            and role[0] == Qt.ItemDataRole.CheckStateRole
            and isinstance(node, DataObjectNode)
        ):
            # you're setting the visibility of a data_object
            graph_properties = node.data_object.graph_properties
            graph_properties.visible = node.visible is True
            self.redraw_data_object_graphs([node.data_object])
            return

        # list of data objects to wipe and update
        wipe_update = []

        # redraw if you're altering a PropertyNode (edit or check)
        # also does constant dq/q <--> pointwise
        if (
            col == 1
            and len(role)
            and role[0]
            in [
                Qt.ItemDataRole.CheckStateRole,
                QtCore.Qt.ItemDataRole.EditRole,
            ]
            and isinstance(node, (PropertyNode, ReflectModelNode))
        ):
            wipe_update = [find_data_object(top_left).data_object]

        # only redraw if you're altering values
        # otherwise we'd be performing continual updates of the model
        if (
            col == 1
            and len(role)
            and role[0] == QtCore.Qt.ItemDataRole.EditRole
            and isinstance(node, ParNode)
        ):
            param = node.parameter
            wipe_update = [find_data_object(top_left).data_object]

            # find if there are dependent parameters on the original
            # parameter. There's an argument for doing this in the treeModel,
            # the model is normally responsible for manipulating data. However,
            # if we do it here then we only need to redraw once.
            ds = self.treeModel.datastore
            for do in ds:
                model = do.model
                cpars = model.parameters.constrained_parameters()
                for cpar in cpars:
                    deps = cpar.dependencies()
                    if param in deps:
                        wipe_update.append(do)
                        break

        if wipe_update:
            wipe_update = list(unique(wipe_update))
            self.clear_data_object_uncertainties(wipe_update)
            self.update_gui_model(wipe_update)

    def calculate_chi2(self, data_objects):
        # calculate chi2 for all the data objects
        if not len(data_objects):
            return

        if self._hold_updating:
            return

        for data_object in data_objects:
            if data_object.name == "theoretical":
                continue

            useerrors = self.settings.useerrors

            # this is if the user requested constant dq/q. If that's the case
            # then make a new dataset to hide the resolution information
            dataset_t = data_object.dataset
            if data_object.constantdq_q and dataset_t.x_err is not None:
                dataset_t = Data1D(data=dataset_t)
                dataset_t.x_err = None

            t = None
            if self.settings.transformdata is not None:
                transform = Transform(self.settings.transformdata)
                t = transform

            # can't have negative points if fitting as logR vs Q
            if self.settings.transformdata == "logY":
                dataset_t = self.filter_neg_reflectivity(dataset_t)

            objective = Objective(
                data_object.model,
                dataset_t,
                transform=t,
                use_weights=useerrors,
            )
            chisqr = objective.chisqr()
            node = self.treeModel.data_object_node(data_object.name)
            node.chi2 = chisqr
            index = self.treeModel.index(node.row(), 2, node.parent().index)
            self.treeModel.dataChanged.emit(index, index)

    def update_gui_model(self, data_objects):
        """
        Recalculates chi2 and the generative model for a list of DataObject
        """
        if not len(data_objects):
            return

        if self._hold_updating:
            return

        self.redraw_data_object_graphs(data_objects)
        self.calculate_chi2(data_objects)

    def filter_neg_reflectivity(self, dataset):
        # if one is fitting as logR vs Q (with/without errors), then negative
        # points mess up the transform. Therefore, filter those negative points
        copy = Data1D(data=dataset)
        copy.mask = dataset.y > 0
        return copy

    def add_data_objects_to_graphs(self, data_objects):
        transform = Transform(self.settings.transformdata)
        t = transform

        self.reflectivitygraphs.add_data_objects(data_objects, transform=t)
        self.sldgraphs.add_data_objects(data_objects)


class ProgressCallback(QtWidgets.QDialog):
    def __init__(self, parent=None, objective=None):
        self.start = time.time()
        self.last_time = time.time()
        self.abort_flag = False
        super().__init__(parent)
        self.parent = parent
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, "progress.ui"), self)
        self.elapsed = 0.0
        self.chi2 = 1.0e308
        self.ui.timer.display(float(self.elapsed))
        self.ui.buttonBox.rejected.connect(self.abort)
        self.objective = objective
        self.iterations = 0

    def abort(self):
        self.abort_flag = True

    def callback(self, xk, *args, **kwds):
        # a callback for scipy.optimize.minimize, which enters
        # every iteration.
        new_time = time.time()
        self.iterations += 1

        # update every 1.5 seconds
        if new_time - self.last_time > 1.5:
            # gp = self.dataset.graph_properties
            # if gp.line2Dfit is not None:
            #     self.parent.redraw_data_object_graphs([self.dataset])
            # else:
            #     self.parent.add_datasets_to_graphs([self.dataset])

            self.elapsed = new_time - self.start
            self.ui.timer.display(float(self.elapsed))
            self.last_time = new_time

            text = "Chi2 : {}\nIterations : {}".format(
                self.objective.chisqr(xk), self.iterations
            )

            self.ui.values.setPlainText(text)
            QtWidgets.QApplication.processEvents()
            if self.abort_flag:
                raise StopIteration("WARNING: FIT WAS TERMINATED EARLY", xk)

        return self.abort_flag


class ProgramSettings:
    def __init__(self, **kwds):
        _members = {
            "fitting_algorithm": "DE",
            "transformdata": "logY",
            "quad_order": 17,
            "current_dataset_name": None,
            "experiment_file_name": "",
            "current_model_name": None,
            "usedq": True,
            "resolution": 5,
            "fit_plugin": None,
            "useerrors": True,
        }

        for key in _members:
            if key in kwds:
                setattr(self, key, kwds[key])
            else:
                setattr(self, key, _members[key])

    def __getitem__(self, key):
        return self.__dict__[key]

    def __getattr_(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if key in self.__dict__:
            setattr(self, key, value)


class NavToolBar(NavigationToolbar):
    """
    This class overloads the NavigationToolbar of matplotlib.
    """

    def __init__(self, canvas, parent, coordinates=True):
        NavigationToolbar.__init__(self, canvas, parent, coordinates)
        self.setIconSize(QtCore.QSize(20, 20))


class MyReflectivityGraphs(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1, 1, 1), edgecolor=(0, 0, 0))

        # reflectivity graph
        self.axes = []
        ax = self.figure.add_axes([0.06, 0.15, 0.9, 0.8])
        ax.margins(0.0005)
        self.axes.append(ax)

        self.axes[0].autoscale(axis="both", tight=False, enable=True)
        self.axes[0].set_xlabel(r"Q / $\AA^{-1}$")
        self.axes[0].set_ylabel("R")
        # self.axes[0].set_yscale('log')

        # residual plot
        # , sharex=self.axes[0]
        # ax2 = self.figure.add_axes([0.1,0.04,0.85,0.14],
        #                            sharex=ax, frame_on = False)
        #   self.axes.append(ax2)
        #   self.axes[1].set_visible(True)
        #   self.axes[1].set_ylabel('residual')

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self.mpl_toolbar = NavToolBar(self, parent)
        self.figure.canvas.mpl_connect("pick_event", self._pick_event)
        # self.figure.canvas.mpl_connect('key_press_event', self._key_press)

        self.draw()

    def _key_press(self, event):
        # auto scale
        if event.key == "super+a":
            self.autoscale()

    def _pick_event(self, event):
        # pick event was a double click on the graph
        if event.mouseevent.dblclick and event.mouseevent.button == 1:
            if isinstance(event.artist, lines.Line2D):
                self.mpl_toolbar.edit_parameters()

    def autoscale(self):
        self.axes[0].relim()
        self.axes[0].autoscale(axis="both", tight=False, enable=True)
        self.draw()

    def add_data_objects(self, data_objects, transform=None):
        for data_object in data_objects:
            dataset = data_object.dataset

            graph_properties = data_object.graph_properties

            if (
                graph_properties.ax_data is None
                and data_object.name != "theoretical"
            ):
                yt = dataset.y
                if transform is not None:
                    yt, edata = transform(dataset.x, dataset.y, dataset.y_err)

                # add the dataset
                error_bar_container = self.axes[0].errorbar(
                    dataset.x,
                    yt,
                    yerr=edata,
                    markersize=3,
                    marker="o",
                    linestyle="",
                    markeredgecolor=None,
                    label=dataset.name,
                )
                line_instance = error_bar_container[0]
                line_instance.set_pickradius(5)
                mfc = artist.getp(line_instance, "markerfacecolor")
                artist.setp(line_instance, **{"markeredgecolor": mfc})

                graph_properties["ax_data"] = error_bar_container
                data_properties = graph_properties["data_properties"].copy()
                if data_properties:
                    artist.setp(error_bar_container[0], **data_properties)
                    artist.setp(
                        error_bar_container[-1], color=data_properties["color"]
                    )

            yfit_t = data_object.generative
            if graph_properties.ax_fit is None and yfit_t is not None:
                if transform is not None:
                    yfit_t, temp = transform(dataset.x, yfit_t)

                color = "b"
                if graph_properties.ax_data is not None:
                    color = artist.getp(graph_properties.ax_data[0], "color")
                # add the fit
                line = self.axes[0].plot(
                    dataset.x,
                    yfit_t,
                    linestyle="-",
                    color=color,
                    lw=1,
                    label="fit_" + data_object.name,
                )[0]
                line.set_pickradius(5.0)
                graph_properties["ax_fit"] = line

                if graph_properties["fit_properties"]:
                    artist.setp(
                        graph_properties.ax_fit,
                        **graph_properties["fit_properties"],
                    )

            # if (dataObject.line2Dresiduals is None and
            #     dataObject.residuals is not None):
            #     dataObject.line2Dresiduals = self.axes[1].plot(
            #         dataObject.x,
            #         dataObject.residuals,
            #         linestyle='-',
            #         lw = 1,
            #         label = 'residuals_' + dataObject.name)[0]
            #
            #     if graph_properties['residuals_properties']:
            #         artist.setp(dataObject.ax_residuals,
            #                     **graph_properties['residuals_properties'])

            graph_properties.save_graph_properties()
        self.draw()

    def redraw_data_objects(self, data_objects, transform=None):
        if not len(data_objects):
            return

        for data_object in data_objects:
            if not data_object:
                continue

            dataset = data_object.dataset

            if data_object.name != "theoretical":
                y = dataset.y
                e = dataset.y_err
                if transform is not None:
                    y, e = transform(dataset.x, y, e)

            if data_object.model is not None:
                yfit = data_object.generative

                if transform is not None:
                    yfit, efit = transform(dataset.x, yfit)

            graph_properties = data_object.graph_properties
            visible = graph_properties.visible

            if graph_properties.ax_data is not None:
                # ax_data is an ErrorbarContainer, so set everything
                ebc = graph_properties.ax_data
                errorbar_set_data(ebc, dataset.x, y, e)
                for line in flatten(ebc.lines):
                    line.set_visible(visible)
            if graph_properties.ax_fit is not None:
                graph_properties.ax_fit.set_data(dataset.x, yfit)
                graph_properties.ax_fit.set_visible(visible)

        #          if dataObject.line2Dresiduals:
        #             dataObject.line2Dresiduals.set_data(dataObject.x,
        #                                          dataObject.residuals)
        #             dataObject.line2Dresiduals.set_visible(visible)

        self.draw()

    def remove_trace(self, data_object):
        graph_properties = data_object.graph_properties
        if graph_properties.ax_data is not None:
            graph_properties.ax_data.remove()
            graph_properties.ax_data = None

        if graph_properties.ax_fit is not None:
            graph_properties.ax_fit.remove()
            graph_properties.ax_fit = None

        if graph_properties.ax_residuals is not None:
            graph_properties.ax_residuals.remove()
            graph_properties.ax_residuals = None
        self.draw()


class MySLDGraphs(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = []
        # SLD plot
        self.axes.append(self.figure.add_subplot(111))

        self.axes[0].autoscale(axis="both", tight=False, enable=True)
        self.axes[0].set_xlabel("z")
        self.axes[0].set_ylabel("SLD")

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.98)
        self.mpl_toolbar = NavigationToolbar(self, parent)

    def redraw_data_objects(self, data_objects):
        for data_object in data_objects:
            if not data_object:
                continue
            graph_properties = data_object.graph_properties
            visible = graph_properties.visible

            if (
                graph_properties.ax_sld_profile
                and data_object.model is not None
            ):
                try:
                    if isinstance(data_object.model, MixedReflectModel):
                        sld_profile = data_object.model.structures[
                            0
                        ].sld_profile(max_delta_z=1.0)
                    else:
                        sld_profile = data_object.model.structure.sld_profile(
                            max_delta_z=1.0
                        )

                    graph_properties.ax_sld_profile.set_data(
                        sld_profile[0], sld_profile[1]
                    )
                    graph_properties.ax_sld_profile.set_visible(visible)
                except AttributeError:
                    # TODO, fix this
                    # this may happen for MixedReflectModel, the model doesnt
                    # have structure.sld_profile()
                    continue

        self.axes[0].relim()
        self.axes[0].autoscale_view(None, True, True)
        self.draw()

    def add_data_objects(self, data_objects):
        for data_object in data_objects:
            graph_properties = data_object.graph_properties
            if (
                graph_properties.ax_sld_profile is None
                and data_object.sld_profile is not None
            ):
                color = "r"
                lw = 2
                if graph_properties.ax_data:
                    # ax_data is an ErrorbarContainer
                    color = artist.getp(graph_properties.ax_data[0], "color")
                    lw = artist.getp(graph_properties.ax_data[0], "lw")

                try:
                    graph_properties["ax_sld_profile"] = self.axes[0].plot(
                        data_object.sld_profile[0],
                        data_object.sld_profile[1],
                        linestyle="-",
                        color=color,
                        lw=lw,
                        label="sld_" + data_object.name,
                    )[0]
                except AttributeError:
                    # this may happen for MixedReflectModel, the model doesnt
                    # have structure.sld_profile()
                    continue

                if graph_properties["sld_profile_properties"]:
                    artist.setp(
                        graph_properties.ax_sld_profile,
                        **graph_properties["sld_profile_properties"],
                    )

        self.axes[0].relim()
        self.axes[0].autoscale(axis="both", tight=False, enable=True)
        self.draw()

    def remove_trace(self, data_object):
        if data_object.graph_properties.ax_sld_profile:
            data_object.graph_properties.ax_sld_profile.remove()
        self.draw()


class EmittingStream(QtCore.QObject):
    # a class for rewriting stdout to a console window

    textWritten = QtCore.Signal(str)

    def __init__(self):
        QtCore.QObject.__init__(self)
        # super(EmittingStream, self).__init__()

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class OpenMenu(QtWidgets.QMenu):
    """
    A context menu class for the model tree view
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.copy_from_action = self.addAction("Copy a model to here")
        self.addSeparator()
        self.add_to_fit_action = self.addAction("Add to fit")
        self.remove_from_fit_action = self.addAction("Remove from fit")
        self.addSeparator()
        self.link_action = self.addAction("Link parameters")
        self.unlink_action = self.addAction("Unlink parameters")
        self.link_equivalent_action = self.addAction(
            "Link equivalent parameters on other datasets"
        )
        self.addSeparator()
        self.add_mixed_area = self.addAction("Mixed area - add a structure")
        self.remove_mixed_area = self.addAction(
            "Mixed area - remove a" " structure"
        )

    def __call__(self, position):
        action = self.exec(self._parent.mapToGlobal(position))
        if action == self.link_action:
            pass
        if action == self.unlink_action:
            pass


def msg(text):
    # utility function for displaying a message
    msgBox = QtWidgets.QMessageBox()
    msgBox.setText(text)
    return msgBox.exec()


def _default_slab(parent=None):
    # a default slab to add to a model
    material = SLD(3.47)
    c = material(15, 3)
    c.name = "slab"
    c.thick.name = "thick"
    c.rough.name = "rough"
    c.sld.real.name = "sld"
    c.sld.imag.name = "isld"
    c.vfsolv.name = "vfsolv"
    return c


class DataObjectSelectorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        # persistent data object selector dlg
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi(
            os.path.join(UI_LOCATION, "data_object_selector.ui"), self
        )

    def addItems(self, items):
        self.data_objects.addItems(items)

    def removeItem(self, item):
        list_items = self.data_objects.findItems(item, QtCore.Qt.MatchExactly)
        if list_items:
            row = self.data_objects.row(list_items[0])
            self.data_objects.takeItem(row)


class CurrentlyFitting(QtCore.QAbstractListModel):
    """
    Keeps a list of the datasets that are currently being fitted
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.datasets = []

    def rowCount(self, index):
        return len(self.datasets)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            row = index.row()
            return self.datasets[row]

    def addItems(self, items):
        new_items = [
            i
            for i in items
            if (i not in self.datasets) and (i != "theoretical")
        ]

        n_current = len(self.datasets)
        if new_items:
            self.beginInsertRows(
                QtCore.QModelIndex(), n_current, n_current + len(new_items) + 1
            )
            self.datasets.extend(new_items)
            self.endInsertRows()

    def removeItems(self, indices):
        # remove by number
        indices.sort()
        indices.reverse()

        for i in indices:
            self.beginRemoveRows(QtCore.QModelIndex(), i, i)
            self.datasets.pop(i)
            self.endRemoveRows()

    def flags(self, column):
        flags = (
            QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsEnabled
        )

        # say that you want the Components to be draggable
        flags |= QtCore.Qt.ItemFlag.ItemIsDropEnabled

        return flags

    def supportedDropActions(self):
        return QtCore.Qt.MoveAction

    def mimeTypes(self):
        return ["application/vnd.treeviewdragdrop.list"]

    def dropMimeData(self, data, action, row, column, parent):
        # drop datasets from the treeview
        if action == QtCore.Qt.IgnoreAction:
            return True
        if not data.hasFormat("application/vnd.treeviewdragdrop.list"):
            return False
        # what was dropped?
        ba = data.data("application/vnd.treeviewdragdrop.list")
        index_info = pickle.loads(ba)

        to_add = []
        for i in index_info:
            if i["name"] is not None:
                to_add.append(i["name"])

        if to_add:
            self.addItems(unique(to_add))
            return True

        return False


def errorbar_set_data(errobj, x, y, yerr=None, xerr=None):
    """
    set_data for an errorbar plot

    Parameters
    ----------
    errobj : ErrorbarContainer
    x : array-like
    y : array-like
    yerr : array-like
    xerr : array-like
    """
    ln, caps, bars = errobj

    x_base = x
    y_base = y

    ln.set_data(x, y)

    if xerr is None:
        xerr = np.zeros_like(x)

    if yerr is None:
        yerr = np.zeros_like(y)

    xerr_top = x_base + xerr
    xerr_bot = x_base - xerr
    yerr_top = y_base + yerr
    yerr_bot = y_base - yerr

    if caps:
        errx_top, errx_bot, erry_top, erry_bot = caps

        errx_top.set_xdata(xerr_top)
        errx_bot.set_xdata(xerr_bot)
        errx_top.set_ydata(y_base)
        errx_bot.set_ydata(y_base)

        erry_top.set_xdata(x_base)
        erry_bot.set_xdata(x_base)
        erry_top.set_ydata(yerr_top)
        erry_bot.set_ydata(yerr_bot)

    if bars:
        new_segments_y = [
            np.array([[x, yt], [x, yb]])
            for x, yt, yb in zip(x_base, yerr_top, yerr_bot)
        ]
        new_segments_x = [
            np.array([[xt, y], [xb, y]])
            for xt, xb, y in zip(xerr_top, xerr_bot, y_base)
        ]

        bars[-1].set_segments(new_segments_y)
        if len(bars) > 1:
            bars[0].set_segments(new_segments_x)
