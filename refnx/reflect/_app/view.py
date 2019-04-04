import os.path
from copy import deepcopy
import pickle
import os
import sys
import time
import csv

import numpy as np
import matplotlib

from PyQt5 import QtCore, QtGui, QtWidgets, uic

# matplotlib.rcParams['backend.qt5'] = 'PyQt5'
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
import matplotlib.artist as artist
import matplotlib.lines as lines


from .SLD_calculator_view import SLDcalculatorView
from .datastore import DataStore
from .treeview_gui_model import (TreeModel, Node, DatasetNode, DataObjectNode,
                                 ComponentNode, StructureNode,
                                 ReflectModelNode, ParNode, TreeFilter,
                                 find_data_object, SlabNode, StackNode)
from ._lipid_leaflet import LipidLeafletDialog
from ._spline import SplineDialog

import refnx
from refnx.analysis import (CurveFitter, Objective,
                            Transform, GlobalObjective)
from refnx.reflect import SLD, ReflectModel, Slab, Stack, Structure
from refnx.dataset import Data1D
from refnx.reflect._code_fragment import code_fragment
from refnx._lib import unique, flatten, MapWrapper


# matplotlib.use('Qt5Agg')
UI_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'ui')


class MotofitMainWindow(QtWidgets.QMainWindow):
    """
    Main View window for Motofit
    """

    def __init__(self, parent=None):
        super(MotofitMainWindow, self).__init__(parent)

        # load the GUI from the ui file
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, 'motofit.ui'),
                             self)

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

        self.ui.treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        # context menu for the treeView
        self.context_menu = OpenMenu(self.ui.treeView)

        self.ui.treeView.customContextMenuRequested.connect(self.context_menu)
        self.context_menu.add_to_fit_action.triggered.connect(
            self.on_add_to_fit_button_clicked)
        self.context_menu.remove_from_fit_action.triggered.connect(
            self.on_remove_from_fit_action)
        self.context_menu.link_action.triggered.connect(
            self.link_action)
        self.context_menu.link_equivalent_action.triggered.connect(
            self.link_equivalent_action)
        self.context_menu.unlink_action.triggered.connect(
            self.unlink_action)
        self.context_menu.copy_from_action.triggered.connect(
            self.copy_from_action)
        self.context_menu.add_mixed_area.triggered.connect(
            self.add_mixed_area_action)
        self.context_menu.remove_mixed_area.triggered.connect(
            self.remove_mixed_area_action)
        self.actionLink_Selected.triggered.connect(self.link_action)
        self.actionUnlink_selected_Parameters.triggered.connect(
            self.unlink_action)
        self.actionLink_Equivalent_Parameters.triggered.connect(
            self.link_equivalent_action)

        self.treeModel.dataChanged.connect(self.tree_model_data_changed)
        self.treeModel.rowsRemoved.connect(self.tree_model_structure_changed)
        self.treeModel.rowsMoved.connect(self.tree_model_structure_changed)
        self.treeModel.rowsInserted.connect(self.tree_model_structure_changed)

        # list view for datasets being fitted
        self.currently_fitting_model = CurrentlyFitting(self)
        self.ui.currently_fitting.setModel(self.currently_fitting_model)
        #######################################################################

        # attach the reflectivity graphs and the SLD profiles
        self.modify_gui()

        # holds miscellaneous information on program settings
        self.settings = ProgramSettings()
        self.settings.current_dataset_name = 'theoretical'

        theoretical = data_container['theoretical']
        model = theoretical.model
        dataset = theoretical.dataset
        resolution = self.settings.resolution / 100.
        transform = Transform(self.settings.transformdata)
        fit = model(dataset.x, x_err=dataset.x * resolution)
        fit, _ = transform(dataset.x, fit)
        sld = model.structure.sld_profile()

        graph_properties = theoretical.graph_properties
        graph_properties['ax_fit'] = self.reflectivitygraphs.axes[0].plot(
            dataset.x,
            fit, color='r',
            linestyle='-', lw=1,
            label='theoretical',
            picker=5)[0]

        graph_properties['ax_sld_profile'] = self.sldgraphs.axes[0].plot(
            sld[0],
            sld[1],
            linestyle='-', color='r')[0]

        self.restore_settings()

        self.spline_dialog = SplineDialog(self)
        self.lipid_leaflet = LipidLeafletDialog(self)
        self.data_object_selector = DataObjectSelectorDialog(self)

        self.ui.treeView.setColumnWidth(0, 200)
        h = self.ui.treeView.header()
        h.setMinimumSectionSize(100)
        # h.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)

        # redirect stdout to a console window
        console = EmittingStream()
        sys.stdout = console
        console.textWritten.connect(self.writeTextToConsole)

        print('Session started at:', time.asctime(time.localtime(time.time())))

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    @QtCore.pyqtSlot(QtGui.QDropEvent)
    def dropEvent(self, event):
        m = event.mimeData()
        urls = m.urls()

        # convert url's to files
        urls_as_files = [url.toLocalFile() for url in urls]

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
        except Exception:
            loaded_names = []

        # then try and load the remainder as models
        remainder_urls = set(urls_as_files).difference(set(loaded_names))
        for url in remainder_urls:
            try:
                self.load_model(url)
                continue
            except Exception:
                pass

    @QtCore.pyqtSlot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event):
        m = event.mimeData()
        if m.hasUrls():
            event.acceptProposedAction()

    def writeTextToConsole(self, text):
        self.ui.console_text_edit.moveCursor(QtGui.QTextCursor.End)
        self.ui.console_text_edit.insertPlainText(text)

    def _saveState(self, experiment_file_name):
        state = {}
        self.settings.experiment_file_name = experiment_file_name
        state['datastore'] = self.treeModel.datastore
        state['history'] = self.ui.console_text_edit.toPlainText()
        state['settings'] = self.settings

        fit_list = self.currently_fitting_model
        state['currently_fitting'] = fit_list.datasets

        with open(os.path.join(experiment_file_name), 'wb') as f:
            pickle.dump(state, f, -1)

        self.setWindowTitle('Motofit - ' + experiment_file_name)

    @QtCore.pyqtSlot()
    def on_actionSave_File_triggered(self):
        if os.path.isfile(self.settings.experiment_file_name):
            self._saveState(self.settings.experiment_file_name)
        else:
            self.on_actionSave_File_As_triggered()

    @QtCore.pyqtSlot()
    def on_actionSave_File_As_triggered(self):
        experiment_file_name, ok = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save experiment as:', 'experiment.mtft')

        if not ok:
            return

        path, ext = os.path.splitext(experiment_file_name)
        if ext != '.mtft':
            experiment_file_name = path + '.mtft'

        self._saveState(experiment_file_name)

    def _restore_state(self, experiment_file_name):
        with open(experiment_file_name, 'rb') as f:
            state = pickle.load(f)

        if not state:
            print("Couldn't load experiment")
            return

        self.treeModel._data = state[
            'datastore']
        self.treeModel.rebuild()

        # remove and add datasetsToGraphs
        self.reflectivitygraphs.remove_traces()
        self.sldgraphs.remove_traces()
        ds = [d for d in self.treeModel.datastore]
        self.add_data_objects_to_graphs(ds)
        self.update_gui_model(ds)
        self.reflectivitygraphs.draw()

        try:
            self.ui.console_text_edit.setPlainText(state['history'])
            self.settings = state['settings']
            self.settings.experiment_file_name = experiment_file_name
            self.restore_settings()

            while self.data_object_selector.data_objects.count():
                self.data_object_selector.data_objects.takeItem(0)

            self.data_object_selector.addItems(self.treeModel.datastore.names)

            self.currently_fitting_model = CurrentlyFitting(self)
            fit_list = self.currently_fitting_model
            self.ui.currently_fitting.setModel(fit_list)
            fit_list.addItems(state['currently_fitting'])

        except KeyError as e:
            print(repr(e))
            return

    def restore_settings(self):
        """
        applies the program settings to the GUI
        """
        title = 'Motofit'
        if len(self.settings.experiment_file_name):
            title += ' - ' + self.settings.experiment_file_name
        self.setWindowTitle(title)

        self.ui.use_errors_checkbox.setChecked(self.settings.useerrors)
        self.ui.actionLevenberg_Marquardt.setChecked(False)
        self.ui.actionDifferential_Evolution.setChecked(False)
        if self.settings.fitting_algorithm == 'LM':
            self.ui.actionLevenberg_Marquardt.setChecked(True)
        elif self.settings.fitting_algorithm == 'DE':
            self.ui.actionDifferential_Evolution.setChecked(True)
        elif self.settings.fitting_algorithm == 'MCMC':
            self.ui.actionMCMC.setChecked(True)
        elif self.settings.fitting_algorithm == 'L-BFGS-B':
            self.ui.actionL_BFGS_B.setChecked(True)

        self.settransformoption(self.settings.transformdata)

    def apply_settings_to_params(self, params):
        for key in self.settings.__dict__:
            params[key] = self.settings[key]

    @QtCore.pyqtSlot()
    def on_actionLoad_File_triggered(self):
        experimentFileName, ok = QtWidgets.QFileDialog.getOpenFileName(
            self,
            caption='Select Experiment File',
            filter='Experiment Files (*.mtft)')
        if not ok:
            return

        self._restore_state(experimentFileName)

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

        loaded_data_objects = [data_object.name for data_object
                               in data_objects]
        new_names = [n for n in loaded_data_objects if
                     n not in existing_data_objects]

        # give a newly loaded data object a simple model. You don't want to do
        # this for an object that's already been loaded.
        for name in new_names:
            fronting = SLD(0, name='fronting')
            sio2 = SLD(3.47, name='1')
            backing = SLD(2.07, name='backing')
            s = fronting() | sio2(15, 3) | backing(0, 3)
            data_object_node = self.treeModel.data_object_node(name)
            model = ReflectModel(s)
            model.name = name
            data_object_node.set_reflect_model(model)

        # for the intersection of loaded and old, refresh the plot.
        refresh_names = [n for n in existing_data_objects if
                         n in loaded_data_objects]

        refresh_data_objects = [datastore[name] for name
                                in refresh_names]
        self.redraw_data_object_graphs(refresh_data_objects)

        # for totally new, then add to graphs
        new_data_objects = [datastore[name] for name in new_names]
        self.add_data_objects_to_graphs(new_data_objects)

        self.calculate_chi2(data_objects)

        # add newly loads to the data object selector dialogue
        self.data_object_selector.addItems(new_names)
        return fnames

    @QtCore.pyqtSlot()
    def on_actionLoad_Data_triggered(self):
        """
        you load data
        """
        files = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            caption='Select Reflectivity Files')

        if files:
            self.load_data(files[0])

    @QtCore.pyqtSlot()
    def on_actionRemove_Data_triggered(self):
        """
        you remove data
        """
        # retrieve data_objects that need to be removed
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle('Select datasets to remove')
        ok = self.data_object_selector.exec_()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]
        if 'theoretical' in names:
            names.pop(names.index('theoretical'))

        fit_list = self.currently_fitting_model

        for which_dataset in names:
            # remove from list of datasets to be fitted, if present
            if which_dataset in fit_list.datasets:
                fit_list.removeItems([fit_list.datasets.index(which_dataset)])

            self.reflectivitygraphs.remove_trace(
                datastore[which_dataset])
            self.sldgraphs.remove_trace(datastore[which_dataset])
            self.treeModel.remove_data_object(which_dataset)

            # remove from data object selector
            self.data_object_selector.removeItem(which_dataset)

    @QtCore.pyqtSlot()
    def on_actionSave_Fit_triggered(self):
        datastore = self.treeModel.datastore
        fits = datastore.names
        fits.append('-all-')

        which_fit, ok = QtWidgets.QInputDialog.getItem(
            self, "Which fit did you want to save?", "", fits, editable=False)
        if not ok:
            return

        if which_fit == '-all-':
            dialog = QtWidgets.QFileDialog(self)
            dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            if dialog.exec_():
                folder = dialog.selectedFiles()
                fits.pop()
                for fit in fits:
                    datastore[fit].save_fit(
                        os.path.join(folder[0], 'fit_' + fit + '.dat'))
        else:
            fitFileName, ok = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save fit as:', 'fit_' + which_fit + '.dat')
            if not ok:
                return
            datastore[which_fit].save_fit(fitFileName)

    def load_model(self, model_file_name):
        with open(model_file_name, 'rb') as f:
            model = pickle.load(f)

        data_object_node = self.treeModel.data_object_node(model.name)
        if data_object_node is not None:
            data_object_node.set_reflect_model(model)
        else:
            # there is not dataset with that name, put the model over the
            # theoretical one
            data_object_node = self.treeModel.data_object_node('theoretical')
            data_object_node.set_reflect_model(model)
        self.update_gui_model([data_object_node.data_object])

    @QtCore.pyqtSlot()
    def on_actionLoad_Model_triggered(self):
        # load a model from a pickle file
        model_file_name, ok = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Model File')
        if not ok:
            return
        self.load_model(model_file_name)

    @QtCore.pyqtSlot()
    def on_actionSave_Model_triggered(self):
        # save a model to a pickle file
        # which model are you saving?
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle('Select models to save')
        ok = self.data_object_selector.exec_()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]
        if 'theoretical' in names:
            names.pop(names.index('theoretical'))

        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setWindowTitle('Where do you want to save the models?')
        if dialog.exec_():
            folder = dialog.selectedFiles()

            for model_name in names:
                model = datastore[model_name]
                fname = os.path.join(folder[0],
                                     'coef_' + model_name + '.pkl')
                with open(fname, 'wb') as f:
                    pickle.dump(model, f)

    @QtCore.pyqtSlot()
    def on_actionExport_parameters_triggered(self):
        # save all parameter values to a text file
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle('Select parameters to export')
        ok = self.data_object_selector.exec_()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]

        suggested_name = os.path.join(os.getcwd(),
                                      'coefficients.csv')
        fname, ok = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Exported file name:',
            suggested_name)
        if not ok:
            return

        with open(fname, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for name in names:
                model = datastore[name].model
                writer.writerow([name])
                writer.writerow([p.name for p in flatten(model.parameters)])
                writer.writerow([p.value for p in flatten(model.parameters)])
                writer.writerow([p.stderr for p in flatten(model.parameters)])

    @QtCore.pyqtSlot()
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

        suggested_name = os.path.join(os.getcwd(), 'mcmc.py')
        modelFileName, ok = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save code fragment as:',
            suggested_name)
        if not ok:
            return
        with open(suggested_name, 'w') as f:
            f.write(code)

    @QtCore.pyqtSlot()
    def on_actionDifferential_Evolution_triggered(self):
        if self.ui.actionDifferential_Evolution.isChecked():
            self.settings.fitting_algorithm = 'DE'
            self.ui.actionLevenberg_Marquardt.setChecked(False)
            self.ui.actionMCMC.setChecked(False)
            self.ui.actionL_BFGS_B.setChecked(False)

    @QtCore.pyqtSlot()
    def on_actionMCMC_triggered(self):
        if self.ui.actionMCMC.isChecked():
            self.settings.fitting_algorithm = 'MCMC'
            self.ui.actionLevenberg_Marquardt.setChecked(False)
            self.ui.actionDifferential_Evolution.setChecked(False)
            self.ui.actionL_BFGS_B.setChecked(False)

    @QtCore.pyqtSlot()
    def on_actionLevenberg_Marquardt_triggered(self):
        if self.ui.actionLevenberg_Marquardt.isChecked():
            self.settings.fitting_algorithm = 'LM'
            self.ui.actionDifferential_Evolution.setChecked(False)
            self.ui.actionMCMC.setChecked(False)
            self.ui.actionL_BFGS_B.setChecked(False)

    @QtCore.pyqtSlot()
    def on_actionL_BFGS_B_triggered(self):
        if self.ui.actionL_BFGS_B.isChecked():
            self.settings.fitting_algorithm = 'L-BFGS-B'
            self.ui.actionDifferential_Evolution.setChecked(False)
            self.ui.actionMCMC.setChecked(False)
            self.ui.actionLevenberg_Marquardt.setChecked(False)

    def change_Q_range(self, qmin, qmax, numpnts):
        data_object_node = self.treeModel.data_object_node('theoretical')

        theoretical = data_object_node._data
        dataset = theoretical.dataset

        new_x = np.linspace(qmin, qmax, numpnts)
        new_y = np.zeros_like(new_x) * np.nan
        dataset.data = (new_x, new_y)
        data_object_node.set_dataset(dataset)
        self.update_gui_model([theoretical])

    @QtCore.pyqtSlot()
    def on_actionChange_Q_range_triggered(self):
        datastore = self.treeModel.datastore
        theoretical = datastore['theoretical']
        qmin = min(theoretical.dataset.x)
        qmax = max(theoretical.dataset.x)
        numpnts = len(theoretical.dataset)

        dvalidator = QtGui.QDoubleValidator(-2.0e-308, 2.0e308, 6)

        qrangeGUI = uic.loadUi(os.path.join(UI_LOCATION, 'qrangedialog.ui'))
        qrangeGUI.numpnts.setValue(numpnts)
        qrangeGUI.qmin.setValidator(dvalidator)
        qrangeGUI.qmax.setValidator(dvalidator)
        qrangeGUI.qmin.setText(str(qmin))
        qrangeGUI.qmax.setText(str(qmax))

        ok = qrangeGUI.exec_()
        if ok:
            self.change_Q_range(float(qrangeGUI.qmin.text()),
                                float(qrangeGUI.qmax.text()),
                                qrangeGUI.numpnts.value())

    @QtCore.pyqtSlot()
    def on_actionTake_Snapshot_triggered(self):
        snapshotname, ok = QtWidgets.QInputDialog.getText(
            self,
            'Take a snapshot',
            'snapshot name')
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

    @QtCore.pyqtSlot()
    def on_actionResolution_smearing_triggered(self):
        currentVal = self.settings.quad_order
        value, ok = QtWidgets.QInputDialog.getInt(
            self,
            'Resolution Smearing',
            'Number of points for Gaussian Quadrature',
            currentVal,
            17)
        if not ok:
            return
        self.settings.quad_order = value
        self.update_gui_model([])

    @QtCore.pyqtSlot()
    def on_actionBatch_Fit_triggered(self):
        datastore = self.treeModel.datastore
        if len(datastore) < 2:
            return msg("You have no loaded datasets")

        theoretical = datastore['theoretical']
        # iterate and fit over all the datasets, but first copy the model from
        # the theoretical model because it's unlikely you're going to setup all
        # the individual models first.
        for data_object in datastore:
            if data_object.name == 'theoretical':
                continue
            name = data_object.name
            new_model = deepcopy(theoretical.model)
            new_model.name = name
            data_object_node = self.treeModel.data_object_node(name)
            data_object_node.set_reflect_model(new_model)
            self.do_a_fit_and_add_to_gui([data_object])

    @QtCore.pyqtSlot()
    def on_actionRefresh_Data_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.treeModel.refresh()
        self.redraw_data_object_graphs(None, all=True)

    @QtCore.pyqtSlot()
    def on_actionlogY_vs_X_triggered(self):
        self.settransformoption('logY')

    @QtCore.pyqtSlot()
    def on_actionY_vs_X_triggered(self):
        self.settransformoption('lin')

    @QtCore.pyqtSlot()
    def on_actionYX4_vs_X_triggered(self):
        self.settransformoption('YX4')

    @QtCore.pyqtSlot()
    def on_actionYX2_vs_X_triggered(self):
        self.settransformoption('YX2')

    def settransformoption(self, transform):
        self.ui.actionlogY_vs_X.setChecked(False)
        self.ui.actionY_vs_X.setChecked(False)
        self.ui.actionYX4_vs_X.setChecked(False)
        self.ui.actionYX2_vs_X.setChecked(False)
        if transform is None:
            self.ui.actionY_vs_X.setChecked(True)
            transform = 'lin'
        if transform == 'lin':
            self.ui.actionY_vs_X.setChecked(True)
        elif transform == 'logY':
            self.ui.actionlogY_vs_X.setChecked(True)
        elif transform == 'YX4':
            self.ui.actionYX4_vs_X.setChecked(True)
        elif transform == 'YX2':
            self.ui.actionYX2_vs_X.setChecked(True)
        self.settings.transformdata = transform

        self.redraw_data_object_graphs(None, all=True)

        # need to relimit graphs and display on a log scale if the transform
        # has changed
        self.reflectivitygraphs.axes[0].autoscale(axis='both',
                                                  tight=False,
                                                  enable=True)
        self.reflectivitygraphs.axes[0].relim()
        if transform in ['lin', 'YX2']:
            self.reflectivitygraphs.axes[0].set_yscale('log')
        else:
            self.reflectivitygraphs.axes[0].set_yscale('linear')
        self.reflectivitygraphs.draw()

    @QtCore.pyqtSlot()
    def on_actionAbout_triggered(self):
        aboutui = uic.loadUi(os.path.join(UI_LOCATION, 'about.ui'))

        licence_dir = os.path.join(UI_LOCATION, 'licences')
        licences = os.listdir(licence_dir)
        licences.remove('about')

        text = [refnx.version.version]
        with open(os.path.join(licence_dir, 'about'), 'r',
                  encoding='utf-8', errors='replace') as f:
            text.append(''.join(f.readlines()))

        for licence in licences:
            fname = os.path.join(licence_dir, licence)
            with open(fname, 'r', encoding='utf-8', errors='replace') as f:
                text.append(''.join(f.readlines()))

        display_text = '\n_______________________________________\n'.join(text)
        aboutui.textBrowser.setText(display_text)
        aboutui.exec_()

    @QtCore.pyqtSlot()
    def on_actiondocumentation_triggered(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(
            'https://refnx.readthedocs.io/en/latest/'))

    @QtCore.pyqtSlot()
    def on_actionAutoscale_graph_triggered(self):
        self.reflectivitygraphs.autoscale()

    @QtCore.pyqtSlot()
    def on_actionSLD_calculator_triggered(self):
        SLDcalculator = SLDcalculatorView(self)
        SLDcalculator.show()

    @QtCore.pyqtSlot()
    def on_actionLipid_browser_triggered(self):
        self.lipid_leaflet.show()

    @QtCore.pyqtSlot()
    def on_add_layer_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return msg('Select a single row within a Structure to insert a'
                       ' new Component.')

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
        _component = [i for i in hierarchy if (isinstance(i, ComponentNode) or
                                               isinstance(i, StackNode))]
        if not _component:
            return msg('Select a single location within a Structure to insert'
                       ' a new Component.')

        # work out which component you have.
        component = _component[0]
        host = component.parent()
        idx = component.row()
        if isinstance(host, StructureNode) and idx == len(host._data) - 1:
            return msg("You can't append a layer after the backing medium,"
                       " select a previous layer")

        # what type of component shall we add?
        comp_type = ['Slab', 'LipidLeaflet', 'Spline', 'Stack']
        which_type, ok = QtWidgets.QInputDialog.getItem(
            self, "What Component type did you want to add?", "", comp_type,
            editable=False)
        if not ok:
            return

        if which_type == 'Slab':
            c = _default_slab(parent=self)
        elif which_type == 'LipidLeaflet':
            self.lipid_leaflet.hide()
            ok = self.lipid_leaflet.exec_()
            if not ok:
                return
            c = self.lipid_leaflet.component()
        elif which_type == 'Spline':
            # if isinstance(host, StackNode):
            #     msg("Can't add Splines to a Stack")
            #     return

            ok = self.spline_dialog.exec_()
            if not ok:
                return
            c = self.spline_dialog.component()
        elif which_type == 'Stack':
            s = _default_slab()
            c = Stack(components=[s], name='Stack')

        host.insert_component(idx + 1, c)

    @QtCore.pyqtSlot()
    def on_remove_layer_clicked(self):
        selected_indices = self.ui.treeView.selectedIndexes()

        if not selected_indices:
            return msg('Select a single row within a Structure to remove.')

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
        _component = [i for i in hierarchy if (isinstance(i, ComponentNode) or
                                               isinstance(i, StackNode))]
        if not _component:
            return msg('Select a single Component within a Structure to'
                       ' remove')

        # work out which component you have.
        component = _component[0]
        host = component.parent()
        idx = component.row()
        if isinstance(host, StructureNode) and idx in [0, len(host._data) - 1]:
            return msg("You can't remove the fronting or backing media")

        # all checking done, remove a layer
        host.remove_component(idx)

    @QtCore.pyqtSlot()
    def on_auto_limits_button_clicked(self):
        names_to_fit = self.currently_fitting_model.datasets

        datastore = self.treeModel.datastore
        data_objects = [datastore[name] for name in names_to_fit]
        # auto set the limits for the theoretical model, because it's used as
        # a springboard for the batch fit.
        data_objects.append(datastore['theoretical'])

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

    @QtCore.pyqtSlot()
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

    @QtCore.pyqtSlot()
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

    @QtCore.pyqtSlot()
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
        self.do_a_fit_and_add_to_gui(data_objects)

    def create_objective(self, data_objects):
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
            if self.settings.transformdata == 'logY':
                dataset_t = self.filter_neg_reflectivity(dataset_t)

            objective = Objective(data_object.model,
                                  dataset_t,
                                  name=data_object.name,
                                  transform=t,
                                  use_weights=useerrors)
            objectives.append(objective)

        if len(objectives) == 1:
            objective = objectives[0]
        else:
            objective = GlobalObjective(objectives)

        return objective

    def do_a_fit_and_add_to_gui(self, data_objects):
        objective = self.create_objective(data_objects)

        alg = self.settings.fitting_algorithm

        # figure out how many varying parameters
        vp = objective.varying_parameters()
        if not vp:
            return msg("No parameters are being varied.")

        fitter = CurveFitter(objective)

        progress = ProgressCallback(self, objective=objective)
        progress.show()

        methods = {'DE': 'differential_evolution',
                   'LM': 'least_squares',
                   'L-BFGS-B': 'L-BFGS-B',
                   'MCMC': 'MCMC'}

        # least squares doesnt have a callback
        kws = {'callback': progress.callback}

        if alg == 'LM':
            kws.pop('callback')

        if methods[alg] != 'MCMC':
            try:
                # workers is added to differential evolution in scipy 1.2
                with MapWrapper(-1) as workers:
                    if alg == 'DE':
                        kws['workers'] = workers
                    fitter.fit(method=methods[alg],
                               **kws)

                print(str(objective))
            except RuntimeError as e:
                # user probably aborted the fit
                # but it's still worth creating a fit curve, so don't return
                # in this catch block
                msg(repr(e))
                print(repr(e))
                print(objective)
            except Exception as e:
                # Typically shown when sensible limits weren't provided
                msg(repr(e))
                progress.close()
                return None
        else:
            # TODO implement MCMC
            pass

        progress.close()

        # mark models as having been updated
        # prevent the GUI from updating whilst we change all the values
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

        # update all the chi2 values
        self.calculate_chi2(data_objects)
        # plot the fit and sld_profile
        # TODO refactor both of the following into a single method?
        self.add_data_objects_to_graphs(data_objects)
        self.redraw_data_object_graphs(data_objects)

    @QtCore.pyqtSlot(int)
    def on_only_fitted_stateChanged(self, arg_1):
        """
        only display fitted parameters
        """
        self.treeFilter.only_fitted = arg_1
        self.treeFilter._fitted_datasets = (
            self.currently_fitting_model.datasets)
        self.treeFilter.invalidateFilter()

    @QtCore.pyqtSlot(int)
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

    @QtCore.pyqtSlot(QtCore.QModelIndex)
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

        self.currentCell['item'] = item
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True

    @QtCore.pyqtSlot(int)
    def on_paramsSlider_valueChanged(self, arg_1):
        # short circuit if the treeview hasn't been clicked yet
        if not hasattr(self, 'currentCell'):
            return

        c = self.currentCell
        item = c['item']

        if not c['readyToChange']:
            return

        val = (c['lowlim'] +
               (arg_1 / 1000.) * np.fabs(c['lowlim'] - c['hilim']))

        item.parameter.value = val

        # get the index for the change
        row = item.index.row()
        # who is the parent
        parent_index = item.parent().index
        index = self.treeModel.index(row, 1, parent_index)
        self.treeModel.dataChanged.emit(index, index,
                                        [QtCore.Qt.EditRole])

    @QtCore.pyqtSlot()
    def on_paramsSlider_sliderReleased(self):
        try:
            self.currentCell['readyToChange'] = False
            self.ui.paramsSlider.setValue(499)
            item = self.currentCell['item']

            val = item.parameter.value

            if val < 0:
                low_lim = 2 * val
                hi_lim = 0
            else:
                low_lim = 0
                hi_lim = 2 * val

            self.currentCell['val'] = val
            self.currentCell['lowlim'] = low_lim
            self.currentCell['hilim'] = hi_lim
            self.currentCell['readyToChange'] = True

        except (ValueError, AttributeError, KeyError):
            return

        # for some reason linked parameters don't update well when the slider
        # moves. Setting focus to the treeview and back to the slider seems
        # to make the update happen. One could issue more dataChanged signals
        # from sliderValue changed, but I'd have to figure out all the other
        # nodes
        self.ui.treeView.setFocus(QtCore.Qt.OtherFocusReason)
        self.ui.paramsSlider.setFocus(QtCore.Qt.OtherFocusReason)

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

        self.ui.paramsSlider.setFocus(QtCore.Qt.OtherFocusReason)
        self.ui.treeView.setFocus(QtCore.Qt.OtherFocusReason)

    def link_equivalent_action(self):
        # link equivalent parameters across a whole range of datasets.
        # the datasets all need to have the same structure for this to work.

        # retrieve data_objects that need to be linked
        datastore = self.treeModel.datastore

        self.data_object_selector.setWindowTitle("Select equivalent datasets"
                                                 " to link")
        ok = self.data_object_selector.exec_()
        if not ok:
            return
        items = self.data_object_selector.data_objects.selectedItems()
        names = [item.text() for item in items]
        if 'theoretical' in names:
            names.pop(names.index('theoretical'))

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
            if not is_same_structure([mstr_obj_node.data_object] +
                                     data_objects):
                return msg("All models must have equivalent structural"
                           " components and the same number of parameters for"
                           " equivalent linking to be available, no linking"
                           " has been done.")

            row_indices = node.row_indices()

            for data_object in data_objects:
                # retrieve the similar parameter from row indices
                row_indices[1] = self.treeModel.data_object_row(
                    data_object.name)
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

        self.ui.paramsSlider.setFocus(QtCore.Qt.OtherFocusReason)
        self.ui.treeView.setFocus(QtCore.Qt.OtherFocusReason)

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
        self.ui.paramsSlider.setFocus(QtCore.Qt.OtherFocusReason)
        self.ui.treeView.setFocus(QtCore.Qt.OtherFocusReason)

    def copy_from_action(self):
        # whose model did you want to use?
        datastore = self.treeModel.datastore

        model_names = datastore.names
        which_model, ok = QtWidgets.QInputDialog.getItem(
            self, "Which model did you want to copy?",
            "model",
            model_names,
            editable=False)

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
        data_object_nodes = [n for n in flatten(nodes) if
                             isinstance(n, DataObjectNode)]

        # get unique data object nodes, these are the data objects whose
        # model you want to overwrite
        data_object_nodes = list(unique(data_object_nodes))

        # now set the model for all those data object nodes
        for don in data_object_nodes:
            new_model = deepcopy(source_model)
            new_model.name = don.data_object.name
            don.set_reflect_model(new_model)

        do = [node.data_object for node in data_object_nodes]
        self.update_gui_model(do)

    @QtCore.pyqtSlot()
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
        reflect_model_node.insert_structure(len(structures) + 3,
                                            copied_structure)

    @QtCore.pyqtSlot()
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
            return msg("Your model only contains a single Structure, removal"
                       " not possible")

        reflect_model_node.remove_structure(item.row())

    def modify_gui(self):
        """
        Only called at program initialisation. Used to add the plots
        """
        self.sldgraphs = MySLDGraphs(self.ui.sld)
        self.ui.gridLayout_4.addWidget(self.sldgraphs)

        self.reflectivitygraphs = MyReflectivityGraphs(self.ui.reflectivity)
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs)

        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs.mpl_toolbar)
        self.ui.gridLayout_4.addWidget(self.sldgraphs.mpl_toolbar)

    def redraw_data_object_graphs(self, data_objects,
                                  all=False,
                                  transform=True):
        """ Asks the graphs to be redrawn, delegating generative
        calculation to reflectivitygraphs.redraw_data_objects

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

        self.reflectivitygraphs.redraw_data_objects(data_objects,
                                                    transform=t)
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

    def tree_model_data_changed(self, top_left, bottom_right,
                                role=QtCore.Qt.EditRole):
        # if you've just changed whether you want to hold or vary a parameter
        # there is no need to update the reflectivity plots
        if not top_left.isValid():
            return

        # find out which data_object / model we're adjusting
        node = top_left.internalPointer()
        if node is None:
            return

        if (len(role) and role[0] == QtCore.Qt.CheckStateRole and
                isinstance(node, DataObjectNode)):
            # set visibility of data_object
            graph_properties = node.data_object.graph_properties
            graph_properties.visible = (node.visible is True)
            self.redraw_data_object_graphs([node.data_object])
            return

        # only redraw if you're altering values
        # otherwise we'd be performing continual updates of the model
        if top_left.column() != 1:
            return

        hierarchy = node.hierarchy()
        for n in hierarchy:
            if isinstance(n, DataObjectNode):
                self.clear_data_object_uncertainties([n.data_object])
                self.update_gui_model([n.data_object])

    def calculate_chi2(self, data_objects):
        # calculate chi2 for all the data objects
        if not len(data_objects):
            return

        if self._hold_updating:
            return

        for data_object in data_objects:
            if data_object.name == 'theoretical':
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
            if self.settings.transformdata == 'logY':
                dataset_t = self.filter_neg_reflectivity(dataset_t)

            objective = Objective(data_object.model,
                                  dataset_t,
                                  transform=t,
                                  use_weights=useerrors)
            chisqr = objective.chisqr()
            node = self.treeModel.data_object_node(data_object.name)
            node.chi2 = chisqr
            index = self.treeModel.index(node.row(), 2, node.parent().index)
            self.treeModel.dataChanged.emit(index, index)

    def update_gui_model(self, data_objects):
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
        super(ProgressCallback, self).__init__(parent)
        self.parent = parent
        self.ui = uic.loadUi(os.path.join(UI_LOCATION, 'progress.ui'), self)
        self.elapsed = 0.
        self.chi2 = 1.e308
        self.ui.timer.display(float(self.elapsed))
        self.ui.buttonBox.rejected.connect(self.abort)
        self.objective = objective

    def abort(self):
        self.abort_flag = True

    def callback(self, xk, *args, **kwds):
        # a callback for scipy.optimize.minimize, which enters
        # every iteration.
        new_time = time.time()
        if new_time - self.last_time > 1:
            # gp = self.dataset.graph_properties
            # if gp.line2Dfit is not None:
            #     self.parent.redraw_data_object_graphs([self.dataset])
            # else:
            #     self.parent.add_datasets_to_graphs([self.dataset])

            self.elapsed = new_time - self.start
            self.ui.timer.display(float(self.elapsed))
            self.last_time = new_time

            text = 'Chi2 : %f' % self.objective.chisqr(xk)
            self.ui.values.setPlainText(text)
            QtWidgets.QApplication.processEvents()
            if self.abort_flag:
                raise RuntimeError("WARNING: FIT WAS TERMINATED EARLY")

        return self.abort_flag


class ProgramSettings(object):

    def __init__(self, **kwds):
        _members = {'fitting_algorithm': 'DE',
                    'transformdata': 'logY',
                    'quad_order': 17,
                    'current_dataset_name': None,
                    'experiment_file_name': '',
                    'current_model_name': None,
                    'usedq': True,
                    'resolution': 5,
                    'fit_plugin': None,
                    'useerrors': True}

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

        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.axes[0].set_xlabel('Q')
        self.axes[0].set_ylabel('R')
        # self.axes[0].set_yscale('log')

# residual plot
# , sharex=self.axes[0]
# ax2 = self.figure.add_axes([0.1,0.04,0.85,0.14], sharex=ax, frame_on = False)
#   self.axes.append(ax2)
#   self.axes[1].set_visible(True)
#   self.axes[1].set_ylabel('residual')

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)

        self.mpl_toolbar = NavToolBar(self, parent)
        self.figure.canvas.mpl_connect('pick_event', self._pick_event)
        # self.figure.canvas.mpl_connect('key_press_event', self._key_press)

        self.draw()

    def _key_press(self, event):
        # auto scale
        if event.key == 'super+a':
            self.autoscale()

    def _pick_event(self, event):
        # pick event was a double click on the graph
        if (event.mouseevent.dblclick and
                event.mouseevent.button == 1):
            if isinstance(event.artist, lines.Line2D):
                self.mpl_toolbar.edit_parameters()

    def autoscale(self):
        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def add_data_objects(self, data_objects, transform=None):
        for data_object in data_objects:
            dataset = data_object.dataset

            graph_properties = data_object.graph_properties

            if (graph_properties.ax_data is None and
                    data_object.name != 'theoretical'):
                yt = dataset.y
                if transform is not None:
                    yt, edata = transform(dataset.x, dataset.y, dataset.y_err)

                # add the dataset
                line_instance = self.axes[0].plot(dataset.x,
                                                  yt,
                                                  markersize=3,
                                                  marker='o',
                                                  linestyle='',
                                                  markeredgecolor=None,
                                                  label=dataset.name,
                                                  picker=5)
                mfc = artist.getp(line_instance[0], 'markerfacecolor')
                artist.setp(line_instance[0], **{'markeredgecolor': mfc})

                graph_properties['ax_data'] = line_instance[0]
                if graph_properties['data_properties']:
                    artist.setp(graph_properties.ax_data,
                                **graph_properties['data_properties'])

            yfit_t = data_object.generative
            if graph_properties.ax_fit is None and yfit_t is not None:
                if transform is not None:
                    yfit_t, temp = transform(dataset.x, yfit_t)

                color = 'b'
                if graph_properties.ax_data is not None:
                    color = artist.getp(graph_properties.ax_data, 'color')
                # add the fit
                graph_properties['ax_fit'] = self.axes[0].plot(
                    dataset.x,
                    yfit_t,
                    linestyle='-',
                    color=color,
                    lw=1,
                    label='fit_' + data_object.name,
                    picker=5)[0]
                if graph_properties['fit_properties']:
                    artist.setp(graph_properties.ax_fit,
                                **graph_properties['fit_properties'])

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

            if data_object.name != 'theoretical':
                y = dataset.y
                if transform is not None:
                    y, _ = transform(dataset.x, y)

            if data_object.model is not None:
                yfit = data_object.generative

                if transform is not None:
                    yfit, efit = transform(dataset.x, yfit)

            graph_properties = data_object.graph_properties
            visible = graph_properties.visible

            if graph_properties.ax_data is not None:
                graph_properties.ax_data.set_data(dataset.x, y)
                graph_properties.ax_data.set_visible(visible)
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

    def remove_traces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]
        self.draw()


class MySLDGraphs(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        self.axes = []
        # SLD plot
        self.axes.append(self.figure.add_subplot(111))

        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.axes[0].set_xlabel('z')
        self.axes[0].set_ylabel('SLD')

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

            if (graph_properties.ax_sld_profile and
                    data_object.model is not None):
                try:
                    sld_profile = data_object.model.structure.sld_profile()

                    graph_properties.ax_sld_profile.set_data(
                        sld_profile[0],
                        sld_profile[1])
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
            if (graph_properties.ax_sld_profile is None and
                    data_object.sld_profile is not None):

                color = 'r'
                lw = 2
                if graph_properties.ax_data:
                    color = artist.getp(graph_properties.ax_data, 'color')
                    lw = artist.getp(graph_properties.ax_data, 'lw')

                try:
                    graph_properties['ax_sld_profile'] = self.axes[0].plot(
                        data_object.sld_profile[0],
                        data_object.sld_profile[1],
                        linestyle='-',
                        color=color,
                        lw=lw,
                        label='sld_' + data_object.name)[0]
                except AttributeError:
                    # this may happen for MixedReflectModel, the model doesnt
                    # have structure.sld_profile()
                    continue

                if graph_properties['sld_profile_properties']:
                    artist.setp(graph_properties.ax_sld_profile,
                                **graph_properties['sld_profile_properties'])

        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def remove_trace(self, data_object):
        if data_object.graph_properties.ax_sld_profile:
            data_object.graph_properties.ax_sld_profile.remove()
        self.draw()

    def remove_traces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]

        self.draw()


class EmittingStream(QtCore.QObject):
    # a class for rewriting stdout to a console window

    textWritten = QtCore.pyqtSignal(str)

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
        super(OpenMenu, self).__init__(parent)
        self._parent = parent
        self.copy_from_action = self.addAction("Copy a model to here")
        self.addSeparator()
        self.add_to_fit_action = self.addAction("Add to fit")
        self.remove_from_fit_action = self.addAction("Remove from fit")
        self.addSeparator()
        self.link_action = self.addAction("Link parameters")
        self.unlink_action = self.addAction("Unlink parameters")
        self.link_equivalent_action = self.addAction(
            "Link equivalent parameters on other datasets")
        self.addSeparator()
        self.add_mixed_area = self.addAction('Mixed area - add a structure')
        self.remove_mixed_area = self.addAction("Mixed area - remove a"
                                                " structure")

    def __call__(self, position):
        action = self.exec_(self._parent.mapToGlobal(position))
        if action == self.link_action:
            pass
        if action == self.unlink_action:
            pass


def msg(text):
    # utility function for displaying a message
    msgBox = QtWidgets.QMessageBox()
    msgBox.setText(text)
    return msgBox.exec_()


def _default_slab(parent=None):
    # a default slab to add to a model
    material = SLD(3.47)
    c = material(15, 3)
    c.name = 'slab'
    c.thick.name = 'thick'
    c.rough.name = 'rough'
    c.sld.real.name = 'sld'
    c.sld.imag.name = 'isld'
    c.vfsolv.name = 'vfsolv'
    return c


_DataObjectDialog = uic.loadUiType(os.path.join(UI_LOCATION,
                                                'data_object_selector.ui'))[0]


class DataObjectSelectorDialog(QtWidgets.QDialog, _DataObjectDialog):
    def __init__(self, parent=None):
        # persistent data object selector dlg
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

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
        super(CurrentlyFitting, self).__init__(parent=parent)
        self.datasets = []

    def rowCount(self, index):
        return len(self.datasets)

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            return self.datasets[row]

    def addItems(self, items):
        new_items = [i for i in items if
                     (i not in self.datasets) and (i != 'theoretical')]

        n_current = len(self.datasets)
        if new_items:
            self.beginInsertRows(QtCore.QModelIndex(),
                                 n_current,
                                 n_current + len(new_items) + 1)
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
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        # say that you want the Components to be draggable
        flags |= QtCore.Qt.ItemIsDropEnabled

        return flags

    def supportedDropActions(self):
        return QtCore.Qt.MoveAction

    def mimeTypes(self):
        return ['application/vnd.treeviewdragdrop.list']

    def dropMimeData(self, data, action, row, column, parent):
        # drop datasets from the treeview
        if action == QtCore.Qt.IgnoreAction:
            return True
        if not data.hasFormat('application/vnd.treeviewdragdrop.list'):
            return False
        # what was dropped?
        ba = data.data('application/vnd.treeviewdragdrop.list')
        index_info = pickle.loads(ba)

        to_add = []
        for i in index_info:
            if i['name'] is not None:
                to_add.append(i['name'])

        if to_add:
            self.addItems(unique(to_add))
            return True

        return False
