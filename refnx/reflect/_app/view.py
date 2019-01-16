import os.path
from copy import deepcopy
import pickle
import os
import sys
import time

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
from .reflectivity_parameters_gui_model import BaseModel, LayerModel
from .limits_gui_model import LimitsModel
from .datastore import DataStore
from .datastore_gui_model import DataStoreModel
from .paramsstore_gui_model import ParamsStoreModel
from .globalfitting_gui_model import (GlobalFitting_DataModel,
                                      GlobalFitting_ParamModel,
                                      GlobalFitting_Settings)

import refnx
from refnx.analysis import (CurveFitter, Objective,
                            Transform)
from refnx.dataset import Data1D


matplotlib.use('Qt5Agg')
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

        # redirect stdout to a console window
        console = EmittingStream()
        sys.stdout = console
        console.textWritten.connect(self.writeTextToConsole)

        #######################################################################
        # Everything ending in 'model' refers to a QtAbstract<x>Model.  These
        # are the basis for GUI elements.  They also contain data.
        data_container = DataStore()

        self.data_store_model = DataStoreModel(data_container, self)
        self.data_store_model.dataChanged.connect(self.dataset_changed)
        self.params_store_model = ParamsStoreModel(data_container, self)

        #######################################################################

        # attach the reflectivity graphs and the SLD profiles
        self.modify_gui()

        # holds miscellaneous information on program settings
        self.settings = ProgramSettings()
        self.settings.current_dataset_name = 'theoretical'

        theoretical = self.data_store_model['theoretical']
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

        # These are QtCore.QAbstract<x>Models for displaying reflectometry
        # parameters
        self.base_model = BaseModel(model, self)
        self.layer_model = LayerModel(model, self)
        self.limits_model = LimitsModel(model, self)

        ################################################
        # set up models for reflectivity panel widgets
        # A widget for deciding which datasets to view in the graphs.
        self.ui.data_options_tableView.setModel(self.data_store_model)

        # combo boxes for the reflectivity dataset selection
        self.ui.dataset_comboBox.setModel(self.data_store_model)
        self.ui.dataset_comboBox.setModelColumn(0)
        self.ui.model_comboBox.setModel(self.params_store_model)

        self.ui.baseModelView.setModel(self.base_model)
        self.ui.layerModelView.setModel(self.layer_model)
        self.ui.limitsModelView.setModel(self.limits_model)

        self.layer_model.dataChanged.connect(self.update_gui_model)
        self.base_model.dataChanged.connect(self.update_gui_model)
        self.base_model.dataChanged.connect(self.limits_model.modelReset)
        self.layer_model.dataChanged.connect(self.limits_model.modelReset)

        # A signal for when the dataset changes in the reflectivity panel
        self.changer = DataSelectionChanges()
        self.ui.dataset_comboBox.currentIndexChanged.connect(
            self.changer.selectionChanged)
        self.changer.change.connect(self.ui.dataset_comboBox.setCurrentIndex)

        self.restore_settings()

        # globalfitting tab
        gf_settings = GlobalFitting_Settings()
        self.globalfitting_data_model = GlobalFitting_DataModel(gf_settings,
                                                                self)
        self.globalfitting_param_model = GlobalFitting_ParamModel(gf_settings,
                                                                  self)

        self.ui.globalfitting_DataView.setModel(self.globalfitting_data_model)
        self.ui.globalfitting_ParamsView.setModel(
            self.globalfitting_param_model)
        self.ui.globalfitting_DataView.setEditTriggers(
            QtWidgets.QAbstractItemView.AllEditTriggers)
        self.globalfitting_data_model.data_model_changed.connect(
            self.globalfitting_param_model.data_model_changed)
        self.globalfitting_param_model.dataChanged.connect(
            self.GFupdate_gui_model)

        print('Session started at:', time.asctime(time.localtime(time.time())))

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def dataset_changed(self, arg_1, arg_2):
        if arg_1.column() < 0:
            return

        name = self.data_store_model.datastore.names[arg_1.row()]
        data_object = self.data_store_model.datastore[name]
        gp = data_object.graph_properties
        if gp['ax_data'] is not None:
            gp['ax_data'].set_visible(gp['visible'])
        self.redraw_data_object_graphs([data_object])

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
        state['data_store_model.datastore'] = self.data_store_model.datastore
        state['history'] = self.ui.console_text_edit.toPlainText()
        state['settings'] = self.settings
# state['globalfitting_settings'] = self.globalfitting_ParamModel.gf_settings

        with open(os.path.join(experiment_file_name), 'wb') as f:
            pickle.dump(state, f, 0)

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

        try:
            self.data_store_model.datastore = state[
                'data_store_model.datastore']
            self.params_store_model.datastore = state[
                'data_store_model.datastore']
            self.ui.console_text_edit.setPlainText(state['history'])
            self.settings = state['settings']
            self.settings.experiment_file_name = experiment_file_name
# self.globalfitting_ParamModel.gf_settings = state['globalfitting_settings']
# self.globalfitting_DataModel.gf_settings = state['globalfitting_settings']

        except KeyError as e:
            print(type(e), e.message)
            return

        self.restore_settings()

        # remove and add datasetsToGraphs
        self.reflectivitygraphs.remove_traces()
        self.sldgraphs.remove_traces()
        ds = [d for d in self.data_store_model]
        self.add_data_objects_to_graphs(ds)

        # when you load in the theoretical model you destroy the link to the
        # gui, reinstate it.
        model = self.data_store_model['theoretical'].model
        self.base_model.model = model
        self.layer_model.model = model
        self.limits_model.model = model

        # either of base, layer should cause update_gui_model to fire
        self.params_store_model.modelReset.emit()
        self.base_model.modelReset.emit()
        self.layer_model.modelReset.emit()
        self.data_store_model.modelReset.emit()
        self.limits_model.modelReset.emit()

        self.reflectivitygraphs.draw()

    def restore_settings(self):
        """
        applies the program settings to the GUI
        """
        datastore = self.data_store_model.datastore
        model_names = self.params_store_model.model_names

        try:
            self.ui.dataset_comboBox.setCurrentIndex(
                datastore.names.index(self.settings.current_dataset_name))
            self.ui.model_comboBox.setCurrentIndex(
                model_names.index(self.settings.current_model_name))
        except (AttributeError, KeyError, ValueError):
            pass

        title = 'Motofit'
        if len(self.settings.experiment_file_name):
            title += ' - ' + self.settings.experiment_file_name
        self.setWindowTitle(title)

        self.ui.res_SpinBox.setValue(self.settings.resolution)
        self.ui.use_dqwave_checkbox.setChecked(self.settings.usedq)
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
        datastore = self.data_store_model.datastore

        existing_data_objects = set(datastore.names)

        data_objects = []
        fnames = []
        for file in files:
            if os.path.isfile(file):
                try:
                    data_object = datastore.load(file)
                    if data_object is not None:
                        data_objects.append(data_object)
                        fnames.append(file)
                except RuntimeError:
                    continue

        # new data_object_names
        loaded_data_objects = set([data_object.name for data_object
                                   in data_objects])

        # for the intersection of loaded and old, refresh the plot.
        refresh_names = existing_data_objects.intersection(loaded_data_objects)
        refresh_data_objects = [self.data_store_model[name] for name
                                in refresh_names]
        self.redraw_data_object_graphs(refresh_data_objects)

        # for totally new, then add to graphs
        new_names = loaded_data_objects.difference(existing_data_objects)
        new_data_objects = [self.data_store_model[name] for name in new_names]
        self.add_data_objects_to_graphs(new_data_objects)

        # refresh the data_store_model
        self.data_store_model.modelReset.emit()
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
        datasets = list(self.data_store_model.datastore.names)
        del(datasets[datasets.index('theoretical')])
        if not len(datasets):
            return

        which_dataset, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Which dataset did you want to remove?",
            "dataset",
            datasets,
            editable=False)

        if not ok:
            return
        self.reflectivitygraphs.remove_trace(
            self.data_store_model[which_dataset])
        self.data_store_model.remove(which_dataset)

    @QtCore.pyqtSlot()
    def on_actionSave_Fit_triggered(self):
        fits = []
        for data_object in self.data_store_model:
            if data_object.model is not None:
                fits.append(data_object.name)

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
                    self.data_store_model[fit].save_fit(
                        os.path.join(folder[0], 'fit_' + fit + '.dat'))
        else:
            fitFileName, ok = QtWidgets.QFileDialog.getSaveFileName(
                self, 'Save fit as:', 'fit_' + which_fit + '.dat')
            if not ok:
                return
            self.data_store_model[which_fit].save_fit(fitFileName)

    def load_model(self, model_file_name):
        with open(model_file_name, 'rb') as f:
            model = pickle.load(f)

        self.params_store_model[model.name] = model
        self.params_store_model.modelReset.emit()

    @QtCore.pyqtSlot()
    def on_actionLoad_Model_triggered(self):
        # load a model
        model_file_name, ok = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select Model File')
        if not ok:
            return
        self.load_model(model_file_name)

    @QtCore.pyqtSlot()
    def on_actionSave_Model_triggered(self):
        # save a model
        # which model are you saving?
        model_names = self.params_store_model.model_names
        model_names.append('-all-')

        which_model, ok = QtWidgets.QInputDialog.getItem(
            self, "Which model did you want to save?",
            "model",
            model_names,
            editable=False)

        if not ok:
            return

        if which_model == '-all-':
            dialog = QtWidgets.QFileDialog(self)
            dialog.setFileMode(QtWidgets.QFileDialog.Directory)
            if dialog.exec_():
                folder = dialog.selectedFiles()

                model_names = self.params_store_model.model_names
                models = self.params_store_model.models
                for model_name, model in zip(model_names, models):
                    fname = os.path.join(folder[0],
                                         'coef_' + model_name + '.pkl')
                    with open(fname, 'wb') as f:
                        pickle.dump(model, f)
        else:
            suggested_name = os.path.join(os.getcwd(),
                                          'coef_' + which_model + '.pkl')
            modelFileName, ok = QtWidgets.QFileDialog.getSaveFileName(
                self,
                'Save model as:',
                suggested_name)
            if not ok:
                return

            # retrieve the data_object
            data_object = self.data_store_model[which_model]
            with open(modelFileName, 'wb') as f:
                pickle.dump(data_object.model, f)

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

    def change_Q_range(self, qmin, qmax, numpnts, res):
        theoretical = self.data_store_model['theoretical']
        dataset = theoretical.dataset

        new_x = np.linspace(qmin, qmax, numpnts)
        new_y = np.zeros_like(new_x) * np.nan
        dataset.data = (new_x, new_y)

        self.update_gui_model()

    @QtCore.pyqtSlot()
    def on_actionChange_Q_range_triggered(self):
        theoretical = self.data_store_model['theoretical']
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

        res = self.ui.res_SpinBox.value()

        ok = qrangeGUI.exec_()
        if ok:
            self.change_Q_range(float(qrangeGUI.qmin.text()),
                                float(qrangeGUI.qmax.text()),
                                qrangeGUI.numpnts.value(), res)

    @QtCore.pyqtSlot()
    def on_actionTake_Snapshot_triggered(self):
        snapshotname, ok = QtWidgets.QInputDialog.getText(
            self,
            'Take a snapshot',
            'snapshot name')
        if not ok:
            return

        # is the snapshot already present?
        snapshot_exists = self.data_store_model[snapshotname]
        data_object = self.data_store_model.snapshot(snapshotname)

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
        self.update_gui_model()

    @QtCore.pyqtSlot()
    def on_actionBatch_Fit_triggered(self):
        if len(self.data_store_model) < 2:
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("You have no loaded datasets")
            msgBox.exec_()
            return

        # iterate and fit over all the datasets
        for data_object in self.data_store_model:
            if data_object.name == 'theoretical':
                continue
            self.do_a_fit_and_add_to_gui(data_object)

    @QtCore.pyqtSlot()
    def on_actionRefresh_Data_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.data_store_model.datastore.refresh()
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
        with open(os.path.join(licence_dir, 'about'), 'r') as f:
            text.append(''.join(f.readlines()))

        for licence in licences:
            fname = os.path.join(licence_dir, licence)
            with open(fname, 'r') as f:
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
    def on_add_layer_clicked(self):
        model = self.data_store_model['theoretical'].model
        oldlayers = len(model.structure) - 2

        dlg = QtWidgets.QInputDialog(self)
        dlg.setInputMode(QtWidgets.QInputDialog.IntInput)
        dlg.setIntMaximum(oldlayers)
        dlg.setIntMinimum(0)
        dlg.setIntValue(0)
        dlg.resize(300, 100)
        dlg.setLabelText("After which layer do you want to insert"
                         " the new one?")
        ok = dlg.exec_()

        if not ok:
            return False

        self.layer_model.add_layer(dlg.intValue())

    @QtCore.pyqtSlot()
    def on_remove_layer_clicked(self):
        model = self.data_store_model['theoretical'].model
        oldlayers = len(model.structure) - 2
        if oldlayers == 0:
            return False

        dlg = QtWidgets.QInputDialog(self)
        dlg.setInputMode(QtWidgets.QInputDialog.IntInput)
        dlg.setIntMaximum(oldlayers)
        dlg.setIntMinimum(1)
        dlg.setIntValue(1)
        dlg.resize(300, 100)
        dlg.setLabelText("Which layer would you like to remove?")
        ok = dlg.exec_()

        if not ok:
            return False

        self.layer_model.remove_layer(dlg.intValue())

    @QtCore.pyqtSlot()
    def on_auto_limits_button_clicked(self):
        model = self.params_store_model['theoretical']
        var_pars = model.parameters.varying_parameters()
        if not len(var_pars):
            return None

        for par in var_pars:
            val = par.value
            bounds = par.bounds
            if val < 0:
                bounds.ub = 0
                bounds.lb = 2 * val
            else:
                bounds.lb = 0
                bounds.ub = 2 * val

        self.limits_model.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())

    @QtCore.pyqtSlot()
    def on_do_fit_button_clicked(self):
        """
        you should do a fit
        """
        cur_data_name = self.settings.current_dataset_name

        if cur_data_name == 'theoretical':
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Please select a dataset to fit.")
            msgBox.exec_()
            return

        data_object = self.data_store_model[cur_data_name]
        self.do_a_fit_and_add_to_gui(data_object)

    def do_a_fit_and_add_to_gui(self, data_object):

        alg = self.settings.fitting_algorithm

        model = self.data_store_model['theoretical'].model
        t = Transform(self.settings.transformdata)
        useerrors = self.settings.useerrors

        # TODO fix resolution
        objective = Objective(model,
                              data_object.dataset,
                              name=data_object.name,
                              transform=t,
                              use_weights=useerrors)

        fitter = CurveFitter(objective)

        progress = ProgressCallback(self, objective=objective)
        progress.show()

        methods = {'DE': 'differential_evolution',
                   'LM': 'least_squares',
                   'L-BFGS-B': 'L-BFGS-B',
                   'MCMC': 'MCMC'}

        if methods[alg] != 'MCMC':
            try:
                fitter.fit(method=methods[alg],
                           callback=progress.callback)
                print(objective)
            except RuntimeError as e:
                # user probably aborted the fit
                # but it's still worth creating a fit curve, so don't return
                # in this catch block
                msgBox = QtWidgets.QMessageBox()
                msgBox.setText(repr(e))
                msgBox.exec_()
                print(repr(e))
                print(objective)
            except ValueError as e:
                # Typically shown when sensible limits weren't provided
                msgBox = QtWidgets.QMessageBox()
                msgBox.setText(repr(e))
                msgBox.exec_()
                progress.close()
                return None
        else:
            # TODO implement MCMC
            pass

        progress.close()

        # copy model into data_object
        new_model = deepcopy(model)
        new_model.name = data_object.name
        data_object.model = new_model
        data_object.objective = objective

        # update the GUI. Both these will cause update_gui_model to be called
        # this will also recalculate chi2, etc.
        self.layer_model.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())
        self.base_model.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())

        # plot the fit and sld_profile
        # TODO refactor both of the following into a single method?
        graph_properties = data_object.graph_properties
        if graph_properties.ax_fit is None:
            self.add_data_objects_to_graphs([data_object])
        else:
            self.redraw_data_object_graphs([data_object])

    @QtCore.pyqtSlot(int)
    def on_tabWidget_currentChanged(self, arg_1):
        if arg_1 == 0:
            self.layer_model.modelReset.emit()
            self.base_model.modelReset.emit()

    @QtCore.pyqtSlot(str)
    def on_dataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        self.settings.current_dataset_name = arg_1
        self.update_gui_model()

    @QtCore.pyqtSlot(str)
    def on_model_comboBox_currentIndexChanged(self, arg_1):
        """
        model selection changed, update view with parameters from model.
        """
        self.settings.current_model_name = arg_1
        self.select_a_model(arg_1)

    @QtCore.pyqtSlot(str)
    def on_model_comboBox_activated(self, arg_1):
        """
        model selection changed, update view with parameters from model.
        """
        self.settings.current_model_name = arg_1
        self.select_a_model(arg_1)

    def select_a_model(self, arg_1):
        try:
            model = self.params_store_model[arg_1]
            if model is None:
                return

            theoretical = deepcopy(model)
            self.params_store_model['theoretical'] = theoretical

            self.base_model.model = theoretical
            self.layer_model.model = theoretical
            self.limits_model.model = theoretical

            self.base_model.modelReset.emit()
            self.layer_model.modelReset.emit()
            self.limits_model.modelReset.emit()
            self.update_gui_model()
        except KeyError:
            return
        except IndexError:
            print(model)

    @QtCore.pyqtSlot(float)
    def on_res_SpinBox_valueChanged(self, arg_1):
        self.settings.resolution = arg_1

        theoretical = self.data_store_model['theoretical']
        dataset = theoretical.dataset
        dataset.x_err = arg_1 * dataset.x / 100.
        self.update_gui_model()

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
        self.update_gui_model()

    @QtCore.pyqtSlot(int)
    def on_use_dqwave_checkbox_stateChanged(self, arg_1):
        """
        """
        if arg_1:
            use = True
        else:
            use = False

        self.settings.usedq = use
        self.update_gui_model()

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_layerModelView_clicked(self, index):
        row = index.row()
        col = index.column()

        structure = self.data_store_model['theoretical'].model.structure
        nlayers = len(structure) - 2

        slab = structure[row]
        par = [slab.thick, slab.sld.real, slab.sld.imag, slab.rough,
               slab.vfsolv][col]

        self.currentCell = {}

        if row == 0 and col in [0, 3, 4]:
            return
        if row == nlayers + 1 and col in [0, 4]:
            return

        try:
            val = par.value
        except ValueError:
            return
        except AttributeError:
            return

        if val < 0:
            lowlim = 2 * val
            hilim = 0
        else:
            lowlim = 0
            hilim = 2 * val

        self.currentCell['par'] = par
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.layer_model

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_baseModelView_clicked(self, index):

        data_object = self.data_store_model['theoretical']
        model = data_object.model

        col = index.column()
        self.currentCell = {}

        par = [model.scale, model.bkg][col]
        try:
            val = float(par)
        except ValueError:
            return
        except AttributeError:
            return

        if val < 0:
            lowlim = 2 * val
            hilim = 0
        else:
            lowlim = 0
            hilim = 2 * val

        self.currentCell['par'] = par
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.base_model

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_globalfitting_ParamsView_clicked(self, index):
        row = index.row()
        col = index.column()
        which_dataset = col // 2

        gf_settings = self.globalfitting_data_model.gf_settings
        params = gf_settings.parameters[which_dataset]
        names = params.keys()

        self.currentCell = {}

        try:
            val = params[names[row]].value
        except ValueError:
            return
        except AttributeError:
            return

        if val < 0:
            lowlim = 2 * val
            hilim = 0
        else:
            lowlim = 0
            hilim = 2 * val

        self.currentCell['col'] = col
        self.currentCell['name'] = names[row]
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.globalfitting_param_model

    @QtCore.pyqtSlot(int)
    def on_paramsSlider_valueChanged(self, arg_1):
        try:
            c = self.currentCell
            par = c['par']

            if not c['readyToChange']:
                return

            val = (c['lowlim'] +
                   (arg_1 / 1000.) * np.fabs(c['lowlim'] - c['hilim']))

            par.value = val

            c['model'].dataChanged.emit(
                QtCore.QModelIndex(),
                QtCore.QModelIndex())
        except AttributeError:
            return
        except KeyError:
            return

    @QtCore.pyqtSlot()
    def on_paramsSlider_sliderReleased(self):
        try:
            self.currentCell['readyToChange'] = False
            self.ui.paramsSlider.setValue(499)
            par = self.currentCell['par']

            val = par.value

            if val < 0:
                low_lim = 2 * val
                hi_lim = 0
            else:
                low_lim = 0
                hi_lim = 2 * val

            self.currentCell['val'] = par.value
            self.currentCell['lowlim'] = low_lim
            self.currentCell['hilim'] = hi_lim
            self.currentCell['readyToChange'] = True

        except (ValueError, AttributeError, KeyError):
            return

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

        header = self.ui.layerModelView.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

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

        if all:
            data_objects = [data_object for data_object in
                            self.data_store_model]

        if callable(transform):
            t = transform
        elif transform:
            t = Transform(self.settings.transformdata)
        else:
            t = None

        self.reflectivitygraphs.redraw_data_objects(data_objects,
                                                    transform=t)
        self.sldgraphs.redraw_data_objects(data_objects)

    def update_gui_model(self):
        theoretical = self.data_store_model['theoretical']

        cur_data_name = self.settings.current_dataset_name
        # self.apply_settings_to_params(params)

        # tell the limits view the number of fitted parameters
        # self.limits_model.modelReset.emit()

        # evaluate the theoretical model
        # TODO correct resolution here
        model = theoretical.model
        # data = theoretical.dataset

        self.redraw_data_object_graphs([theoretical])

        if (cur_data_name != 'theoretical' and
                cur_data_name is not None):

            try:
                current_data_object = self.data_store_model[cur_data_name]
            except KeyError:
                return

            # res = self.settings.resolution
            # usedq = self.settings.usedq
            useerrors = self.settings.useerrors

            t = None
            if self.settings.transformdata is not None:
                transform = Transform(self.settings.transformdata)
                t = transform

            # TODO implement correct resolution smearing
            objective = Objective(model,
                                  current_data_object.dataset,
                                  transform=t,
                                  use_weights=useerrors)
            chisqr = objective.chisqr()
            self.ui.chi2.setValue(chisqr)

    def add_data_objects_to_graphs(self, data_objects):
        transform = Transform(self.settings.transformdata)
        t = transform

        self.reflectivitygraphs.add_data_objects(data_objects, transform=t)
        self.sldgraphs.add_data_objects(data_objects)

    @QtCore.pyqtSlot()
    def on_addGFDataSet_clicked(self):
        datasets = self.data_store_model.datastore.names

        which_dataset, ok = QtWidgets.QInputDialog.getItem(
            self, "Which dataset did you want to add?", "dataset",
            datasets, editable=False)
        if not ok:
            return

        self.globalfitting_data_model.add_DataSet(which_dataset)

    @QtCore.pyqtSlot()
    def on_linkGFparam_clicked(self):
        select = self.ui.globalfitting_DataView.selectionModel()
        indices = select.selectedIndexes()
        self.globalfitting_data_model.link_selection(indices)

    @QtCore.pyqtSlot()
    def on_unlinkGFparam_clicked(self):
        select = self.ui.globalfitting_DataView.selectionModel()
        indices = select.selectedIndexes()
        self.globalfitting_data_model.unlink_selection(indices)

    @QtCore.pyqtSlot(int)
    def on_globalParamsSlider_valueChanged(self, arg_1):
        try:
            c = self.currentCell

            which_dataset = c['col'] // 2

            gf_settings = self.globalfitting_data_model.gf_settings
            params = gf_settings.parameters[which_dataset]
            name = c['name']

            if not c['readyToChange']:
                return

            val = c['lowlim'] + \
                (arg_1 / 1000.) * np.abs(c['lowlim'] - c['hilim'])

            params[name].value = val

            c['model'].dataChanged.emit(
                QtCore.QModelIndex(),
                QtCore.QModelIndex())
        except AttributeError:
            return
        except KeyError:
            return

    @QtCore.pyqtSlot()
    def on_globalParamsSlider_sliderReleased(self):
        try:
            c = self.currentCell
            c['readyToChange'] = False

            which_dataset = c['col'] // 2

            gf_settings = self.globalfitting_data_model.gf_settings
            params = gf_settings.parameters[which_dataset]
            name = c['name']

            self.ui.globalParamsSlider.setValue(499)

            val = params[name].value

            if val < 0:
                lowlim = 2 * val
                hilim = 0
            else:
                lowlim = 0
                hilim = 2 * val

            self.currentCell['val'] = val
            self.currentCell['lowlim'] = lowlim
            self.currentCell['hilim'] = hilim
            self.currentCell['readyToChange'] = True

        except (ValueError, AttributeError, KeyError):
            return

    @QtCore.pyqtSlot()
    def on_do_gf_fit_clicked(self):
        print('___________________________________________________')
        print(self.settings.fitting_algorithm, self.settings.transformdata)

        # how did you want to fit the dataset - logY vs X, lin Y vs X, etc.
        # select a transform.  Note that we have to transform the data for
        # the fit as well
        transform_fnctn = Transform(
            self.settings.transformdata).transform
        alg = self.settings.fitting_algorithm

        gf_settings = self.globalfitting_data_model.gf_settings
        global_fitter = self.create_gf_object(
            use_errors=self.settings.useerrors,
            transform=transform_fnctn)

        if alg == 'DE':
            global_fitter.fit(method='differential_evolution')

        elif alg == 'LM':
            global_fitter.fit(method='leastsq')

        elif alg == 'MCMC':
            global_fitter.mcmc()

        # update the GUI
        self.GFupdate_gui_model()

        for name, params in gf_settings.dataset_names, gf_settings.parameters:
            new_params = deepcopy(params)
            self.params_store_model.add(new_params, 'coef_%s' % name)

        print('___________________________________________________')

    def GFupdate_gui_model(self):
        gf_settings = self.globalfitting_data_model.gf_settings

        use_errors = self.settings.useerrors
        global_fitter = self.create_gf_object(use_errors=use_errors)
        residuals = global_fitter.residuals(global_fitter.params).ravel()
        chisqr = np.sum(residuals**2)
        chisqr /= residuals.size
        self.ui.chi2GF.setValue(chisqr)

        global_fitter = self.create_gf_object(transform=False,
                                              use_errors=use_errors)

        # evaluate the fit and sld_profile
        for fitter, dataset_name in zip(global_fitter.fitters,
                                        gf_settings.dataset_names):
            dataset = self.data_store_model[dataset_name]
            dataset.fit = fitter.model(fitter.params)

            if hasattr(fitter, 'sld_profile'):
                dataset.sld_profile = fitter.sld_profile(fitter.params)

        # new_params = deepcopy(minimizer.params)
            if dataset.graph_properties.line2Dfit is None:
                self.add_data_objects_to_graphs([dataset])
            else:
                self.redraw_data_object_graphs([dataset])

        return global_fitter.model(global_fitter.params)

    def create_gf_object(self, transform=True, use_errors=True):
        """
        Create a global fitter object that can fit data

        Parameters
        ----------
        transform: bool
            True - transform the data according to program default
            False - no transform applied
        use_errors: bool
            Use error bar weighting during a fit?
        """
        gf_settings = self.globalfitting_data_model.gf_settings
        datasets = self.data_store_model.datastore

        fitters = list()

        # constraints = gf_settings.linkages

        for idx, dataset_name in enumerate(gf_settings.dataset_names):
            # retrieve the dataobject
            dataset = datasets[dataset_name]

            # get the parameters
            params = gf_settings.parameters[idx]

            if transform:
                transform_fnctn = Transform(
                    self.settings.transformdata).transform
            else:
                transform_fnctn = Transform(None).transform

            tempdataset = Data1D(dataset.data)

            tempdataset.y, tempdataset.y_err = transform_fnctn(
                tempdataset.x,
                tempdataset.y,
                tempdataset.y_err)

            # find out which points in the dataset aren't finite
            mask = ~np.isfinite(tempdataset.y)

            if not use_errors:
                tempdataset.y_err = np.ones_like(tempdataset.y)

            # have a kws dictionary
            kws = {'dqvals': tempdataset.x_err, 'transform': transform_fnctn}

            fit_plugin = self.plugin_store_model[gf_settings.fitplugins[idx]]

            if issubclass(fit_plugin, CurveFitter):
                fitter = fit_plugin(tempdataset.x,
                                    tempdataset.y,
                                    params,
                                    edata=tempdataset.y_err,
                                    mask=mask,
                                    fcn_kws=kws)
            elif hasattr(fit_plugin, 'fitfuncwraps'):
                fitter = CurveFitter(fit_plugin,
                                     tempdataset.x,
                                     tempdataset.y,
                                     params,
                                     edata=tempdataset.y_err,
                                     mask=mask,
                                     fcn_kws=kws)

            fitters.append(fitter)

        # global_fitter = GlobalFitter(fitters, constraints=constraints)
        global_fitter = None
        return global_fitter


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

            text = 'Chi2 : %f' % self.objective.chisqr()
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
                    'useerrors': False}

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
            model = data_object.model

            if data_object.name != 'theoretical':
                y = dataset.y
                if transform is not None:
                    y, _ = transform(dataset.x, y)

            if model is not None:
                # TODO take care of correct resolution
                yfit = model(dataset.x, x_err=dataset.x_err)

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
                sld_profile = data_object.model.structure.sld_profile()

                graph_properties.ax_sld_profile.set_data(
                    sld_profile[0],
                    sld_profile[1])
                graph_properties.ax_sld_profile.set_visible(visible)
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

                graph_properties['ax_sld_profile'] = self.axes[0].plot(
                    data_object.sld_profile[0],
                    data_object.sld_profile[1],
                    linestyle='-',
                    color=color,
                    lw=lw,
                    label='sld_' + data_object.name)[0]

                if graph_properties['sld_profile_properties']:
                    artist.setp(graph_properties.ax_sld_profile,
                                **graph_properties['sld_profile_properties'])

        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def remove_trace(self, data_object):
        if data_object.ax_sld_profile:
            data_object.ax_sld_profile.remove()
        self.draw()

    def remove_traces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]

        self.draw()


class DataSelectionChanges(QtCore.QObject):
    change = QtCore.pyqtSignal(int)

    def __init__(self):
        super(DataSelectionChanges, self).__init__()

    @QtCore.pyqtSlot(int)
    def selectionChanged(self, arg_1):
        self.change.emit(arg_1)


class EmittingStream(QtCore.QObject):
    # a class for rewriting stdout to a console window
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

