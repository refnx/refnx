from __future__ import print_function, division
from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow
import limitsUI, progressUI, qrangedialogUI, SLDcalculatorView, aboutUI

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.artist as artist
import matplotlib.lines as lines

from UDF_GUImodel import PluginStoreModel, UDFParametersModel
from paramsstore_GUImodel import ParamsStoreModel
from reflectivity_parameters_GUImodel import BaseModel, LayerModel
import limits_GUImodel
from globalfitting_GUImodel import (GlobalFitting_DataModel,
                                    FitPluginItemDelegate,
                                    GlobalFitting_ParamModel,
                                    GlobalFitting_Settings)
from datastore_GUImodel import DataStoreModel

import refnx.analysis.reflect as reflect
import refnx.analysis.curvefitter as curvefitter
from refnx.analysis import (ReflectivityFitFunction, Transform, CurveFitter,
                            GlobalFitter)
from refnx.dataset import Data1D, ReflectDataset

from lmfit import Parameters, fit_report

import os.path
from copy import deepcopy
import numpy as np
import pickle
import datastore
import os
import sys
import time


class MyMainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.error_handler = QtGui.QErrorMessage()

        # redirect stdout to a console window
        console = EmittingStream()
        sys.stdout = console
        console.textWritten.connect(self.writeTextToConsole)

        #######################################################################
        # Everything ending in 'model' refers to a QtAbstract<x>Model.  These
        # are the basis for GUI elements.  They also contain data.

        self.data_store_model = DataStoreModel(self)
        self.data_store_model.dataChanged.connect(self.dataset_changed)

        self.params_store_model = ParamsStoreModel(self)

        self.plugin_store_model = PluginStoreModel(self)
        self.minimizer_store = datastore.MinimizersStore()

        #######################################################################
        #attach the reflectivity graphs and the SLD profiles
        self.modify_gui()

        # holds miscellaneous information on program settings
        self.settings = ProgramSettings()
        self.settings.fit_plugin = self.plugin_store_model['default']

        #create a set of theoretical parameters
        coefs = np.array([1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 25, 3.47, 0, 3])
        params = curvefitter.to_parameters(coefs,
                                names=ReflectivityFitFunction.parameter_names(
                                    nparams=coefs.size))
        params['nlayers'].vary = False

        self.settings.current_dataset_name = 'theoretical'

        tempq = np.linspace(0.008, 0.5, 1000)
        tempr = np.ones_like(tempq)
        tempe = np.ones_like(tempq)
        tempdq = tempq * 5 / 100.
        theoretical = ReflectDataset((tempq, tempr, tempe, tempdq))
        theoretical.name = 'theoretical'
        kws = {'dqvals': theoretical.x_err}

        evaluator = ReflectivityFitFunction(theoretical.x,
                                       theoretical.y,
                                       params,
                                       fcn_kws=kws)

        self.minimizer_store['default'] = evaluator

        theoretical.fit = evaluator.model(params)
        theoretical.residuals = evaluator.residuals(params)
        theoretical.params = params
        theoretical.chisqr = np.sum(theoretical.residuals.ravel()**2)
        theoretical.sld_profile = evaluator.sld_profile(params)
        self.data_store_model.add(theoretical)

        self.params_store_model.add(params, 'theoretical')

        graph_properties = theoretical.graph_properties
        graph_properties.line2Dfit = self.reflectivitygraphs.axes[0].plot(
                                        theoretical.x,
                                        theoretical.fit, color='r',
                                        linestyle='-', lw=1,
                                        label='theoretical',
                                        picker=5)[0]

        graph_properties.line2Dsld_profile = self.sldgraphs.axes[0].plot(
                                        theoretical.sld_profile[0],
                                        theoretical.sld_profile[1],
                                        linestyle='-', color='r')[0]

        # These are QtCore.QAbstract<x>Models for displaying reflectometry
        # parameters
        self.base_model = BaseModel(params, self)
        self.layer_model = LayerModel(params, self)

        # A QAbstractListModel for holding parameters for UDF plugins.
        # displays the UDF parameters in a combobox.
        self.UDFparams_store_model = self.params_store_model

        # A QAbstractTableModel for displaying UDF parameters.
        self.UDFparams_model = UDFParametersModel(None, self)

        # A widget for deciding which datasets to view in the graphs.
        self.ui.data_options_tableView.setModel(self.data_store_model)

        # combo boxes for the reflectivity dataset selection
        self.ui.dataset_comboBox.setModel(self.data_store_model)
        self.ui.dataset_comboBox.setModelColumn(0)

        # set up models for reflectivity panel widgets
        self.ui.model_comboBox.setModel(self.params_store_model)
        self.ui.baseModelView.setModel(self.base_model)
        self.ui.layerModelView.setModel(self.layer_model)
        self.layer_model.dataChanged.connect(self.update_gui_model)
        self.base_model.dataChanged.connect(self.update_gui_model)

        # which datasets are you going to fit as a UDF
        self.ui.UDFdataset_comboBox.setModel(self.data_store_model)
        self.ui.UDFdataset_comboBox.setModelColumn(0)

        # A signal for when the dataset changes in the reflectivity panel
        self.changer = DataSelectionChanges()
        self.ui.dataset_comboBox.currentIndexChanged.connect(
             self.changer.selectionChanged)
        self.changer.change.connect(self.ui.dataset_comboBox.setCurrentIndex)

        # A signal for when the UDF dataset selection changes
        self.UDFchanger = DataSelectionChanges()
        self.ui.UDFdataset_comboBox.currentIndexChanged.connect(
            self.UDFchanger.selectionChanged)
        self.UDFchanger.change.connect(
        self.ui.UDFdataset_comboBox.setCurrentIndex)

        # User defined function tab
        self.ui.UDFmodelView.setModel(self.UDFparams_model)
        self.UDFparams_model.dataChanged.connect(self.UDFupdate_gui_model)
        self.ui.UDFmodel_comboBox.setModel(self.UDFparams_store_model)
        self.ui.UDFplugin_comboBox.setModel(self.plugin_store_model)

        self.restore_settings()
        self.redraw_dataset_graphs([theoretical])

        # globalfitting tab
        gf_settings = GlobalFitting_Settings()
        self.globalfitting_DataModel = GlobalFitting_DataModel(gf_settings, self)
        self.globalfitting_ParamModel = GlobalFitting_ParamModel(gf_settings, self)

        self.ui.globalfitting_DataView.setModel(self.globalfitting_DataModel)
        self.ui.globalfitting_ParamsView.setModel(self.globalfitting_ParamModel)

        self.ui.FitPluginDelegate = FitPluginItemDelegate(
                                    self.plugin_store_model.plugins,
                                    self.ui.globalfitting_DataView)

        self.ui.globalfitting_DataView.setEditTriggers(
            QtGui.QAbstractItemView.AllEditTriggers)
        self.ui.globalfitting_DataView.setItemDelegateForRow(
            0,
            self.ui.FitPluginDelegate)

        self.globalfitting_DataModel.data_model_changed.connect(
            self.globalfitting_ParamModel.data_model_changed)
        self.globalfitting_ParamModel.dataChanged.connect(
             self.GFupdate_gui_model)

        print('Session started at:', time.asctime(time.localtime(time.time())))

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def dataset_changed(self, arg_1, arg_2):
        if arg_1.column() < 0:
            return

        name = self.data_store_model.datastore.names[arg_1.row()]
        dataset = self.data_store_model.datastore[name]
        graph_properties = dataset.graph_properties
        if graph_properties['line2D'] is not None:
            graph_properties['line2D'].set_visible(graph_properties['visible'])
        self.redraw_dataset_graphs([dataset],
                                   visible=graph_properties['visible'])

    @QtCore.Slot(QtGui.QDropEvent)
    def dropEvent(self, event):
        m = event.mimeData()
        urls = m.urls()
        for url in urls:
            try:
                self._restoreState(url.toLocalFile())
                continue
            except Exception:
                pass

            try:
                self.load_data([url.toLocalFile()])
                continue
            except Exception as inst:
                pass

            try:
                self.load_params(url.toLocalFile())
                continue
            except Exception:
                pass

            try:
                self.pluginStoreModel.add(url.toLocalFile())
                continue
            except Exception:
                pass

    @QtCore.Slot(QtGui.QDragEnterEvent)
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
        state['params_store_model.params_store'] = self.params_store_model.params_store
        state['UDFparams_store_model.params_store'] = self.UDFparams_store_model.params_store

        state['history'] = self.ui.console_text_edit.toPlainText()
        state['settings'] = self.settings
        state['plugins'] = self.plugin_store_model.plugins
#        state['globalfitting_settings'] = self.globalfitting_ParamModel.gf_settings

        with open(os.path.join(experiment_file_name), 'wb') as f:
            pickle.dump(state, f, 0)

        self.setWindowTitle('Motofit - ' + experiment_file_name)

    @QtCore.Slot()
    def on_actionSave_File_triggered(self):
        if os.path.isfile(self.settings.experiment_file_name):
            self._saveState(self.settings.experiment_file_name)
        else:
            self.on_actionSave_File_As_triggered()

    @QtCore.Slot()
    def on_actionSave_File_As_triggered(self):
        experiment_file_name, ok = QtGui.QFileDialog.getSaveFileName(
            self, caption='Save experiment as:', dir='experiment.mtft')

        if not ok:
            return

        path, ext = os.path.splitext(experiment_file_name)
        if ext != '.mtft':
            experiment_file_name = path + '.mtft'

        self._saveState(experiment_file_name)


    def _restoreState(self, experiment_file_name):
        state = None
        with open(experiment_file_name, 'rb') as f:
            state = pickle.load(f)

        if not state:
            print("Couldn't load experiment")
            return

        try:
            self.data_store_model.datastore = state[
                                    'data_store_model.datastore']
            self.params_store_model.params_store = state[
                                    'params_store_model.params_store']
            self.UDFparams_store_model.params_store = state[
                                    'UDFparams_store_model.params_store']
            self.ui.console_text_edit.setPlainText(state['history'])
            self.settings = state['settings']
            self.settings.experiment_file_name = experiment_file_name
            self.plugin_store_model.plugins = state['plugins']
            # self.globalfitting_ParamModel.gf_settings = state['globalfitting_settings']
            # self.globalfitting_DataModel.gf_settings = state['globalfitting_settings']

        except KeyError as e:
            print(type(e), e.message)
            return

        self.restore_settings()
        self.params_store_model.modelReset.emit()
        self.UDFparams_store_model.modelReset.emit()

        # remove and add datasetsToGraphs
        self.reflectivitygraphs.removeTraces()
        self.sldgraphs.removeTraces()
        ds = [d for d in self.data_store_model]
        self.add_datasets_to_graphs(ds)

#        self.reflectivitygraphs.axes[0].lines.remove(self.theoretical.line2D)
#        self.reflectivitygraphs.axes[1].lines.remove(self.theoretical.line2Dresiduals)

        # when you load in the theoretical model you destroy the link to the
        # gui, reinstate it.
        params = self.params_store_model['theoretical']
        self.base_model.params = params
        self.layer_model.params = params
        self.update_gui_model()

        #re-add the theoretical line to the reflectivity graph
        # theoretical = self.data_store_model['theoretical']
        # graph_properties = theoretical.graph_properties
        # graph_properties.line2Dfit = self.reflectivitygraphs.axes[0].plot(
        #                                             theoretical.x,
        #                                             theoretical.fit, color='b',
        #                                             linestyle='-', lw=1,
        #                                             label='theoretical')[0]
        self.reflectivitygraphs.draw()

    def restore_settings(self):
        '''
            applies the program settings to the GUI
        '''

        datastore = self.data_store_model.datastore
        params_store = self.params_store_model.params_store
        UDFparams_store = self.UDFparams_store_model.params_store

        try:
            self.ui.dataset_comboBox.setCurrentIndex(
                datastore.names.index(self.settings.current_dataset_name))
            self.ui.UDFdataset_comboBox.setCurrentIndex(
                datastore.names.index(self.settings.UDFcurrent_dataset_name))
            self.ui.model_comboBox.setCurrentIndex(
                params_store.names.index(self.settings.current_model_name))
            self.ui.UDFmodel_comboBox.setCurrentIndex(
                UDFparams_store.names.index(self.settings.UDFcurrent_model_name))
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

        self.settransformoption(self.settings.transformdata)

    def apply_settings_to_params(self, params):
         for key in self.settings.__dict__:
             params[key] = self.settings[key]

    @QtCore.Slot()
    def on_actionLoad_File_triggered(self):
        experimentFileName, ok = QtGui.QFileDialog.getOpenFileName(self,
                                                                   caption='Select Experiment File',
                                                                   filter='Experiment Files (*.mtft)')
        if not ok:
            return

        self._restoreState(experimentFileName)

    def load_data(self, files):
        for file in files:
            dataset = self.data_store_model.load(file)
            self.add_datasets_to_graphs([dataset])

    @QtCore.Slot()
    def on_actionLoad_Data_triggered(self):
        """
            you load data
        """
        files, ok = QtGui.QFileDialog.getOpenFileNames(
            self, caption='Select Reflectivity Files')
        if not ok:
            return
        self.load_data(files)

    @QtCore.Slot()
    def on_actionRemove_Data_triggered(self):
        """
            you remove data
        """
        datasets = list(self.data_store_model.datastore.names)
        del(datasets[datasets.index('theoretical')])
        if not len(datasets):
            return

        which_dataset, ok = QtGui.QInputDialog.getItem(
            self, "Which fit did you want to remove?", "dataset", datasets, editable=False)

        if not ok:
            return
        self.reflectivitygraphs.removeTrace(
            self.data_store_model[which_dataset])
        self.data_store_model.remove(which_dataset)

    @QtCore.Slot()
    def on_actionSave_Fit_triggered(self):
        fits = []
        for dataset in self.data_store_model:
            if dataset.fit is not None:
                fits.append(dataset.name)

        fits.append('-all-')

        which_fit, ok = QtGui.QInputDialog.getItem(
            self, "Which fit did you want to save?", "fit", fits, editable=False)
        if not ok:
            return

        if which_fit == '-all-':
            dialog = QtGui.QFileDialog(self)
            dialog.setFileMode(QtGui.QFileDialog.Directory)
            if dialog.exec_():
                folder = dialog.selectedFiles()
                fits.pop()
                for fit in fits:
                    self.data_store_model.datastore[fit].savefit(
                        os.path.join(folder[0], 'fit_' + fit + '.dat'))
        else:
            fitFileName, ok = QtGui.QFileDialog.getSaveFileName(
                self, caption='Save fit as:', dir='fit_' + which_fit)
            if not ok:
                return
            self.data_store_model.datastore[which_fit].save_fit(fitFileName)

    def load_params(self, fileName):
        self.params_store_model.params_store.load(fileName)
        if self.ui.model_comboBox.count() == 1:
            self.ui.model_comboBox.setCurrentIndex(-1)

    @QtCore.Slot()
    def on_actionLoad_Model_triggered(self):
        # load a model
        modelFileName, ok = QtGui.QFileDialog.getOpenFileName(
            self, caption='Select Model File')
        if not ok:
            return
        self.load_params(modelFileName)

    @QtCore.Slot()
    def on_actionSave_Model_triggered(self):
        # save a model
        # which model are you saving?
        listofmodels = list(self.params_store_model.params_store.names)
        listofmodels.append('-all-')

        which_model, ok = QtGui.QInputDialog.getItem(
            self, "Which model did you want to save?", "model", listofmodels,
            editable=False)

        if not ok:
            return

        if which_model == '-all-':
            dialog = QtGui.QFileDialog(self)
            dialog.setFileMode(QtGui.QFileDialog.Directory)
            if dialog.exec_():
                folder = dialog.selectedFiles()
                self.params_store_model.modelStore.saveModelStore(folder[0])
        else:
            suggested_name = os.path.join(os.getcwd(), which_model + '.pkl')
            params = self.params_store_model[which_model]
            modelFileName, ok = QtGui.QFileDialog.getSaveFileName(
                self, caption='Save model as:', dir=suggested_name)
            if not ok:
                return

            self.params_store_model.params_store.save(which_model, modelFileName)

    @QtCore.Slot()
    def on_actionDifferential_Evolution_triggered(self):
        if self.ui.actionDifferential_Evolution.isChecked():
            self.settings.fitting_algorithm = 'DE'
            self.ui.actionLevenberg_Marquardt.setChecked(False)
            self.ui.actionMCMC.setChecked(False)

    @QtCore.Slot()
    def on_actionMCMC_triggered(self):
        if self.ui.actionMCMC.isChecked():
            self.settings.fitting_algorithm = 'MCMC'
            self.ui.actionLevenberg_Marquardt.setChecked(False)
            self.ui.actionDifferential_Evolution.setChecked(False)

    @QtCore.Slot()
    def on_actionLevenberg_Marquardt_triggered(self):
        if self.ui.actionLevenberg_Marquardt.isChecked():
            self.settings.fitting_algorithm = 'LM'
            self.ui.actionDifferential_Evolution.setChecked(False)
            self.ui.actionMCMC.setChecked(False)

    def change_Q_range(self, qmin, qmax, numpnts, res):
        theoretical = self.data_store_model['theoretical']

        theoretical.xdata = np.linspace(qmin, qmax, numpnts)
        theoretical.ydata = np.resize(theoretical.ydata, numpnts)

        minimizer = self.minimizer_store['default']

        minimizer.xdata = theoretical.xdata
        minimizer.ydata = theoretical.ydata
        minimizer.edata = np.ones_like(theoretical.ydata)
        minimizer.set_dq(float(res))
        minimizer.userkws.update({'dqvals': theoretical.xdataSD})

        self.update_gui_model()

    @QtCore.Slot()
    def on_actionChange_Q_range_triggered(self):
        theoretical = self.data_store_model['theoretical']
        qmin = theoretical.xdata[0]
        qmax = theoretical.xdata[-1]
        numpnts = len(theoretical.xdata)

        dvalidator = QtGui.QDoubleValidator(-2.0e-308, 2.0e308, 6)

        qrangedialog = QtGui.QDialog()
        qrangeGUI = qrangedialogUI.Ui_qrange()
        qrangeGUI.setupUi(qrangedialog)
        qrangeGUI.numpnts.setValue(numpnts)
        qrangeGUI.qmin.setValidator(dvalidator)
        qrangeGUI.qmax.setValidator(dvalidator)
        qrangeGUI.qmin.setText(str(qmin))
        qrangeGUI.qmax.setText(str(qmax))

        res = self.ui.res_SpinBox.value()

        ok = qrangedialog.exec_()
        if ok:
            self.change_Q_range(float(qrangeGUI.qmin.text()),
                                float(qrangeGUI.qmax.text()),
                                qrangeGUI.numpnts.value(), res)

    @QtCore.Slot()
    def on_actionTake_Snapshot_triggered(self):
        snapshotname, ok = QtGui.QInputDialog.getText(self,
                                                      'Take a snapshot',
                                                      'snapshot name')
        if not ok:
            return

        self.data_store_model.snapshot(snapshotname)
        self.add_datasets_to_graphs(
            [self.data_store_model[snapshotname]])

    @QtCore.Slot()
    def on_actionResolution_smearing_triggered(self):
        currentVal = self.settings.quad_order
        value, ok = QtGui.QInputDialog.getInt(self,
                                              'Resolution Smearing',
                                              'Number of points for Gaussian Quadrature',
                                              currentVal,
                                              17)
        if not ok:
            return
        self.settings.quad_order = value
        self.minimizer_store['default'].set_dq(False, quad_order=value)
        self.update_gui_model()

    @QtCore.Slot()
    def on_actionLoad_Plugin_triggered(self):
        # load a model plugin
        self.load_plugin()

    @QtCore.Slot()
    def on_actionBatch_Fit_triggered(self):
        if len(self.data_store_model) < 2:
            msgBox = QtGui.QMessageBox()
            msgBox.setText("You have no loaded datasets")
            msgBox.exec_()
            return

        theoreticalmodel = self.params_store_model['theoretical']
        theoreticalmodel.default_limits()
        if self.settings.fitting_algorithm != 'LM':
            ok, limits = self.get_limits(theoreticalmodel.parameters,
                                         theoreticalmodel.fitted_parameters,
                                         theoreticalmodel.limits)

            if not ok:
                return

            theoreticalmodel.limits = np.copy(limits)

#       Have to iterate over list because datasets are stored in dictionary
        for name in self.data_store_model.datastore.names:
            if name == 'theoretical':
                continue
            self.do_a_fit_and_add_to_gui(
                self.data_store_model.datastore[name],
                theoreticalmodel)

    @QtCore.Slot()
    def on_actionRefresh_Data_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.data_store_model.datastore.refresh()
        self.redraw_dataset_graphs(None, all=True)

    @QtCore.Slot()
    def on_actionlogY_vs_X_triggered(self):
        self.settransformoption('logY')

    @QtCore.Slot()
    def on_actionY_vs_X_triggered(self):
        self.settransformoption('lin')

    @QtCore.Slot()
    def on_actionYX4_vs_X_triggered(self):
        self.settransformoption('YX4')

    @QtCore.Slot()
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

        self.redraw_dataset_graphs(None, all=True)

        # need to relimit graphs and display on a log scale if the transform
        # has changed
        self.reflectivitygraphs.axes[0].autoscale(axis='both', tight = False,
                                                  enable = True)
        self.reflectivitygraphs.axes[0].relim()
        if transform in ['lin', 'YX2']:
            self.reflectivitygraphs.axes[0].set_yscale('log')
        else:
            self.reflectivitygraphs.axes[0].set_yscale('linear')
        self.reflectivitygraphs.draw()

    @QtCore.Slot()
    def on_actionAbout_triggered(self):
        about_dialog = QtGui.QDialog()
        aboutGUI = aboutUI.Ui_Dialog()
        aboutGUI.setupUi(about_dialog)
        ok = about_dialog.exec_()

    @QtCore.Slot()
    def on_actionAutoscale_graph_triggered(self):
        self.reflectivitygraphs.autoscale()

    @QtCore.Slot()
    def on_actionSLD_calculator_triggered(self):
        SLDcalculator = SLDcalculatorView.SLDcalculatorView(self)
        SLDcalculator.show()

    def get_limits(self, params):
        alg = self.settings.fitting_algorithm
        finite_bounds = (alg == 'DE' or alg == 'MCMC')

        limitsdialog = QtGui.QDialog()
        limitsGUI = limitsUI.Ui_Dialog()
        limitsGUI.setupUi(limitsdialog)

        if finite_bounds:
            varys = curvefitter.varys(params)
            names = curvefitter.names(params)
            for name in names:
                param = params[name]
                if not np.isfinite(param.min):
                    param.min = 0.
                if not np.isfinite(param.max):
                    param.max = 2. * param.value
                if param.min > param.max:
                    param.min, param.max = param.max, param.min
                if param.min == param.max:
                    param.max = param.min + 0.1

        limitsModel = limits_GUImodel.LimitsModel(params,
                                                  finite_bounds=finite_bounds,
                                                  parent=self)

        limitsGUI.limits.setModel(limitsModel)
        header = limitsGUI.limits.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)

        ok = limitsdialog.exec_()

        if finite_bounds:
            varys = curvefitter.varys(params)
            fitted = np.where(varys)[0]
            bounds = np.array(curvefitter.bounds(params))[fitted]

            still_ok = np.all(np.isfinite(bounds))
            if not still_ok:
                msgBox = QtGui.QMessageBox()
                msgBox.setText('If you select DE or MCMC then all bounds must '
                               'be finite')
                msgBox.exec_()
                return False

        return ok

    @QtCore.Slot()
    def on_add_layer_clicked(self):
        params = self.params_store_model['theoretical']
        oldlayers = params['nlayers'].value

        insertpoint, ok = QtGui.QInputDialog.getInt(None,
                    'Where would you like to insert the new layer?',
                    'layer',
                    value=0,
                    minValue=0,
                    maxValue=oldlayers)
        if not ok:
            return False

        self.layer_model.add_layer(insertpoint)

    @QtCore.Slot()
    def on_remove_layer_clicked(self):
        params = self.params_store_model['theoretical']
        oldlayers = params['nlayers'].value
        if oldlayers == 0:
            return False

        remove_point, ok = QtGui.QInputDialog.getInt(None,
                    'Which layer would you like to remove?',
                    'layer',
                    value=1,
                    minValue=1,
                    maxValue=oldlayers)
        if not ok:
            return False

        self.layer_model.remove_layer(remove_point)

    @QtCore.Slot()
    def on_do_fit_button_clicked(self):
        '''
            you should do a fit
        '''
        cur_data_name = self.settings.current_dataset_name
        alg = self.settings.fitting_algorithm

        if cur_data_name == 'theoretical':
            msgBox = QtGui.QMessageBox()
            msgBox.setText("Please select a dataset to fit.")
            msgBox.exec_()
            return

        dataset = self.data_store_model[cur_data_name]
        params = self.params_store_model['theoretical']

        try:
            previous_coefs = self.params_store_model['coef_' + cur_data_name]
            if len(previous_coefs) == len(params):
                msgBox = QtGui.QMessageBox()
                msgBox.setText("Did you want to use previous limits/expressions?")
                msgBox.setStandardButtons(QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                msgBox.setDefaultButton(QtGui.QMessageBox.Yes)
                ret = msgBox.exec_()
                if ret == QtGui.QMessageBox.Yes:
                    for name, param in params.items():
                        if param.vary == True:
                            param.min = previous_coefs[name].min
                            param.max = previous_coefs[name].max
                        if param.expr is not None and len(param.expr):
                            param.expr = previous_coefs[name].expr
                else:
                    pass
        except KeyError:
            pass

            #copy the limits for those that are varying

        ok = self.get_limits(params)
        if not ok:
            return

        params['nlayers'].vary = False
        params['nlayers'].expr = None
        self.do_a_fit_and_add_to_gui(dataset, params)

    @QtCore.Slot()
    def on_do_UDFfit_button_clicked(self):
        '''
            you should do a fit using the fit plugin
        '''
        cur_data_name = self.settings.UDFcurrent_dataset_name
        dataset = self.data_store_model[cur_data_name]

        if cur_data_name == 'theoretical':
            msgBox = QtGui.QMessageBox()
            msgBox.setText('Please select a dataset to fit.')
            msgBox.exec_()
            return

        params = self.UDFparams_model.params

        ok = self.get_limits(params)

        if not ok:
            return

        self.UDFdo_a_fit_and_add_to_gui(dataset,
                                     params,
                                     fit_plugin=self.settings.fit_plugin)

    def UDFdo_a_fit_and_add_to_gui(self, dataset, params, fit_plugin=None):
        if dataset.name == 'theoretical':
            print('You tried to fit the theoretical dataset')
            return

        print ('___________________________________________________')
        print ('fitting to:', dataset.name)
        print(self.settings.fitting_algorithm, self.settings.transformdata)

        # how did you want to fit the dataset - logY vs X, lin Y vs X, etc.
        # select a transform.  Note that we have to transform the data for
        # the fit as well
        transform_fnctn = Transform(
            self.settings.transformdata).transform
        alg = self.settings.fitting_algorithm

        tempdataset = Data1D(dataset.data)

        tempdataset.y, tempdataset.y_err = transform_fnctn(
                                                tempdataset.x,
                                                tempdataset.y,
                                                tempdataset.y_err)

        # find out which points in the dataset aren't finite
        mask = ~np.isfinite(tempdataset.y)

        #have a kws dictionary
        kws = {'dqvals': tempdataset.x_err, 'transform': transform_fnctn}

        if fit_plugin is None:
            print('Please choose a fit plugin again')
            return

        if isinstance(fit_plugin, CurveFitter):
            minimizer = fit_plugin(tempdataset.x,
                                   tempdataset.y,
                                   params,
                                   edata=tempdataset.y_err,
                                   mask=mask,
                                   fcn_kws=kws)
        elif hasattr(fit_plugin, 'fitfuncwraps'):
            minimizer = CurveFitter(fit_plugin,
                                    tempdataset.x,
                                    tempdataset.y,
                                    params,
                                    edata=tempdataset.y_err,
                                    mask=mask,
                                    fcn_kws=kws)
        else:
            raise TypeError('fit_plugin did not subclass CurveFitter')

        minimizer.transform = transform_fnctn

        if not self.settings.useerrors:
            tempdataset.y_err = None

        progress = ProgressCallback(self, minimizer=minimizer,
                                    dataset=dataset)

        progress.show()
        minimizer.iter_cb = progress.callback2

        if alg == 'DE':
            minimizer.kws.update({'callback': progress.callback})
            minimizer.fit(method='differential_evolution')
            minimizer.kws.pop('callback')

        elif alg == 'LM':
            minimizer.fit(method='leastsq')

        elif alg == 'MCMC':
            minimizer.mcmc()

        progress.destroy()
        minimizer.iter_cb = None
        minimizer.transform = None

        #evaluate the fit and sld_profile
        dataset.fit = minimizer.model(minimizer.params)
        if hasattr(minimizer, 'sld_profile'):
            dataset.sld_profile = minimizer.sld_profile(minimizer.params)

        new_params = deepcopy(minimizer.params)
        self.UDFparams_model.modelReset.emit()
        self.UDFparams_store_model.add(new_params, 'coef_' + dataset.name)

        #update the chi2 value
        self.UDFupdate_gui_model()

        print(fit_report(minimizer))
        print('___________________________________________________')

        if dataset.graph_properties.line2Dfit is None:
            self.add_datasets_to_graphs([dataset])
        else:
            self.redraw_dataset_graphs([dataset],
                                  visible=dataset.graph_properties['visible'])

    def do_a_fit_and_add_to_gui(self, dataset, params):
        if dataset.name == 'theoretical':
            print('You tried to fit the theoretical dataset')
            return

        print ('___________________________________________________')
        print ('fitting to:', dataset.name)
        # try:
        print(self.settings.fitting_algorithm, self.settings.transformdata)

        # how did you want to fit the dataset - logY vs X, lin Y vs X, etc.
        # select a transform.  Note that we have to transform the data for
        # the fit as well
        transform_fnctn = Transform(
            self.settings.transformdata).transform
        alg = self.settings.fitting_algorithm
        res = self.settings.resolution

        tempdataset = Data1D(dataset.data)

        tempdataset.y, tempdataset.y_err = transform_fnctn(
                                                tempdataset.x,
                                                tempdataset.y,
                                                tempdataset.y_err)

        # find out which points in the dataset aren't finite
        mask = ~np.isfinite(tempdataset.y)

        minimizer = ReflectivityFitFunction(tempdataset.x,
                                       tempdataset.y,
                                       params,
                                       edata=tempdataset.y_err,
                                       mask=mask)

        minimizer.transform = transform_fnctn

        if not self.settings.useerrors:
            tempdataset.y_err = None

        if self.settings.usedq:
            minimizer.set_dq(tempdataset.x_err)
        else:
            minimizer.set_dq(float(self.settings.resolution))

        progress = ProgressCallback(self, minimizer=minimizer,
                                    dataset=dataset)

        progress.show()
        minimizer.iter_cb = progress.callback2

        if alg == 'DE':
            minimizer.kws.update({'callback': progress.callback})
            minimizer.fit(method='differential_evolution')
            minimizer.kws.pop('callback')

        elif alg == 'LM':
            minimizer.fit(method='leastsq')

        elif alg == 'MCMC':
            minimizer.mcmc()

        progress.destroy()
        minimizer.iter_cb = None
        minimizer.transform = None

        #evaluate the fit and sld_profile
        dataset.fit = minimizer.model(minimizer.params)
        dataset.sld_profile = minimizer.sld_profile(minimizer.params)

        new_params = deepcopy(minimizer.params)

        curvefitter.clear_bounds(minimizer.params)
        self.params_store_model['theoretical'] = minimizer.params
        self.base_model.params = minimizer.params
        self.layer_model.params = minimizer.params
        self.layer_model.modelReset.emit()
        self.base_model.modelReset.emit()

        self.params_store_model.add(new_params, 'coef_' + dataset.name)

        #update the chi2 value
        self.update_gui_model()

        # except fitting.FitAbortedException as e:
        #     print('you aborted the fit')
        #     raise e

        print(fit_report(minimizer))
        print('___________________________________________________')

        # update GUI
        self.layer_model.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())
        self.base_model.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())

        if dataset.graph_properties.line2Dfit is None:
            self.add_datasets_to_graphs([dataset])
        else:
            self.redraw_dataset_graphs([dataset],
                                  visible=dataset.graph_properties['visible'])

        if self.ui.model_comboBox.findText('coef_' + dataset.name) < 0:
            self.ui.model_comboBox.setCurrentIndex(
                self.ui.model_comboBox.findText('coef_' + dataset.name))

    @QtCore.Slot(int)
    def on_tabWidget_currentChanged(self, arg_1):
        if arg_1 == 0:
            self.layer_model.modelReset.emit()
            self.base_model.modelReset.emit()
        if arg_1 == 2:
            self.UDFparams_model.modelReset.emit()

    @QtCore.Slot(unicode)
    def on_UDFplugin_comboBox_currentIndexChanged(self, arg_1):
        self.settings.fit_plugin = self.plugin_store_model[arg_1]

    @QtCore.Slot(unicode)
    def on_UDFmodel_comboBox_currentIndexChanged(self, arg_1):
        try:
            params = self.UDFparams_store_model[arg_1]
            theoretical = deepcopy(params)
            self.UDFparams_model.params = theoretical
            self.settings.UDFcurrent_model_name = arg_1
            self.UDFupdate_gui_model()
        except Exception as e:
            print(e)
            raise e

    @QtCore.Slot(unicode)
    def on_UDFdataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        self.settings.UDFcurrent_dataset_name = arg_1

    @QtCore.Slot()
    def on_UDFloadPlugin_clicked(self):
        self.load_plugin()

    def load_plugin(self):
        pluginFileName, ok = QtGui.QFileDialog.getOpenFileName(self,
                                                               caption='Select plugin File',
                                                               filter='Python Plugin File (*.py)')
        if not ok:
            return
        self.plugin_store_model.add(pluginFileName)

    @QtCore.Slot(unicode)
    def on_dataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        self.settings.current_dataset_name = arg_1
        self.update_gui_model()

    @QtCore.Slot(unicode)
    def on_model_comboBox_currentIndexChanged(self, arg_1):
        '''
        model selection changed, update view with parameters from model.
        '''
        self.settings.current_model_name = arg_1
        self.select_a_model(arg_1)

    def select_a_model(self, arg_1):
        try:
            params = self.params_store_model[arg_1]
            if params is not None:
                # see if it's a proper reflectivity model
                if params.values()[0] * 4 + 8 != len(params):
                    return

                theoretical = deepcopy(params)
                curvefitter.clear_bounds(params)
                self.params_store_model['theoretical'] = theoretical

            self.base_model.params = theoretical
            self.layer_model.params = theoretical

            self.base_model.modelReset.emit()
            self.layer_model.modelReset.emit()
            self.update_gui_model()
        except KeyError as e:
            return
        except IndexError as AttributeError:
            print(params)

    @QtCore.Slot(float)
    def on_res_SpinBox_valueChanged(self, arg_1):
        self.settings.resolution = arg_1

        theoretical = self.data_store_model['theoretical']
        theoretical.xdataSD = arg_1 * theoretical.xdata / 100.
        self.update_gui_model()

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
        self.update_gui_model()

    @QtCore.Slot(int)
    def on_use_dqwave_checkbox_stateChanged(self, arg_1):
        """
        """
        if arg_1:
            use = True
        else:
            use = False

        self.settings.usedq = use
        self.update_gui_model()

    @QtCore.Slot(QtCore.QModelIndex)
    def on_layerModelView_clicked(self, index):
        row = index.row()
        col = index.column()

        self.currentCell = {}

        if row == 0 and (col == 0 or col == 3):
            return

        params = self.params_store_model.params_store['theoretical']
        curvefitter.clear_bounds(params)
        self.layer_model.params = params

        nlayers = int(params['nlayers'].value)
        if row == nlayers + 1 and col == 0:
            return

        try:
            name = self.layer_model.rowcol_to_name(row, col, nlayers)
            val = params[name].value

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

        self.currentCell['name'] = name
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.layer_model

    @QtCore.Slot(QtCore.QModelIndex)
    def on_baseModelView_clicked(self, index):
        row = index.row()
        col = index.column()

        params = self.params_store_model.params_store['theoretical']
        curvefitter.clear_bounds(params)
        self.base_model.params = params

        self.currentCell = {}

        col2par = ['scale', 'bkg']
        try:
            val = params[col2par[col]]
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

        self.currentCell['name'] = col2par[col]
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.base_model

    def UDFCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()

        self.currentCell = {}

        if row == 0:
            return

        params = self.params_store_model.params_store['theoretical']
        names = curvefitter.names(params)

        try:
            name = names[row - 1]
            val = params[name].value

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

        self.currentCell['name'] = name
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.UDFModel

    @QtCore.Slot(QtCore.QModelIndex)
    def on_globalfitting_ParamsView_clicked(self, index):
        row = index.row()
        col = index.column()
        which_dataset = col // 2

        gf_settings = self.globalfitting_DataModel.gf_settings
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
        self.currentCell['model'] = self.globalfitting_ParamModel

    @QtCore.Slot(int)
    def on_horizontalSlider_valueChanged(self, arg_1):
        try:
            c = self.currentCell

            if not c['readyToChange']:
                return

            params = self.params_store_model['theoretical']

            val = c['lowlim'] + \
                (arg_1 / 1000.) * np.fabs(c['lowlim'] - c['hilim'])

            params[c['name']].value = val

            c['model'].dataChanged.emit(
                QtCore.QModelIndex(),
                QtCore.QModelIndex())
        except AttributeError:
            return
        except KeyError:
            return

    @QtCore.Slot()
    def on_horizontalSlider_sliderReleased(self):
        try:
            self.currentCell['readyToChange'] = False
            self.ui.horizontalSlider.setValue(499)
            c = self.currentCell
            params = self.params_store_model['theoretical']

            val = params[c['name']].value

            if val < 0:
                lowlim = 2 * val
                hilim = 0
            else:
                lowlim = 0
                hilim = 2 * val

            self.currentCell['val'] = params[c['name']].value
            self.currentCell['lowlim'] = lowlim
            self.currentCell['hilim'] = hilim
            self.currentCell['readyToChange'] = True

        except (ValueError, AttributeError, KeyError):
            return

    def modify_gui(self):
        # add the plots
        self.sldgraphs = MySLDGraphs(self.ui.sld)
        self.ui.gridLayout_4.addWidget(self.sldgraphs)

        self.reflectivitygraphs = MyReflectivityGraphs(self.ui.reflectivity)
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs)

        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs.mpl_toolbar)
        self.ui.gridLayout_4.addWidget(self.sldgraphs.mpl_toolbar)

        header = self.ui.layerModelView.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)

        header = self.ui.UDFmodelView.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)

    def redraw_dataset_graphs(self, datasets, visible=True, all=False,
                              transform=True):
        """ Asks the graphs to be redrawn

        Parameters
        ----------
        datasets: list
            all the datasets to be redrawn
        visible: bool
            show or hide a particular dataset
        all: bool
            redraw all the datasets

        transform: bool or curvefitter.Transform
            Transform the data in the dataset for visualisation.
            False - don't transform
            True - transform the data using the program default
            curvefitter.Transform - use a custom transform
        """

        if all:
            datasets = [dataset for dataset in self.data_store_model]

        if callable(transform):
            t = transform
        elif transform:
            transform = Transform(self.settings.transformdata)
            t = transform.transform
        else:
            t = None

        self.reflectivitygraphs.redraw_datasets(datasets, visible=visible,
                                    transform=t)
        self.sldgraphs.redraw_datasets(datasets, visible=visible)

    def UDFupdate_gui_model(self):
        params = self.UDFparams_model.params
        cur_data_name = self.settings.UDFcurrent_dataset_name
        theoretical = self.data_store_model['theoretical']

        plugin = self.settings.fit_plugin

        if issubclass(plugin, CurveFitter):
            minimizer = plugin(theoretical.xdata,
                               theoretical.ydata,
                               params)
        elif hasattr(plugin, 'fitfuncwraps'):
            minimizer = CurveFitter(plugin,
                                    theoretical.xdata,
                                    theoretical.ydata,
                                    params)
        else:
            raise TypeError('fit_plugin did not subclass CurveFitter')

        # evaluate the model against the dataset
        try:
            model = minimizer.model(params)
        except Exception as e:
            print(e)
            raise e

        theoretical.fit = model
        self.redraw_dataset_graphs([theoretical])

        if hasattr(minimizer, 'sld_profile'):
            theoretical.sld_profile = minimizer.sld_profile(params)
        else:
            theoretical.sld_profile = None

        if (cur_data_name != 'theoretical'
            and cur_data_name is not None):

            try:
                current_dataset = self.data_store_model[cur_data_name]
            except KeyError:
                return


            tempdataset = Data1D(current_dataset.data)

            kws = {}
            if self.settings.transformdata is not None:
                transform = Transform(self.settings.transformdata)
                t = transform.transform
            else:
                t = None
            if t is not None:
                tempdataset.y, tempdataset.y_err = t(tempdataset.x,
                                                   tempdataset.y,
                                                   edata=tempdataset.y_err)
                kws['transform'] = t

            kws['dqvals'] = tempdataset.x_err

            useerrors = self.settings.useerrors
            if not useerrors:
                tempdataset.y_err[:] = 1

            #mask non finite values
            mask = ~np.isfinite(tempdataset.y)

            minimizer.data = tempdataset.data
            minimizer.userkws = kws
            minimizer.mask = mask

            chisqr = np.sum(minimizer.residuals(params).ravel()**2) / (minimizer.ydata.size)

            self.ui.chi2UDF.setValue(chisqr)

    def update_gui_model(self):
        params = self.params_store_model['theoretical']

        cur_data_name = self.settings.current_dataset_name
        #self.apply_settings_to_params(params)

        minimizer = self.minimizer_store['default']
        theoretical = self.data_store_model['theoretical']

        # evaluate the model against the dataset
        theoretical.fit = minimizer.model(params)
        self.redraw_dataset_graphs([theoretical])

        if isinstance(minimizer, ReflectivityFitFunction):
            theoretical.sld_profile = minimizer.sld_profile(params)

        if (cur_data_name != 'theoretical'
            and cur_data_name is not None):

            try:
                current_dataset = self.data_store_model[cur_data_name]
            except KeyError:
                return

            res = self.settings.resolution
            usedq = self.settings.usedq
            useerrors = self.settings.useerrors

            tempdataset = Data1D(current_dataset.data)

            t = None
            if self.settings.transformdata is not None:
                transform = Transform(self.settings.transformdata)
                t = transform.transform

            if t is not None:
                tempdataset.y, tempdataset.y_err = t(
                                        tempdataset.x,
                                        tempdataset.y,
                                        edata=tempdataset.y_err)
                minimizer.transform = t

            #mask non finite values
            mask = ~np.isfinite(tempdataset.y)

            minimizer.data = tempdataset.data
            minimizer.mask = mask

            if usedq:
                minimizer.set_dq(tempdataset.x_err)
            else:
                minimizer.set_dq(float(res))

            if not useerrors:
                minimizer.edata[:] = 1

            chisqr = np.sum(minimizer.residuals(params).ravel()**2) / (minimizer.ydata.size)

            self.ui.chi2.setValue(chisqr)
            minimizer.data = theoretical.data
            minimizer.set_dq(float(res))

            minimizer.transform = None
            minimizer.mask = None

    def add_datasets_to_graphs(self, datasets):
        transform = Transform(self.settings.transformdata)
        t = transform.transform

        for dataset in datasets:
            self.reflectivitygraphs.add_dataset(dataset, transform=t)
            self.sldgraphs.add_dataset(dataset)

    @QtCore.Slot()
    def on_addGFDataSet_clicked(self):
        datasets = self.data_store_model.datastore.names

        which_dataset, ok = QtGui.QInputDialog.getItem(
            self, "Which dataset did you want to add?", "dataset", datasets, editable=False)
        if not ok:
            return

        self.globalfitting_DataModel.add_DataSet(which_dataset)

    @QtCore.Slot()
    def on_linkGFparam_clicked(self):
        select = self.ui.globalfitting_DataView.selectionModel()
        indices = select.selectedIndexes()
        self.globalfitting_DataModel.link_selection(indices)

    @QtCore.Slot()
    def on_unlinkGFparam_clicked(self):
        select = self.ui.globalfitting_DataView.selectionModel()
        indices = select.selectedIndexes()
        self.globalfitting_DataModel.unlink_selection(indices)

    @QtCore.Slot(int)
    def on_globalParamsSlider_valueChanged(self, arg_1):
        try:
            c = self.currentCell

            which_dataset = c['col'] // 2

            gf_settings = self.globalfitting_DataModel.gf_settings
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

    @QtCore.Slot()
    def on_globalParamsSlider_sliderReleased(self):
        try:
            c = self.currentCell
            c['readyToChange'] = False

            which_dataset = c['col'] // 2

            gf_settings = self.globalfitting_DataModel.gf_settings
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

    @QtCore.Slot()
    def on_do_gf_fit_clicked(self):
        print ('___________________________________________________')
        print(self.settings.fitting_algorithm, self.settings.transformdata)

        # how did you want to fit the dataset - logY vs X, lin Y vs X, etc.
        # select a transform.  Note that we have to transform the data for
        # the fit as well
        transform_fnctn = Transform(
            self.settings.transformdata).transform
        alg = self.settings.fitting_algorithm

        gf_settings = self.globalfitting_DataModel.gf_settings
        global_fitter = self.create_gf_object(use_errors=self.settings.useerrors,
                                              transform=transform_fnctn)

        if alg == 'DE':
            global_fitter.fit(method='differential_evolution')

        elif alg == 'LM':
            global_fitter.fit(method='leastsq')

        elif alg == 'MCMC':
            global_fitter.mcmc()

        #update the GUI
        self.GFupdate_gui_model()

        for name, params in gf_settings.dataset_names, gf_settings.parameters:
            new_params = deepcopy(params)
            self.params_store_model.add(new_params, 'coef_%s' % name)

        print(fit_report(minimizer))
        print('___________________________________________________')


    def GFupdate_gui_model(self):
        gf_settings = self.globalfitting_DataModel.gf_settings

        use_errors = self.settings.useerrors
        global_fitter = self.create_gf_object(use_errors=use_errors)
        residuals = global_fitter.residuals(global_fitter.params).ravel()
        chisqr = np.sum(residuals**2)
        chisqr /= residuals.size
        self.ui.chi2GF.setValue(chisqr)

        global_fitter = self.create_gf_object(transform=False,
                                              use_errors=use_errors)

        #evaluate the fit and sld_profile
        for fitter, dataset_name in zip(global_fitter.fitters, gf_settings.dataset_names):
            dataset = self.data_store_model[dataset_name]
            dataset.fit = fitter.model(fitter.params)

            if hasattr(fitter, 'sld_profile'):
                dataset.sld_profile = fitter.sld_profile(fitter.params)

        # new_params = deepcopy(minimizer.params)
        # self.UDFparams_store_model.add(new_params, 'coef_' + dataset.name)
        # #update the chi2 value
        # self.UDFupdate_gui_model()
        #
            if dataset.graph_properties.line2Dfit is None:
                self.add_datasets_to_graphs([dataset])
            else:
                self.redraw_dataset_graphs([dataset],
                                      visible=dataset.graph_properties['visible'])

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
        gf_settings = self.globalfitting_DataModel.gf_settings
        datasets = self.data_store_model.datastore

        fitters = list()

        constraints = gf_settings.linkages

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

            #have a kws dictionary
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

        global_fitter = GlobalFitter(fitters, constraints=constraints)

        return global_fitter


class ProgressCallback(QtGui.QDialog):
    def __init__(self, parent=None, minimizer=None, dataset=None):
        self.start = time.time()
        self.last_time = time.time()
        self.abort_flag = False
        super(ProgressCallback, self).__init__(parent)
        self.parent = parent
        self.ui = progressUI.Ui_progress()
        self.ui.setupUi(self)
        self.elapsed = 0.
        self.chi2 = 1.e308
        self.ui.timer.display(float(self.elapsed))
        self.ui.buttonBox.rejected.connect(self.abort)
        self.minimizer = minimizer
        self.dataset = dataset

    def abort(self):
        self.abort_flag = True

    def callback(self, xk, *args, **kwds):
        # a callback for scipy.optimize.minimize, which enters
        # every iteration.
        new_time = time.time()
        if new_time - self.last_time > 1:
            if hasattr(self.minimizer, 'transform'):
                t = self.minimizer.transform
                self.minimizer.transform = None

            self.dataset.fit = self.minimizer.model(self.minimizer.params)

            gp = self.dataset.graph_properties
            if gp.line2Dfit is not None:
                self.parent.redraw_dataset_graphs([self.dataset])
            else:
                self.parent.add_datasets_to_graphs([self.dataset])

            if hasattr(self.minimizer, 'transform'):
                self.minimizer.transform = t

            self.elapsed = new_time - self.start
            self.ui.timer.display(float(self.elapsed))
            self.last_time = new_time

        return self.abort_flag

    def callback2(self, xk, iter, resid, *fcn_args, **fcn_kws):
        # a callback for lmfit, which enters at every calculation
        # of the residuals.

        new_time = time.time()

        #we only want to update every so often
        if iter % 10:
            return

        chi2 = np.sum(resid.ravel()**2) / resid.size
        if chi2 < self.chi2:
            self.chi2 = chi2
        text = ('Function evaluations : %d\n'
                'Chi2 : %f' % (iter, self.chi2))
        self.ui.values.setPlainText(text)
        QtGui.QApplication.processEvents()


class ProgramSettings(object):

    def __init__(self, **kwds):
        _members = {'fitting_algorithm': 'DE',
                     'transformdata': 'logY',
                     'quad_order': 17,
                     'current_dataset_name': None,
                     'UDFcurrent_dataset_name': None,
                     'experiment_file_name': '',
                     'current_model_name': None,
                     'UDFcurrent_model_name': None,
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
#         if key in self.__dict__:
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
        #self.axes[0].set_yscale('log')

# residual plot
# , sharex=self.axes[0]
#         ax2 = self.figure.add_axes([0.1,0.04,0.85,0.14], sharex=ax, frame_on = False)
#         self.axes.append(ax2)
#         self.axes[1].set_visible(True)
#         self.axes[1].set_ylabel('residual')

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.canvas.setFocusPolicy( QtCore.Qt.ClickFocus )

        self.mpl_toolbar = NavToolBar(self, parent)
        self.figure.canvas.mpl_connect('pick_event', self._pick_event)
        # self.figure.canvas.mpl_connect('key_press_event', self._key_press)

        self.draw()

    def _key_press(self, event):
        #auto scale
        if event.key == 'super+a':
            self.autoscale()

    def _pick_event(self, event):
        #pick event was a double click on the graph
        if (event.mouseevent.dblclick == True
            and event.mouseevent.button == 1):
            if isinstance(event.artist, lines.Line2D):
                self.mpl_toolbar.edit_parameters()

    def autoscale(self):
        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def add_dataset(self, dataset, transform=None):
        graph_properties = dataset.graph_properties

        if graph_properties.line2D is None and dataset.name != 'theoretical':
            yt = dataset.ydata
            if transform is not None:
                yt, edata = transform(dataset.xdata, dataset.ydata, dataset.ydataSD)

            lineInstance = self.axes[0].plot(dataset.xdata,
                                             yt,
                                             markersize=3,
                                             marker='o',
                                             linestyle='',
                                             markeredgecolor=None,
                                             label=dataset.name,
                                             picker=5)
            mfc = artist.getp(lineInstance[0], 'markerfacecolor')
            artist.setp(lineInstance[0], **{'markeredgecolor': mfc})

            graph_properties.line2D = lineInstance[0]
            if graph_properties['line2D_properties']:
                artist.setp(graph_properties.line2D,
                            **graph_properties['line2D_properties'])

        if graph_properties.line2Dfit is None and dataset.fit is not None:
            yfit_t = dataset.fit
            if transform is not None:
                yfit_t, edata = transform(dataset.xdata, dataset.fit)

            yfit_t, temp = transform(dataset.xdata, dataset.fit)
            color = 'b'
            if graph_properties.line2D:
                color = artist.getp(graph_properties.line2D, 'color')
            graph_properties.line2Dfit = self.axes[0].plot(dataset.xdata,
                                                     yfit_t,
                                                     linestyle='-',
                                                     color=color,
                                                     lw=1,
                                                     label='fit_' + dataset.name,
                                                     picker=5)[0]
            if graph_properties['line2Dfit_properties']:
                artist.setp(graph_properties.line2Dfit,
                            **graph_properties['line2Dfit_properties'])

#         if dataObject.line2Dresiduals is None and dataObject.residuals is not None:
#             dataObject.line2Dresiduals = self.axes[1].plot(dataObject.x,
#                                                   dataObject.residuals,
#                                                    linestyle='-',
#                                                     lw = 1,
#                                                      label = 'residuals_' + dataObject.name)[0]
#
#             if dataObject.graph_properties['line2Dresiduals_properties']:
#                 artist.setp(dataObject.line2Dresiduals, **dataObject.graph_properties['line2Dresiduals_properties'])

        self.draw()
        graph_properties.save_graph_properties()

    def redraw_datasets(self, datasets, visible=True, transform=None):
        if not len(datasets):
            return

        for dataset in datasets:
            if not dataset:
                continue

            x = dataset.xdata

            if transform is not None:
                y, e = transform(dataset.xdata, dataset.ydata)
            else:
                y, e = dataset.ydata, dataset.ydataSD

            if dataset.fit is not None and transform is not None:
                yfit, efit = transform(dataset.xdata, dataset.fit)
            else:
                yfit = dataset.fit

            graph_properties = dataset.graph_properties

            if graph_properties.line2D:
                graph_properties.line2D.set_data(x, y)
                graph_properties.line2D.set_visible(visible)
            if graph_properties.line2Dfit:
                graph_properties.line2Dfit.set_data(x, yfit)
                graph_properties.line2Dfit.set_visible(visible)
#             if dataObject.line2Dresiduals:
#                dataObject.line2Dresiduals.set_data(dataObject.x, dataObject.residuals)
#                dataObject.line2Dresiduals.set_visible(visible)

        self.draw()

    def removeTrace(self, dataset):
        graph_properties = dataset.graph_properties
        if graph_properties.line2D:
            graph_properties.line2D.remove()
        if graph_properties.line2Dfit:
            graph_properties.line2Dfit.remove()
        if graph_properties.line2Dresiduals:
            graph_properties.line2Dresiduals.remove()
        self.draw()

    def removeTraces(self):
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

    def redraw_datasets(self, datasets, visible=True):
        for dataset in datasets:
            if not dataset:
                continue
            graph_properties = dataset.graph_properties
            if graph_properties.line2Dsld_profile and dataset.sld_profile is not None:
                graph_properties.line2Dsld_profile.set_data(
                    dataset.sld_profile[0],
                    dataset.sld_profile[1])
                graph_properties.line2Dsld_profile.set_visible(visible)
        self.axes[0].relim()
        self.axes[0].autoscale_view(None, True, True)
        self.draw()

    def add_dataset(self, dataset):
        graph_properties = dataset.graph_properties
        if (graph_properties.line2Dsld_profile is None
            and dataset.sld_profile is not None):

            color = 'r'
            lw = 2
            if graph_properties.line2D:
                color = artist.getp(graph_properties.line2D, 'color')
                lw = artist.getp(graph_properties.line2D, 'lw')

            graph_properties.line2Dsld_profile = self.axes[0].plot(
                                                dataset.sld_profile[0],
                                                dataset.sld_profile[1],
                                                linestyle='-',
                                                color=color,
                                                lw=lw,
                                                label='sld_' + dataset.name)[0]

            if graph_properties['line2Dsld_profile_properties']:
                artist.setp(graph_properties.line2Dsld_profile,
                    **graph_properties['line2Dsld_profile_properties'])

        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def removeTrace(self, dataset):
        if dataset.line2Dsld_profile:
            dataset.line2Dsld_profile.remove()
        self.draw()

    def removeTraces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]

        self.draw()


class DataSelectionChanges(QtCore.QObject):
    change = QtCore.Signal(int)

    def __init__(self):
        super(DataSelectionChanges, self).__init__()

    @QtCore.Slot(int)
    def selectionChanged(self, arg_1):
        self.change.emit(arg_1)


class EmittingStream(QtCore.QObject):
    # a class for rewriting stdout to a console window
    textWritten = QtCore.Signal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
