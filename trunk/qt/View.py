from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow

import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4'] = 'PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.artist as artist

import UDF_GUImodel
import datastore_GUImodel
import modelstore_GUImodel
import reflectivity_parameters_GUImodel
import limits_GUImodel
import globalfitting_GUImodel

import pyplatypus.analysis.model as model
import pyplatypus.analysis.reflect as reflect
import pyplatypus.analysis.fitting as fitting
import pyplatypus.analysis.globalfitting as globalfitting
from dataobject import DataObject
import limitsUI
import qrangedialogUI
import SLDcalculatorView
import os.path
from copy import deepcopy
import numpy as np
import pickle
import datastore
import math
import tempfile
import shutil
import zipfile
import os
import sys
import time


class MyMainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.errorHandler = QtGui.QErrorMessage()

        # redirect stdout to a console window
        console = EmittingStream()
        sys.stdout = console
        console.textWritten.connect(self.writeTextToConsole)

        self.dataStoreModel = datastore_GUImodel.DataStoreModel(self)
        self.dataStoreModel.dataChanged.connect(
            self.dataObjects_visibilityChanged)
        self.current_dataset = None

        self.modelStoreModel = modelstore_GUImodel.ModelStoreModel(self)
        self.pluginStoreModel = UDF_GUImodel.PluginStoreModel(self)

        self.modifyGui()

        parameters = np.array([1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 25, 3.47, 0, 3])
        fitted_parameters = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        tempq = np.linspace(0.008, 0.5, num=1000)
        tempr = np.ones_like(tempq)
        tempe = np.ones_like(tempq)
        tempdq = np.copy(tempq) * 5 / 100.
        dataTuple = (tempq, tempr, tempe, tempdq)

        self.current_dataset = None

        self.theoretical = DataObject(dataTuple=dataTuple)

        theoreticalmodel = model.Model(
            parameters=parameters,
            fitted_parameters=fitted_parameters)
        self.modelStoreModel.add(theoreticalmodel, 'theoretical')

        self.baseModel = reflectivity_parameters_GUImodel.BaseModel(
            self.modelStoreModel.modelStore['theoretical'],
            self)
        self.layerModel = reflectivity_parameters_GUImodel.LayerModel(
            self.modelStoreModel.modelStore['theoretical'],
            self)
        self.UDFmodel = UDF_GUImodel.UDFParametersModel(
            self.modelStoreModel.modelStore['theoretical'],
            self)

        # holds miscellaneous information on program settings
        self.settings = ProgramSettings()
        self.settings.fitPlugin = self.pluginStoreModel.plugins[0]
        self.restoreSettings()
        
        self.theoretical.evaluate_model(theoreticalmodel, store=True)
        self.dataStoreModel.add(self.theoretical)

        self.theoretical.line2Dfit = self.reflectivitygraphs.axes[0].plot(
            self.theoretical.xdata,
            self.theoretical.fit, color='b',
            linestyle='-', lw=2, label='theoretical')[0]

        self.theoretical.line2Dsld_profile = self.sldgraphs.axes[0].plot(
            self.theoretical.sld_profile[0],
            self.theoretical.sld_profile[
                1],
            linestyle='-', color='b')[0]

        self.redraw_dataObject_graphs([self.theoretical])

        self.ui.dataOptions_tableView.setModel(self.dataStoreModel)

        # combo boxes for the dataset selection
        self.ui.dataset_comboBox.setModel(self.dataStoreModel)
        self.ui.dataset_comboBox.setModelColumn(0)
        self.ui.UDFdataset_comboBox.setModel(self.dataStoreModel)
        self.ui.UDFdataset_comboBox.setModelColumn(0)
        self.changer = DataSelectionChanges()
        self.ui.dataset_comboBox.currentIndexChanged.connect(
            self.changer.selectionChanged)
        self.ui.UDFdataset_comboBox.currentIndexChanged.connect(
            self.changer.selectionChanged)
        self.changer.change.connect(self.ui.dataset_comboBox.setCurrentIndex)
        self.changer.change.connect(
            self.ui.UDFdataset_comboBox.setCurrentIndex)

        self.ui.model_comboBox.setModel(self.modelStoreModel)

        self.ui.baseModelView.setModel(self.baseModel)
        self.ui.layerModelView.setModel(self.layerModel)
        self.baseModel.layersAboutToBeInserted.connect(
            self.layerModel.layersAboutToBeInserted)
        self.baseModel.layersAboutToBeRemoved.connect(
            self.layerModel.layersAboutToBeRemoved)
        self.baseModel.layersFinishedBeingInserted.connect(
            self.layerModel.layersFinishedBeingInserted)
        self.baseModel.layersFinishedBeingRemoved.connect(
            self.layerModel.layersFinishedBeingRemoved)
        self.layerModel.dataChanged.connect(self.update_gui_modelChanged)
        self.baseModel.dataChanged.connect(self.update_gui_modelChanged)
        self.ui.baseModelView.clicked.connect(self.baseCurrentCellChanged)
        self.ui.layerModelView.clicked.connect(self.layerCurrentCellChanged)

        # User defined function tab
#         self.ui.UDFmodelView.clicked.connect(self.UDFCurrentCellChanged)
        self.ui.UDFmodelView.setModel(self.UDFmodel)
        self.UDFmodel.dataChanged.connect(self.update_gui_modelChanged)
        self.ui.UDFplugin_comboBox.setModel(self.pluginStoreModel)
        self.ui.UDFmodel_comboBox.setModel(self.modelStoreModel)

        # globalfitting tab
        self.globalfitting_DataModel = globalfitting_GUImodel.GlobalFitting_DataModel(
            self)
        self.globalfitting_ParamModel = globalfitting_GUImodel.GlobalFitting_ParamModel(
            self)

        self.ui.globalfitting_DataView.setModel(self.globalfitting_DataModel)
        self.ui.gfparams_tableView.setModel(self.globalfitting_ParamModel)

        self.ui.FitPluginDelegate = globalfitting_GUImodel.FitPluginItemDelegate(
            self.pluginStoreModel.plugins,
            self.ui.gfparams_tableView)
        self.ui.globalfitting_DataView.setEditTriggers(
            QtGui.QAbstractItemView.AllEditTriggers)
        self.ui.globalfitting_DataView.setItemDelegateForRow(
            1,
            self.ui.FitPluginDelegate)

        self.globalfitting_DataModel.changed_linkages.connect(
            self.globalfitting_ParamModel.changed_linkages)
        self.globalfitting_DataModel.added_DataSet.connect(
            self.globalfitting_ParamModel.added_DataSet)
        self.globalfitting_DataModel.removed_DataSet.connect(
            self.globalfitting_ParamModel.removed_DataSet)
        self.globalfitting_DataModel.added_params.connect(
            self.globalfitting_ParamModel.added_params)
        self.globalfitting_DataModel.removed_params.connect(
            self.globalfitting_ParamModel.removed_params)
        self.globalfitting_DataModel.resized_rows.connect(
            self.globalfitting_ParamModel.resized_rows)
        self.globalfitting_DataModel.changed_fitplugin.connect(
            self.globalfitting_ParamModel.changed_fitplugin)
        self.globalfitting_ParamModel.dataChanged.connect(
            self.calculate_gf_model)

        print 'Session started at:', time.asctime(time.localtime(time.time()))

    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__

    def dataObjects_visibilityChanged(self, arg_1, arg_2):
        if arg_1.column() < 0:
            return

        name = self.dataStoreModel.dataStore.names[arg_1.row()]
        dataObject = self.dataStoreModel.dataStore[name]
        if dataObject.line2D is not None:
            dataObject.line2D.set_visible(
                dataObject.graph_properties['visible'])
        self.redraw_dataObject_graphs(
            [dataObject],
            visible=dataObject.graph_properties['visible'])

    @QtCore.Slot(QtGui.QDropEvent)
    def dropEvent(self, event):
        m = event.mimeData()
        urls = m.urls()
        for url in urls:
            try:
                self.loadData([url.toLocalFile()])
                continue
            except Exception as inst:
                pass

            try:
                self.loadModel(url.toLocalFile())
                continue
            except Exception:
                pass

            try:
                self.pluginStoreModel.add(url.toLocalFile())
                continue
            except Exception:
                pass

            try:
                self.__restoreState(url.toLocalFile())
                continue
            except Exception:
                pass

    @QtCore.Slot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event):
        m = event.mimeData()
        if m.hasUrls():
            event.acceptProposedAction()

    def writeTextToConsole(self, text):
        self.ui.plainTextEdit.moveCursor(QtGui.QTextCursor.End)
        self.ui.plainTextEdit.insertPlainText(text)

    def __saveState(self, experimentFileName):
        state = {}
        state['dataStoreModel.dataStore'] = self.dataStoreModel.dataStore
        state['modelStoreModel.modelStore'] = self.modelStoreModel.modelStore
        state['history'] = self.ui.plainTextEdit.toPlainText()
        state['settings'] = self.settings
        state['plugins'] = self.pluginStoreModel.plugins
        print state['plugins']

        try:
            tempdirectory = tempfile.mkdtemp()

            with open(os.path.join(tempdirectory, 'state'), 'wb') as f:
                pickle.dump(state, f, -1)

            with zipfile.ZipFile(experimentFileName, 'w') as zip:
                datastore.zipper(tempdirectory, zip)
        except Exception as e:
            print type(e), e.message
        finally:
            shutil.rmtree(tempdirectory)

    @QtCore.Slot()
    def on_actionSave_File_triggered(self):
        experimentFileName, ok = QtGui.QFileDialog.getSaveFileName(
            self, caption='Save experiment as:', dir='experiment.fdob')

        if not ok:
            return

        path, ext = os.path.splitext(experimentFileName)
        if ext != '.fdob':
            experimentFileName = path + '.fdob'

        self.__saveState(experimentFileName)

    def __restoreState(self, experimentFileName):
        state = None
        try:
            tempdirectory = tempfile.mkdtemp()
            with zipfile.ZipFile(experimentFileName, 'r') as zip:
                zip.extractall(tempdirectory)

            with open(os.path.join(tempdirectory, 'state'), 'rb') as f:
                state = pickle.load(f)
        except Exception as e:
            print type(e), e.message
        finally:
            shutil.rmtree(tempdirectory)

        if not state:
            print "Couldn't load experiment"
            return

        try:
            self.dataStoreModel.dataStore = state['dataStoreModel.dataStore']
            self.modelStoreModel.modelStore = state[
                'modelStoreModel.modelStore']
            self.ui.plainTextEdit.setPlainText(state['history'])
            self.settings = state['settings']
            self.pluginStoreModel.plugins = state['plugins']
        except KeyError as e:
            print type(e), e.message
            return

        self.dataStoreModel.modelReset.emit()
        self.modelStoreModel.modelReset.emit()

        dataStore = self.dataStoreModel.dataStore
        self.restoreSettings()

        # remove and add dataObjectsToGraphs
        self.reflectivitygraphs.removeTraces()
        self.sldgraphs.removeTraces()
        self.add_dataObjectsToGraphs(dataStore)

        self.theoretical = dataStore['theoretical']
#        self.reflectivitygraphs.axes[0].lines.remove(self.theoretical.line2D)
# self.reflectivitygraphs.axes[1].lines.remove(self.theoretical.line2Dresiduals)

        # when you load in the theoretical model you destroy the link to the
        # gui, reinstate it.
        self.theoreticalmodel = self.modelStoreModel.modelStore['theoretical']
        self.baseModel.model = self.theoreticalmodel
        self.layerModel.model = self.theoreticalmodel
        self.theoretical.evaluate_model(self.theoreticalmodel, store=True)

        self.theoretical.line2Dfit = self.reflectivitygraphs.axes[0].plot(
            self.theoretical.xdata,
            self.theoretical.fit, color='b',
            linestyle='-', lw=2, label='theoretical')[0]
        self.reflectivitygraphs.draw()

    def restoreSettings(self):
        '''
            applies the program settings to the GUI
        '''

        dataStore = self.dataStoreModel.dataStore
        modelStore = self.modelStoreModel.modelStore

        try:
            self.current_dataset = dataStore[
                self.settings.current_dataset_name]
            self.ui.dataset_comboBox.setCurrentIndex(
                dataStore.names.index(self.current_dataset.name))
            self.ui.model_comboBox.setCurrentIndex(
                modelStore.names.index(self.settings.current_model_name))
        except AttributeError as KeyError:
            pass

        self.ui.res_SpinBox.setValue(self.settings.resolution)
        self.ui.use_dqwave_checkbox.setChecked(self.settings.usedq)
        self.ui.use_errors_checkbox.setChecked(self.settings.useerrors)
        self.ui.actionLevenberg_Marquardt.setChecked(False)
        self.ui.actionDifferential_Evolution.setChecked(False)
        if self.settings.fittingAlgorithm == 'LM':
            self.ui.actionLevenberg_Marquardt.setChecked(True)
        elif self.settings.fittingAlgorithm == 'DE':
            self.ui.actionDifferential_Evolution.setChecked(True)

        self.settransformoption(self.settings.transform)

    @QtCore.Slot()
    def on_actionLoad_File_triggered(self):
        experimentFileName, ok = QtGui.QFileDialog.getOpenFileName(self,
                                                                   caption='Select Experiment File',
                                                                   filter='Experiment Files (*.fdob)')
        if not ok:
            return

        self.__restoreState(experimentFileName)

    def loadData(self, files):
        for file in files:
            dataObject = self.dataStoreModel.load(file)
            self.add_dataObjectsToGraphs([dataObject])

    @QtCore.Slot()
    def on_actionLoad_Data_triggered(self):
        """
            you load data
        """
        files, ok = QtGui.QFileDialog.getOpenFileNames(
            self, caption='Select Reflectivity Files')
        if not ok:
            return

        self.loadData(files)

    @QtCore.Slot()
    def on_actionRemove_Data_triggered(self):
        """
            you remove data
        """
        datasets = list(self.dataStoreModel.dataStore.names)
        del(datasets[datasets.index('theoretical')])
        which_dataset, ok = QtGui.QInputDialog.getItem(
            self, "Which fit did you want to remove?", "dataset", datasets, editable=False)

        if not ok:
            return
        self.reflectivitygraphs.removeTrace(
            self.dataStoreModel.dataStore[which_dataset])
        self.dataStoreModel.remove(which_dataset)

    @QtCore.Slot()
    def on_actionSave_Fit_triggered(self):
        fits = []
        for dataObject in self.dataStoreModel:
            if dataObject.fit is not None:
                fits.append(dataObject.name)

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
                    self.dataStoreModel.dataStore[fit].savefit(
                        os.path.join(folder[0], 'fit_' + fit + '.dat'))
        else:
            fitFileName, ok = QtGui.QFileDialog.getSaveFileName(
                self, caption='Save fit as:', dir='fit_' + which_fit)
            if not ok:
                return
            self.dataStoreModel.dataStore[which_fit].savefit(fitfilename)

    def loadModel(self, fileName):
        with open(fileName, 'Ur') as f:
            themodel = model.Model(None, file=f)

        modelName = os.path.basename(fileName)
        self.modelStoreModel.modelStore.add(
            themodel,
            os.path.basename(modelName))
        if self.ui.model_comboBox.count() == 1:
            self.ui.model_comboBox.setCurrentIndex(-1)

    @QtCore.Slot()
    def on_actionLoad_Model_triggered(self):
        # load a model
        modelFileName, ok = QtGui.QFileDialog.getOpenFileName(
            self, caption='Select Model File')
        if not ok:
            return

        self.loadModel(modelFileName)

    @QtCore.Slot()
    def on_actionSave_Model_triggered(self):
        # save a model
        # which model are you saving?
        listofmodels = list(self.modelStoreModel.modelStore.names)
        listofmodels.append('-all-')

        which_model, ok = QtGui.QInputDialog.getItem(
            self, "Which model did you want to save?", "model", listofmodels, editable=False)
        if not ok:
            return

        if which_model == '-all-':
            dialog = QtGui.QFileDialog(self)
            dialog.setFileMode(QtGui.QFileDialog.Directory)
            if dialog.exec_():
                folder = dialog.selectedFiles()
                self.modelStoreModel.modelStore.saveModelStore(folder[0])
        else:
            themodel = self.modelStoreModel.modelStore[which_model]
            modelFileName, ok = QtGui.QFileDialog.getSaveFileName(
                self, caption='Save model as:', dir=which_model)
            if not ok:
                return

            with open(modelFileName, 'w+') as f:
                themodel.save(f)

    @QtCore.Slot()
    def on_actionDifferential_Evolution_triggered(self):
        if self.ui.actionDifferential_Evolution.isChecked():
            self.settings.fittingAlgorithm = 'DE'
            self.ui.actionLevenberg_Marquardt.setChecked(False)

    @QtCore.Slot()
    def on_actionLevenberg_Marquardt_triggered(self):
        if self.ui.actionLevenberg_Marquardt.isChecked():
            self.settings.fittingAlgorithm = 'LM'
            self.ui.actionDifferential_Evolution.setChecked(False)

    def change_Q_range(self, qmin, qmax, numpnts, res):
        theoretical = self.dataStoreModel.dataStore['theoretical']

        theoretical.xdata = np.linspace(qmin, qmax, numpnts)
        theoretical.ydata = np.resize(theoretical.ydata, numpnts)
        theoretical.ydataSD = np.resize(theoretical.ydataSD, numpnts)
        theoretical.xdataSD = theoretical.xdata * res / 100.

        self.update_gui_modelChanged()

    @QtCore.Slot()
    def on_actionChange_Q_range_triggered(self):
        theoretical = self.dataStoreModel.dataStore['theoretical']
        qmin = theoretical.xdata[0]
        qmax = theoretical.xdata[-1]
        numpnts = len(theoretical.xdata)

        qrangedialog = QtGui.QDialog()
        qrangeGUI = qrangedialogUI.Ui_qrange()
        qrangeGUI.setupUi(qrangedialog)
        qrangeGUI.numpnts.setValue(numpnts)
        qrangeGUI.qmin.setValue(qmin)
        qrangeGUI.qmax.setValue(qmax)

        res = self.ui.res_SpinBox.value()

        ok = qrangedialog.exec_()
        if ok:
            self.change_Q_range(qrangeGUI.qmin.value(),
                                qrangeGUI.qmax.value(),
                                qrangeGUI.numpnts.value(), res)

    @QtCore.Slot()
    def on_actionTake_Snapshot_triggered(self):
        snapshotname, ok = QtGui.QInputDialog.getText(self,
                                                      'Take a snapshot',
                                                      'snapshot name')
        if not ok:
            return

        self.dataStoreModel.snapshot(snapshotname)
        self.add_dataObjectsToGraphs(
            [self.dataStoreModel.dataStore[snapshotname]])

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
        self.update_gui_modelChanged()

    @QtCore.Slot()
    def on_actionLoad_Plugin_triggered(self):
        # load a model plugin
        self.loadPlugin()

    @QtCore.Slot()
    def on_actionBatch_Fit_triggered(self):
        if self.dataStoreModel.dataStore.numDataObjects < 2:
            msgBox = QtGui.QMessageBox()
            msgBox.setText("You have no loaded datasets")
            msgBox.exec_()
            return

        theoreticalmodel = self.modelStoreModel.modelStore['theoretical']
        theoreticalmodel.default_limits()
        if self.settings.fittingAlgorithm != 'LM':
            ok, limits = self.get_limits(theoreticalmodel.parameters,
                                         theoreticalmodel.fitted_parameters,
                                         theoreticalmodel.limits)

            if not ok:
                return

            theoreticalmodel.limits = np.copy(limits)

#       Have to iterate over list because dataObjects are stored in dictionary
        for name in self.dataStoreModel.dataStore.names:
            if name == 'theoretical':
                continue
            self.do_a_fit_and_add_to_gui(
                self.dataStoreModel.dataStore[name],
                theoreticalmodel)

    @QtCore.Slot()
    def on_actionRefresh_Data_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.dataStoreModel.dataStore.refresh()
        for dataObject in self.dataStoreModel.dataStore:
            if dataObject.line2D:
                dataObject.line2D.set_data(dataObject.xdata, dataObject.ydata)
        self.reflectivitygraphs.draw()

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
        self.settings.transform = transform

    @QtCore.Slot()
    def on_actionSLD_calculator_triggered(self):
        SLDcalculator = SLDcalculatorView.SLDcalculatorView(self)
        SLDcalculator.show()

    def get_limits(self, parameters, fitted_parameters, limits):

        limitsdialog = QtGui.QDialog()
        limitsGUI = limitsUI.Ui_Dialog()
        limitsGUI.setupUi(limitsdialog)

        limitsModel = limits_GUImodel.LimitsModel(
            parameters,
            fitted_parameters,
            limits)
        limitsGUI.limits.setModel(limitsModel)
        header = limitsGUI.limits.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)

        ok = limitsdialog.exec_()
        limits = limitsModel.limits[:]

        return ok, limits

    @QtCore.Slot()
    def on_do_fit_button_clicked(self):
        '''
            you should do a fit
        '''
        if self.current_dataset is None or self.current_dataset.name == 'theoretical':
            msgBox = QtGui.QMessageBox()
            msgBox.setText("Please select a dataset to fit.")
            msgBox.exec_()
            return

        theoreticalmodel = self.modelStoreModel.modelStore['theoretical']
        alreadygotlimits = False

        if ('coef_' + self.current_dataset.name) in self.modelStoreModel.modelStore.names:
            model = self.modelStoreModel.modelStore[
                'coef_' +
                self.current_dataset.name]
            if model.limits is not None and model.limits.ndim == 2 and np.size(theoreticalmodel.parameters) == np.size(model.limits, 1):
                theoreticalmodel.limits = np.copy(model.limits)
                alreadygotlimits = True

        if not alreadygotlimits:
            theoreticalmodel.default_limits(True)

        if self.settings.fittingAlgorithm != 'LM':
            ok, limits = self.get_limits(theoreticalmodel.parameters,
                                         theoreticalmodel.fitted_parameters,
                                         theoreticalmodel.limits)

            if not ok:
                return

            theoreticalmodel.limits = np.copy(limits)

        self.do_a_fit_and_add_to_gui(self.current_dataset, theoreticalmodel)

    @QtCore.Slot()
    def on_do_UDFfit_button_clicked(self):
        """
            you should do a fit using the fit plugin
        """
        if self.current_dataset is None or self.current_dataset.name == 'theoretical':
            msgBox = QtGui.QMessageBox()
            msgBox.setText("Please select a dataset to fit.")
            msgBox.exec_()
            return

        theoreticalmodel = self.modelStoreModel.modelStore['theoretical']
        alreadygotlimits = False

        if ('coef_' + self.current_dataset.name) in self.modelStoreModel.modelStore.names:
            model = self.modelStoreModel.modelStore[
                'coef_' +
                self.current_dataset.name]
            if model.limits is not None and model.limits.ndim == 2 and np.size(theoreticalmodel.parameters) == np.size(model.limits, 1):
                theoreticalmodel.limits = np.copy(model.limits)
                alreadygotlimits = True

        if not alreadygotlimits:
            theoreticalmodel.default_limits()

        if self.settings.fittingAlgorithm != 'LM':
            ok, limits = self.get_limits(theoreticalmodel.parameters,
                                         theoreticalmodel.fitted_parameters,
                                         theoreticalmodel.limits)

            if not ok:
                return

            theoreticalmodel.limits = np.copy(limits)

        self.do_a_fit_and_add_to_gui(self.current_dataset,
                                     theoreticalmodel,
                                     fitPlugin=self.settings.fitPlugin['rfo'])

    def do_a_fit_and_add_to_gui(self, dataset, model, fitPlugin=None):
        if dataset.name == 'theoretical':
            print 'You tried to fit the theoretical dataset'
            return

        print '___________________________________________________'
        print 'fitting to:', dataset.name
        try:
            print self.settings.fittingAlgorithm
            print self.settings.transform

            # how did you want to fit the dataset - logY vs X, lin Y vs X, etc.
            # select a transform.  Note that we have to transform the data for
            # the fit as well
            transform_fnctn = reflect.Transform(
                self.settings.transform).transform
            tempdataset = DataObject(dataset.get_data())
            tempdataset.ydata, tempdataset.ydataSD = transform_fnctn(
                tempdataset.xdata,
                tempdataset.ydata,
                tempdataset.ydataSD)

            model.quad_order = self.settings.quad_order
            model.fitPlugin = fitPlugin
            model.useerrors = self.ui.use_errors_checkbox.isChecked()
            model.transform = transform_fnctn
            if not self.settings.useerrors:
                tempdataset.ydataSD = None

            tempdataset.do_a_fit(model,
                                 fitPlugin=fitPlugin,
                                 method=self.settings.fittingAlgorithm)

            model.transform = None

            # but then evaluate against the untransformed data
            dataset.evaluate_model(model, store=True, fitPlugin=fitPlugin)

        except fitting.FitAbortedException as e:
            print 'you aborted the fit'
            raise e

        print 'Chi2 :', tempdataset.chi2 / tempdataset.numpoints
        np.set_printoptions(suppress=True, precision=4)
        print 'parameters:'
        print model.parameters
        print 'uncertainties:'
        print model.uncertainties
        print '___________________________________________________'

        newmodel = Model.Model(parameters=np.copy(model.parameters),
                               fitted_parameters=np.copy(
                                   model.fitted_parameters),
                               uncertainties=np.copy(model.uncertainties),
                               covariance=np.copy(model.covariance),
                               limits=np.copy(model.limits),
                               fitPlugin=model.fitPlugin,
                               useerrors=model.useerrors,
                               usedq=model.usedq,
                               quad_order=model.quad_order)

        self.modelStoreModel.add(newmodel, 'coef_' + dataset.name)

        # update GUI
        self.layerModel.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())
        self.baseModel.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())
        self.UDFmodel.dataChanged.emit(
            QtCore.QModelIndex(),
            QtCore.QModelIndex())

        self.add_dataObjectsToGraphs([dataset])

        if self.ui.model_comboBox.findText('coef_' + dataset.name) < 0:
            self.ui.model_comboBox.setCurrentIndex(
                self.ui.model_comboBox.findText('coef_' + dataset.name))
        self.update_gui_modelChanged()
        self.redraw_dataObject_graphs(
            [dataset],
            visible=dataset.graph_properties['visible'])

    @QtCore.Slot(int)
    def on_tabWidget_currentChanged(self, arg_1):
        if arg_1 == 0:
            self.layerModel.modelReset.emit()
            self.baseModel.modelReset.emit()
        if arg_1 == 2:
            self.UDFmodel.modelReset.emit()

    @QtCore.Slot(int)
    def on_UDFplugin_comboBox_currentIndexChanged(self, arg_1):

        if arg_1 == 0:
            # the default reflectometry calculation is being used
            self.modelStoreModel.modelStore.displayOtherThanReflect = False

            '''
                you need to display a model suitable for reflectometry
            '''
            areAnyModelsValid = [
                reflect.is_proper_Abeles_input(
                    model.parameters) for model in self.modelStoreModel.modelStore]
            try:
                idx = areAnyModelsValid.index(True)
                self.select_a_model(self.modelStoreModel.modelStore.names[idx])
            except ValueError:
                # none are valid, reset the theoretical model
                parameters = np.array(
                    [1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 25, 3.47, 0, 3])
                fitted_parameters = np.array(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
                self.modelStoreModel.modelStore[
                    'theoretical'].parameters = parameters[
                    :]
                self.modelStoreModel.modelStore[
                    'theoretical'].fitted_parameters = fitted_parameters[
                    :]
                self.modelStoreModel.modelStore[
                    'theoretical'].default_limits(True)

                self.UDFmodel.modelReset.emit()
        else:
            # a user plugin is being used.
            self.modelStoreModel.modelStore.displayOtherThanReflect = True
        self.settings.fitPlugin = self.pluginStoreModel.plugins[arg_1]

    @QtCore.Slot(unicode)
    def on_UDFmodel_comboBox_currentIndexChanged(self, arg_1):
        self.select_a_model(arg_1)

    @QtCore.Slot(unicode)
    def on_UDFdataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        self.current_dataset = self.dataStoreModel.dataStore[arg_1]

    @QtCore.Slot()
    def on_UDFloadPlugin_clicked(self):
        self.loadPlugin()

    def loadPlugin(self):
        pluginFileName, ok = QtGui.QFileDialog.getOpenFileName(self,
                                                               caption='Select plugin File',
                                                               filter='Python Plugin File (*.py)')
        if not ok:
            return
        self.pluginStoreModel.add(pluginFileName)

    @QtCore.Slot(unicode)
    def on_dataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        self.current_dataset = self.dataStoreModel.dataStore[arg_1]
        self.settings.current_dataset_name = arg_1
        self.update_gui_modelChanged()

    @QtCore.Slot(unicode)
    def on_model_comboBox_currentIndexChanged(self, arg_1):
        '''
        model selection changed, update view with parameters from model.
        '''
        self.settings.current_model_name = arg_1
        self.select_a_model(arg_1)

    def select_a_model(self, arg_1):
        try:
            model = self.modelStoreModel.modelStore.models[arg_1]
            if model.parameters is not None:
                self.modelStoreModel.modelStore[
                    'theoretical'].parameters = model.parameters[
                    :]
            if model.fitted_parameters is not None:
                self.modelStoreModel.modelStore[
                    'theoretical'].fitted_parameters = model.fitted_parameters[
                    :]

            if model.limits is not None and model.limits.ndim == 2 and np.size(model.limits, 1) == np.size(model.parameters):
                self.modelStoreModel.modelStore[
                    'theoretical'].limits = np.copy(model.limits)
            else:
                self.modelStoreModel.modelStore[
                    'theoretical'].default_limits(True)

            self.baseModel.dataChanged.emit(
                QtCore.QModelIndex(),
                QtCore.QModelIndex())
            self.layerModel.dataChanged.emit(
                self.layerModel.createIndex(0,
                                            0),
                self.layerModel.createIndex(2 + int(model.parameters[0]),
                                            3))
            self.layerModel.modelReset.emit()
            self.UDFmodel.modelReset.emit()

            self.update_gui_modelChanged()
        except KeyError as e:
            return
        except IndexError as AttributeError:
            print model.parameters, model.fitted_parameters

    @QtCore.Slot(float)
    def on_res_SpinBox_valueChanged(self, arg_1):
        self.settings.resolution = arg_1
        self.theoretical.xdataSD = arg_1 * self.theoretical.xdata / 100.

        self.update_gui_modelChanged()

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
        self.update_gui_modelChanged()

    @QtCore.Slot(int)
    def on_use_dqwave_checkbox_stateChanged(self, arg_1):
        """
        """
        if arg_1:
            use = True
        else:
            use = False

        self.settings.usedq = use
        self.update_gui_modelChanged()

    def layerCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()

        self.currentCell = {}

        if row == 0 and (col == 0 or col == 3):
            return

        theoreticalmodel = self.modelStoreModel.modelStore['theoretical']

        if row == int(theoreticalmodel.parameters[0]) + 1 and col == 0:
            return

        try:
            param = self.layerModel.rowcoltoparam(
                row,
                col,
                int(theoreticalmodel.parameters[0]))
            val = theoreticalmodel.parameters[param]
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

        self.currentCell['param'] = param
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.layerModel

    def baseCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()

        theoreticalmodel = self.modelStoreModel.modelStore['theoretical']

        self.currentCell = {}

        if col == 0:
            return

        col2par = [0, 1, 6]
        try:
            val = theoreticalmodel.parameters[col2par[col]]
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

        self.currentCell['param'] = col2par[col]
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.baseModel

    def UDFCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()

        self.currentCell = {}

        if row == 0:
            return

        theoreticalmodel = self.modelStoreModel.modelStore['theoretical']

        try:
            param = row - 1
            val = theoreticalmodel.parameters[row - 1]
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

        self.currentCell['param'] = param
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        self.currentCell['model'] = self.UDFModel

    @QtCore.Slot(int)
    def on_horizontalSlider_valueChanged(self, arg_1):
        try:
            c = self.currentCell

            if not c['readyToChange']:
                return

            theoreticalmodel = self.modelStoreModel.modelStore['theoretical']

            val = c['lowlim'] + \
                (arg_1 / 1000.) * math.fabs(c['lowlim'] - c['hilim'])

            param = c['param']
            theoreticalmodel.parameters[c['param']] = val

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
            parameters = self.modelStoreModel.modelStore[
                'theoretical'].parameters

            val = parameters[c['param']]

            if val < 0:
                lowlim = 2 * val
                hilim = 0
            else:
                lowlim = 0
                hilim = 2 * val

            self.currentCell['val'] = parameters[c['param']]
            self.currentCell['lowlim'] = lowlim
            self.currentCell['hilim'] = hilim
            self.currentCell['readyToChange'] = True

        except (ValueError, AttributeError, KeyError):
            return

    def modifyGui(self):
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

    def redraw_dataObject_graphs(self, dataObjects, visible=True):
        self.reflectivitygraphs.redraw_dataObjects(
            dataObjects,
            visible=visible)
        self.sldgraphs.redraw_dataObjects(dataObjects, visible=visible)

    def update_gui_modelChanged(self):
        model = self.modelStoreModel.modelStore['theoretical']
        fitPlugin = self.settings.fitPlugin['rfo']

        # evaluate the model against the dataset
        try:
            self.theoretical.evaluate_model(
                model,
                store=True,
                fitPlugin=fitPlugin)
            if self.current_dataset is not None and self.current_dataset.name != 'theoretical':
                transform_fnctn = reflect.Transform(
                    self.settings.transform).transform
                tempdataset = DataObject(self.current_dataset.get_data())
                tempdataset.ydata, tempdataset.ydataSD = transform_fnctn(
                    tempdataset.xdata,
                    tempdataset.ydata,
                    tempdataset.ydataSD)

                model.useerrors = self.settings.useerrors
                model.quad_order = self.settings.quad_order
                model.transform = transform_fnctn
                if not model.useerrors:
                    tempdataset.ydataSD = None

                energy = tempdataset.evaluate_chi2(
                    model,
                    fitPlugin=fitPlugin)
                model.transform = None
                self.ui.chi2.setText(str(round(energy, 3)))
                self.ui.chi2UDF.setText(str(round(energy, 3)))

            self.redraw_dataObject_graphs([self.theoretical])

        except ValueError:
            print 'The model parameters were not correct for the type of fitting plugin you requested'

    def add_dataObjectsToGraphs(self, dataObjects):
        for dataObject in dataObjects:
            self.reflectivitygraphs.add_dataObject(dataObject)
            self.sldgraphs.add_dataObject(dataObject)

    @QtCore.Slot()
    def on_addGFDataSet_clicked(self):
        datasets = self.dataStoreModel.dataStore.names

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

    @QtCore.Slot()
    def on_do_gf_fit_clicked(self):
        print globalfitting_GUImodel.generate_linkage_matrix(
            self.globalfitting_DataModel.linkages,
            self.globalfitting_DataModel.numparams)

    def calculate_gf_model(self):
        self.create_gf_object()

    def create_gf_object(self):
        datamodel = self.globalfitting_DataModel
        parammodel = self.globalfitting_ParamModel
        fitObjects = list()

        linkageArray = \
            globalfitting_GUImodel.generate_linkage_matrix(datamodel.linkages,
                                                           datamodel.numparams)

        for idx, dataset in enumerate(datamodel.dataset_names):
            # retrieve the dataobject
            dataobject = self.dataStoreModel.dataStore[dataset]
            # get the parameters
            model = parammodel.models[idx]

            callerInfo = {'xdata': dataobject.xdata,
                          'ydata': dataobject.ydata,
                          'edata': dataobject.ydataSD,
                          'parameters': model.parameters,
                          'fitted_parameters': model.fitted_parameters,
                          'dqvals': dataobject.xdataSD
                          }

            # retrieve the required fitplugin from the pluginStoreModel
            fitClass = self.pluginStoreModel[datamodel.fitplugins[idx]]['rfo']

            fitObject = fitClass(**callerInfo)
            fitObjects.append(fitObject)

        globalFitting = globalfitting.GlobalFitObject(tuple(fitObjects),
                                                      linkageArray)


class ProgramSettings(object):

    def __init__(self, **kwds):
        __members = {'fittingAlgorithm': 'DE',
                     'transform': 'logY',
                     'quad_order': 17,
                     'current_dataset_name': None,
                     'current_model_name': None,
                     'usedq': True,
                     'resolution': 5,
                     'fitPlugin': None,
                     'useerrors': True}

        for key in __members:
            if key in kwds:
                setattr(self, key, kwds[key])
            else:
                setattr(self, key, __members[key])

    def __getitem__(self, key):
        if key in self.__members:
            return self.key

    def __setitem__(self, key, value):
        if key in self.__members:
            setattr(self, key, value)


class MyReflectivityGraphs(FigureCanvas):

    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        # reflectivity graph
        self.axes = []
        ax = self.figure.add_axes([0.1, 0.15, 0.85, 0.8])
        self.axes.append(ax)

        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.axes[0].set_xlabel('Q')
        self.axes[0].set_ylabel('R')
        self.axes[0].set_yscale('log')

# residual plot
# , sharex=self.axes[0]
#         ax2 = self.figure.add_axes([0.1,0.04,0.85,0.14], sharex=ax, frame_on = False)
#         self.axes.append(ax2)
#         self.axes[1].set_visible(True)
#         self.axes[1].set_ylabel('residual')

        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
#        self.figure.subplots_adjust(left=0.1, right=0.95, top = 0.98)
        self.mpl_toolbar = NavigationToolbar(self, parent)
        self.draw()

    def add_dataObject(self, dataObject):
        if dataObject.name == 'theoretical':
            return

        if dataObject.line2D is None:
            lineInstance = self.axes[0].plot(dataObject.xdata,
                                             dataObject.ydata,
                                             markersize=5,
                                             marker='o',
                                             linestyle='',
                                             label=dataObject.name)
            dataObject.line2D = lineInstance[0]
            if dataObject.graph_properties['line2D_properties']:
                artist.setp(
                    dataObject.line2D,
                    **dataObject.graph_properties['line2D_properties'])

        if dataObject.line2Dfit is None and dataObject.fit is not None:
            if dataObject.line2D:
                color = artist.getp(dataObject.line2D, 'color')
            dataObject.line2Dfit = self.axes[0].plot(dataObject.xdata,
                                                     dataObject.fit,
                                                     linestyle='-',
                                                     color=color,
                                                     lw=2,
                                                     label='fit_' + dataObject.name)[0]
            if dataObject.graph_properties['line2Dfit_properties']:
                artist.setp(
                    dataObject.line2Dfit,
                    **dataObject.graph_properties['line2Dfit_properties'])

#         if dataObject.line2Dresiduals is None and dataObject.residuals is not None:
#             dataObject.line2Dresiduals = self.axes[1].plot(dataObject.xdata,
#                                                   dataObject.residuals,
#                                                    linestyle='-',
#                                                     lw = 2,
#                                                      label = 'residuals_' + dataObject.name)[0]
#
#             if dataObject.graph_properties['line2Dresiduals_properties']:
#                 artist.setp(dataObject.line2Dresiduals, **dataObject.graph_properties['line2Dresiduals_properties'])

        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()
        dataObject._save_graph_properties()

    def redraw_dataObjects(self, dataObjects, visible=True):
        for dataObject in dataObjects:
            if not dataObject:
                continue
            if dataObject.line2D:
                dataObject.line2D.set_data(dataObject.xdata, dataObject.ydata)
                dataObject.line2D.set_visible(visible)
            if dataObject.line2Dfit:
                dataObject.line2Dfit.set_data(dataObject.xdata, dataObject.fit)
                dataObject.line2Dfit.set_visible(visible)
#             if dataObject.line2Dresiduals:
#                dataObject.line2Dresiduals.set_data(dataObject.xdata, dataObject.residuals)
#                dataObject.line2Dresiduals.set_visible(visible)

#         self.axes[0].autoscale(axis='both', tight = False, enable = True)
        self.axes[0].relim()
        self.axes[0].autoscale_view(None, True, True)
        self.draw()

    def removeTrace(self, dataObject):
        if dataObject.line2D:
            dataObject.line2D.remove()
        if dataObject.line2Dfit:
            dataObject.line2Dfit.remove()
        if dataObject.line2Dresiduals:
            dataObject.line2Dresiduals.remove()
        self.draw()

    def removeTraces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]

#         while len(self.axes[1].lines):
#             del self.axes[1].lines[0]

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

    def redraw_dataObjects(self, dataObjects, visible=True):
        for dataObject in dataObjects:
            if not dataObject:
                continue
            if dataObject.line2Dsld_profile and dataObject.sld_profile is not None:
                dataObject.line2Dsld_profile.set_data(
                    dataObject.sld_profile[0],
                    dataObject.sld_profile[1])
                dataObject.line2Dsld_profile.set_visible(visible)
        self.axes[0].relim()
        self.axes[0].autoscale_view(None, True, True)
        self.draw()

    def add_dataObject(self, dataObject):
        if dataObject.line2Dsld_profile is None and dataObject.sld_profile is not None:
            color = 'b'
            if dataObject.line2D:
                color = artist.getp(dataObject.line2D, 'color')

            dataObject.line2Dsld_profile = self.axes[0].plot(
                dataObject.sld_profile[0],
                dataObject.sld_profile[
                    1],
                linestyle='-',
                color=color,
                lw=2,
                label='sld_' + dataObject.name)[0]

            if dataObject.graph_properties['line2Dsld_profile_properties']:
                artist.setp(
                    dataObject.line2Dsld_profile,
                    **dataObject.graph_properties['line2Dsld_profile_properties'])

        self.axes[0].relim()
        self.axes[0].autoscale(axis='both', tight=False, enable=True)
        self.draw()

    def removeTrace(self, dataObject):
        if dataObject.line2Dsld_profile:
            dataObject.line2Dsld_profile.remove()
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
