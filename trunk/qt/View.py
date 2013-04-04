from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow

import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.artist as artist
import DataStoreModel as DSM
import PluginStoreModel as PSM
import pyplatypus.analysis.reflect as reflect
import limitsUI
import os.path
from copy import deepcopy
import numpy as np
import pickle
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
        
        #redirect stdout to a console window
        console = EmittingStream()
        sys.stdout = console
        console.textWritten.connect(self.writeTextToConsole)
        
        self.dataStore = DSM.DataStore()
        self.dataStore.dataChanged.connect(self.dataObjects_visibilityChanged)
        self.current_dataset = None

        self.modelStore = DSM.ModelStore()
        self.pluginStoreModel = PSM.PluginStoreModel()
        self.reflectPlugin = self.pluginStoreModel.plugins[0]
                
        self.modifyGui()
        
        parameters = np.array([1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 25, 3.47, 0, 3])
        fitted_parameters = np.array([1,2,3,4,5,6,7,8,9, 10, 11])

        tempq = np.linspace(0.008, 0.5, num = 1000)
        tempr = np.ones_like(tempq)
        tempe = np.zeros_like(tempq)
        tempdq  = np.copy(tempq) * 5 / 100.
        dataTuple = (tempq, tempr, tempe, tempdq)
        
        self.current_dataset = None
        self.theoretical = DSM.dataObject(dataTuple = dataTuple)

        theoreticalmodel = DSM.Model(parameters=parameters, fitted_parameters = fitted_parameters)
        self.modelStore.addModel(theoreticalmodel, 'theoretical')
        self.baseModel = DSM.BaseModel(self.modelStore.models['theoretical'])
        self.layerModel = DSM.LayerModel(self.modelStore.models['theoretical'])
        self.genericModel = PSM.PluginParametersModel(self.modelStore.models['theoretical'])

        self.theoretical.evaluate_model(theoreticalmodel, store = True)
        
        self.dataStore.addDataObject(self.theoretical)
        
        self.theoretical.line2Dsld_profile = self.sldgraphs.axes[0].plot(self.theoretical.sld_profile[0],
                                                   self.theoretical.sld_profile[1],
                                                    linestyle='-')[0]
        self.theoretical.line2Dfit = self.reflectivitygraphs.axes[0].plot(self.theoretical.W_q,
                                                   self.theoretical.fit,
                                                    linestyle='-', lw=2, label = 'theoretical')[0]

        self.redraw_dataObject_graphs([self.theoretical])
        
        self.ui.dataOptions_tableView.setModel(self.dataStore)
        
        self.ui.dataset_comboBox.setModel(self.dataStore)
        self.ui.dataset_comboBox.setModelColumn(0)
        self.ui.model_comboBox.setModel(self.modelStore)
        
        self.ui.baseModelView.setModel(self.baseModel)
        self.ui.layerModelView.setModel(self.layerModel)
        self.baseModel.layersAboutToBeInserted.connect(self.layerModel.layersAboutToBeInserted)
        self.baseModel.layersAboutToBeRemoved.connect(self.layerModel.layersAboutToBeRemoved)
        self.baseModel.layersFinishedBeingInserted.connect(self.layerModel.layersFinishedBeingInserted)
        self.baseModel.layersFinishedBeingRemoved.connect(self.layerModel.layersFinishedBeingRemoved)
        self.layerModel.dataChanged.connect(self.update_gui_modelChanged)
        self.baseModel.dataChanged.connect(self.update_gui_modelChanged)
        self.ui.baseModelView.clicked.connect(self.baseCurrentCellChanged)
        self.ui.layerModelView.clicked.connect(self.layerCurrentCellChanged)
        self.ui.genericModelView.clicked.connect(self.genericCurrentCellChanged)

        self.ui.genericModelView.setModel(self.genericModel)
        self.genericModel.dataChanged.connect(self.update_gui_modelChanged)
        self.ui.UDF_comboBox.setModel(self.pluginStoreModel)
        
        print 'Session started at:', time.asctime( time.localtime(time.time()) )

    
    def __del__(self):
        # Restore sys.stdout
        sys.stdout = sys.__stdout__
    
    def writeTextToConsole(self, text):
        self.ui.plainTextEdit.insertPlainText(text)
         
    def __saveState(self, f):
        state = [self.dataStore, self.modelStore, self.current_dataset.name]
        pickle.dump(self.state, f, -1)

    def dataObjects_visibilityChanged(self, arg_1, arg_2):
        if arg_1.column() < 0:
            return
        
        name = self.dataStore.names[arg_1.row()]
        dataObject = self.dataStore.dataObjects[name]
        if dataObject.line2D is not None:
            dataObject.line2D.set_visible(dataObject.graph_properties['visible'])
        self.redraw_dataObject_graphs([dataObject], visible = dataObject.graph_properties['visible'])

    @QtCore.Slot(QtGui.QDropEvent)
    def dropEvent(self, event):
        m = event.mimeData()
        urls = m.urls()
        for url in urls:
            try:
                dataObject = self.dataStore.loadDataObject(url.toLocalFile())
                if dataObject is not None:
                    self.reflectivitygraphs.add_dataObject(dataObject)
                    self.sldgraphs.add_dataObject(dataObject)
            except Exception as inst:
                pass
            
            try:
                self.loadModel(url.toLocalFile())
            except Exception:
                pass
            

                
    @QtCore.Slot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event):
        m = event.mimeData()
        if m.hasUrls():
            event.acceptProposedAction()
    
    @QtCore.Slot()
    def on_actionSave_experiment_triggered(self):
        experimentFileName, ok = QtGui.QFileDialog.getSaveFileName(self, caption = 'Save experiment as:', dir='experiment.fdob')
        
        if not ok:
            return
        
        try:
            tempdirectory = tempfile.mkdtemp()
            datasetd = os.path.join(tempdirectory,'datasets')
            os.mkdir(datasetd)
            self.dataStore.saveDataStore(datasetd)
            modeld = os.path.join(tempdirectory,'models')
            os.mkdir(modeld)
            self.modelStore.saveModelStore(modeld)
            with zipfile.ZipFile(experimentFileName, 'w') as zip:
                DSM.zipper(tempdirectory, zip)
        except Exception as inst:
            print type(inst)
        finally: 
            shutil.rmtree(tempdirectory)
    
    @QtCore.Slot()
    def on_actionLoad_experiment_triggered(self):
        experimentFileName, ok = QtGui.QFileDialog.getOpenFileName(self,
                                                                  caption = 'Select Experiment File',
                                                                  filter = 'Experiment Files (*.fdob)')
        if not ok:
            return
        
        tempdirectory = tempfile.mkdtemp()
         
        with zipfile.ZipFile(experimentFileName, 'r') as zip:
            zip.extractall(tempdirectory)
        
        datasetd = os.path.join(tempdirectory,'datasets')
        files = [os.path.join(datasetd, file) for file in os.listdir(datasetd) if not os.path.isdir(file)]
        self.dataStore.loadDataStore(files, clear = True)

        modeld = os.path.join(tempdirectory,'models')
        files = [os.path.join(modeld, file) for file in os.listdir(modeld) if not os.path.isdir(file)]
        self.modelStore.loadModelStore(files, clear = True)
        
        #remove and add dataObjectsToGraphs
        self.reflectivitygraphs.removeTraces()
        for dataObject in self.dataStore:
            self.reflectivitygraphs.add_dataObject(dataObject)
            
        self.theoretical = self.dataStore.getDataObject('_theoretical_')
#        self.reflectivitygraphs.axes[0].lines.remove(self.theoretical.line2D)
#        self.reflectivitygraphs.axes[1].lines.remove(self.theoretical.line2Dresiduals)
        
        #when you load in the theoretical model you destroy the link to the gui, reinstate it.
        self.theoreticalmodel = self.modelStore.models['theoretical']
        self.baseModel.model = self.theoreticalmodel
        self.layerModel.model = self.theoreticalmodel
        self.theoretical.evaluate_model(self.theoreticalmodel, store = True)
            
        self.theoretical.line2Dfit = self.reflectivitygraphs.axes[0].plot(self.theoretical.W_q,
                                                self.theoretical.fit,
                                                 linestyle='-', lw=2, label = 'theoretical')[0]
        self.reflectivitygraphs.draw()
                    
        shutil.rmtree(tempdirectory)

                       
    @QtCore.Slot()
    def on_actionLoad_Data_triggered(self):
        """
            you load data
        """ 
        theFiles, ok = QtGui.QFileDialog.getOpenFileNames(self,  caption = 'Select Reflectivity Files')
        if not ok:
            return
            
        for file in theFiles:
            dataObject = self.dataStore.loadDataObject(file)
            self.reflectivitygraphs.add_dataObject(dataObject)

    @QtCore.Slot()
    def on_actionSave_Data_triggered(self):
        listoffits = []
        for key in self.dataStore.dataObjects:
            if self.dataStore.dataObjects[key].fit is not None:
                listoffits.append(key)
                
        which_fit, ok = QtGui.QInputDialog.getItem(self, "Which fit did you want to save?", "fit", listoffits, editable=False)
        
        if not ok:
            return
        
        fitFileName, ok = QtGui.QFileDialog.getSaveFileName(self, caption = 'Save fit as:', dir='fit_' + which_fit)
        
        if not ok:
            return
        
        dataObject = self.dataStore.dataObjects[which_fit]
        
        with open(fitFileName, 'wb') as f:
            np.savetxt(f, np.column_stack((dataObject.W_q, dataObject.fit)))
            
    @QtCore.Slot()
    def on_actionSave_Model_triggered(self):
        #save a model
        #which model are you saving?
        listofmodels = self.modelStore.names
        
        which_model, ok = QtGui.QInputDialog.getItem(self, "Which model did you want to save?", "model", listofmodels, editable=False)
        if not ok:
            return
            
        themodel = self.modelStore.models[which_model]
        modelFileName, ok = QtGui.QFileDialog.getSaveFileName(self, caption = 'Save model as:', dir=which_model)
        if not ok:
            return
             
        with open(modelFileName, 'w+') as f:
            themodel.save(f)
    
                   
    @QtCore.Slot()
    def on_actionLoad_Model_triggered(self):
        #load a model
        modelFileName, ok = QtGui.QFileDialog.getOpenFileName(self,  caption = 'Select Model File')
        if not ok:
            return
        
        self.loadModel(modelFileName)

    def loadModel(self, fileName):
        themodel = DSM.Model()
        
        with open(fileName, 'Ur') as f:
            themodel.load(f)

        modelName = os.path.basename(fileName)
        self.modelStore.addModel(themodel, os.path.basename(modelName))
        if self.ui.model_comboBox.count() == 1:
            self.ui.model_comboBox.setCurrentIndex(-1)

    @QtCore.Slot()
    def on_actionLoad_Plugin_triggered(self):
        #load a model
        pluginFileName, ok = QtGui.QFileDialog.getOpenFileName(self,
                                                              caption = 'Select plugin File',
                                                             filter = 'Python Plugin File (*.py)')
        if not ok:
            return
        
        self.pluginStoreModel.addPlugin(pluginFileName)
            
    @QtCore.Slot()
    def on_actionRefresh_Datasets_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.dataStore.refresh()
        for dataObject in self.dataStore:
            if dataObject.line2D:
                dataObject.line2D.set_data(dataObject.W_q, dataObject.W_ref)
        self.reflectivitygraphs.draw()
        
 
    def get_limits(self, parameters, fitted_parameters, limits):

        limitsdialog = QtGui.QDialog()
        limitsGUI = limitsUI.Ui_Dialog()
        limitsGUI.setupUi(limitsdialog)

        limitsModel = DSM.LimitsModel(parameters, fitted_parameters, limits)       
        limitsGUI.limits.setModel(limitsModel)
        header = limitsGUI.limits.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)
        
        ok = limitsdialog.exec_()
        limits = limitsModel.limits[:]      
        
        return ok, limits
                       
    @QtCore.Slot()
    def on_do_fit_button_clicked(self):
        """
            you should do a fit
        """
        if self.current_dataset is None:
            return
            
        theoreticalmodel = self.modelStore.models['theoretical']
        alreadygotlimits = False
          
        if ('coef_' + self.current_dataset.name) in self.modelStore.names: 
            model = self.modelStore.models['coef_' + self.current_dataset.name]            
            if model.limits is not None and model.limits.ndim == 2 and np.size(theoreticalmodel.parameters) == np.size(model.limits, 1):
                theoreticalmodel.limits = np.copy(model.limits)
                alreadygotlimits = True
        
        if not alreadygotlimits:
            theoreticalmodel.defaultlimits()              
            
        ok, limits = self.get_limits(theoreticalmodel.parameters,
                                                     theoreticalmodel.fitted_parameters,
                                                      theoreticalmodel.limits)
        
        if not ok:
            return
        
        theoreticalmodel.limits = np.copy(limits)                                
        self.do_a_fit_and_add_to_gui(self.current_dataset, theoreticalmodel)
        
        
    def do_a_fit_and_add_to_gui(self, dataset, model):
        print "___________________________________________________"        
        print "fitting to:", dataset.name
        dataset.do_a_fit(model)
        print "Chi2 :", dataset.chi2 
        np.set_printoptions(suppress=True, precision = 4)
        
        print model.parameters
        print "___________________________________________________"        
    
        newmodel = DSM.Model(parameters = model.parameters,
                                     fitted_parameters = model.fitted_parameters,
                                        limits = model.limits)
                
        self.modelStore.addModel(newmodel, 'coef_' + dataset.name)        
        
        #update GUI
        self.layerModel.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())
        self.baseModel.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

        self.reflectivitygraphs.add_dataObject(dataset)
        self.sldgraphs.add_dataObject(dataset)
                                                             
        if self.ui.model_comboBox.findText('coef_' + dataset.name) < 0:
            self.ui.model_comboBox.setCurrentIndex(self.ui.model_comboBox.findText('coef_' + dataset.name))
        self.update_gui_modelChanged()
        self.redraw_dataObject_graphs([dataset], visible = dataset.graph_properties['visible'])
    
    @QtCore.Slot(int)
    def on_UDF_comboBox_currentIndexChanged(self, arg_1):
    
        if arg_1 == 0:
            #the default reflectometry calculation is being used
            self.modelStore.displayOtherThanReflect = False
            self.ui.genericModelView.hide()
            
            '''
                you need to display a model suitable for reflectometry
            '''
            areAnyModelsValid = [reflect.isProperAbelesInput(model.parameters) for model in self.modelStore]
            try:
                idx = areAnyModelsValid.index(True)
                self.select_a_model(self.modelStore.names[idx])
            except ValueError:
                #none are valid, reset the theoretical model
                parameters = np.array([1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 25, 3.47, 0, 3])
                fitted_parameters = np.array([1,2,3,4,5,6,7,8,9, 10, 11])
                self.modelStore.models['theoretical'].parameters = parameters[:]
                self.modelStore.models['theoretical'].fitted_parameters = fitted_parameters[:]
                    
            self.ui.baseModelView.show()
            self.ui.layerModelView.show()
        else:
            #a user reflectometry plugin is being used.
            self.modelStore.displayOtherThanReflect = True
            self.ui.baseModelView.hide()
            self.ui.layerModelView.hide()
            self.ui.genericModelView.show()

        self.reflectPlugin = self.pluginStoreModel.plugins[arg_1]

                 
    @QtCore.Slot(unicode)
    def on_dataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        self.current_dataset = self.dataStore.dataObjects[arg_1]
        self.update_gui_modelChanged()
                           
    @QtCore.Slot(unicode)
    def on_model_comboBox_currentIndexChanged(self, arg_1):
        """
        model selection changed, update view with parameters from model.
        """
        self.select_a_model(arg_1)        
    
    def select_a_model(self, arg_1):
        try:
            model = self.modelStore.models[arg_1]
            if model.parameters is not None:
                self.modelStore.models['theoretical'].parameters =  model.parameters[:]
            if model.fitted_parameters is not None:
                self.modelStore.models['theoretical'].fitted_parameters =  model.fitted_parameters[:]

            if model.limits is not None and model.limits.ndim == 2 and np.size(model.limits, 1) == np.size(model.parameters):
                self.modelStore.models['theoretical'].limits = np.copy(model.limits)
            else:
                self.modelStore.models['theoretical'].defaultlimits()
            
            self.baseModel.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

            if model.parameters is not None:
                self.modelStore.models['theoretical'].parameters =  model.parameters[:]
            if model.fitted_parameters is not None:            
                self.modelStore.models['theoretical'].fitted_parameters =  model.fitted_parameters[:]

            self.layerModel.dataChanged.emit(self.layerModel.createIndex(0,0), self.layerModel.createIndex(2 + int(model.parameters[0]),3))
                        
            self.layerModel.modelReset.emit()

#            self.update_gui_modelChanged()
        except KeyError:
            return
        except IndexError, AttributeError:
            print model.parameters, model.fitted_parameters

    @QtCore.Slot(float)
    def on_res_SpinBox_valueChanged(self, arg_1):
        if arg_1 < 0.5:
            arg_1 = 0
            
        self.theoretical.W_qSD = arg_1 * self.theoretical.W_q / 100.

        self.update_gui_modelChanged()
                 
    @QtCore.Slot(int)
    def on_use_errors_checkbox_stateChanged(self, arg_1):
        """
        want to weight by error bars, recalculate chi2
        """
        
        theoreticalmodel = self.modelStore.models['theoretical']
        if arg_1:
            theoreticalmodel.useerrors = True
            theoreticalmodel.costfunction = reflect.costfunction_logR_weight
        else:
            theoreticalmodel.costfunction = reflect.costfunction_logR_noweight
            
        self.update_gui_modelChanged()
            
    @QtCore.Slot(int)
    def on_use_dqwave_checkbox_stateChanged(self, arg_1):
        """
        """
        theoreticalmodel = self.modelStore.models['theoretical']
        
        if arg_1:
            theoreticalmodel.usedq = True
        else:
            theoreticalmodel.usedq = False
            
    def layerCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()

        self.currentCell= {}

        if row == 0 and (col == 0 or col == 3):
            return

        theoreticalmodel = self.modelStore.models['theoretical']

        if row == int(theoreticalmodel.parameters[0]) + 1 and col == 0:
            return
                    
        try:
            param = self.layerModel.rowcoltoparam(row, col, int(theoreticalmodel.parameters[0]))
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

        theoreticalmodel = self.modelStore.models['theoretical']

        self.currentCell= {}

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

    
    def genericCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()

        self.currentCell= {}

        if row == 0:
            return

        theoreticalmodel = self.modelStore.models['theoretical']

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
        self.currentCell['model'] = self.genericModel
        
    @QtCore.Slot(int)
    def on_horizontalSlider_valueChanged(self, arg_1):
        try:
            c = self.currentCell

            if not c['readyToChange']:
                return

            theoreticalmodel = self.modelStore.models['theoretical']
                
            val = c['lowlim'] + (arg_1 / 1000.) * math.fabs(c['lowlim'] - c['hilim'])
              
            param = c['param']         
            theoreticalmodel.parameters[c['param']] = val

            c['model'].dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())
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
            parameters = self.modelStore.models['theoretical'].parameters
            
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
        #add the plots
        self.sldgraphs = MySLDGraphs(self.ui.sld)
        self.ui.gridLayout_4.addWidget(self.sldgraphs)

        self.reflectivitygraphs = MyReflectivityGraphs(self.ui.reflectivity)        
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs)
        
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs.mpl_toolbar)
        self.ui.gridLayout_4.addWidget(self.sldgraphs.mpl_toolbar)
        
        header = self.ui.layerModelView.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)
        
        header2 = self.ui.genericModelView.horizontalHeader()
        header2.setResizeMode(QtGui.QHeaderView.Stretch)
        
        self.ui.genericModelView.hide()
                     
    def redraw_dataObject_graphs(self, dataObjects, visible = True):
        self.reflectivitygraphs.redraw_dataObjects(dataObjects, visible = visible)        
        self.sldgraphs.redraw_dataObjects(dataObjects, visible = visible)
                
    def update_gui_modelChanged(self, store = False):
        theoreticalmodel = self.modelStore.models['theoretical']

        self.theoretical.evaluate_model(theoreticalmodel, store = True, reflectPlugin = self.reflectPlugin['rfo'])

        if self.current_dataset is not None:
            energy = self.current_dataset.evaluate_chi2(theoreticalmodel, reflectPlugin = self.reflectPlugin['rfo'])
            self.ui.chi2lineEdit.setText(str(energy))     
        
        self.redraw_dataObject_graphs([self.theoretical])
        
            
class MyReflectivityGraphs(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1,1,1), edgecolor=(0,0,0))
        #reflectivity graph
        self.axes = []
        ax = self.figure.add_axes([0.1,0.22,0.85,0.75])
        self.axes.append(ax)
        self.axes[0].autoscale(axis='both', tight = False)
        self.axes[0].set_xlabel('Q')
        self.axes[0].set_ylabel('R')
        self.axes[0].set_yscale('log')
        
        #residual plot
        #, sharex=self.axes[0]
        ax2 = self.figure.add_axes([0.1,0.04,0.85,0.14], sharex=ax, frame_on = False)
        self.axes.append(ax2)
        self.axes[1].set_visible(True)
        self.axes[1].set_ylabel('residual')
                       
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
#        self.figure.subplots_adjust(left=0.1, right=0.95, top = 0.98)
        self.mpl_toolbar = NavigationToolbar(self, parent)
        
    def add_dataObject(self, dataObject):
        if dataObject.line2D is None:
            lineInstance = self.axes[0].plot(dataObject.W_q,
                                          dataObject.W_ref,
                                           markersize=5,
                                            marker='o',
                                             linestyle='',
                                              label = dataObject.name)
            dataObject.line2D = lineInstance[0]
            if dataObject.graph_properties['line2D_properties']:
                artist.setp(dataObject.line2D, **dataObject.graph_properties['line2D_properties'])
        
        if dataObject.line2Dfit is None and dataObject.fit is not None:
            dataObject.line2Dfit = self.axes[0].plot(dataObject.W_q,
                                                                        dataObject.fit,
                                                                           linestyle='-',
                                                                            lw = 2,
                                                                             label = 'fit_' + dataObject.name)[0]
            if dataObject.graph_properties['line2Dfit_properties']:
                artist.setp(dataObject.line2Dfit, **dataObject.graph_properties['line2Dfit_properties'])
        
        if dataObject.line2Dresiduals is None and dataObject.residuals is not None:
            dataObject.line2Dresiduals = self.axes[1].plot(dataObject.W_q,
                                                  dataObject.residuals,
                                                   linestyle='-',
                                                    lw = 2,
                                                     label = 'residuals_' + dataObject.name)[0]

            if dataObject.graph_properties['line2Dresiduals_properties']:
                artist.setp(dataObject.line2Dresiduals, **dataObject.graph_properties['line2Dresiduals_properties'])
                                                      
        self.draw()
    
    def redraw_dataObjects(self, dataObjects, visible = True):
        for dataObject in dataObjects:
            if not dataObject:
                continue
            if dataObject.line2D:
               dataObject.line2D.set_data(dataObject.W_q, dataObject.W_ref)
               dataObject.line2D.set_visible(visible)               
            if dataObject.line2Dfit:
               dataObject.line2Dfit.set_data(dataObject.W_q, dataObject.fit)
               dataObject.line2Dfit.set_visible(visible)
            if dataObject.line2Dresiduals:
               dataObject.line2Dresiduals.set_data(dataObject.W_q, dataObject.residuals)
               dataObject.line2Dresiduals.set_visible(visible)
        self.draw()
                       
    def removeTraces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]
 
        while len(self.axes[1].lines):
            del self.axes[1].lines[0]
                        
        self.draw()
        
        
                                            
class MySLDGraphs(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1,1,1), edgecolor=(0,0,0))
        self.axes = []
        #SLD plot
        self.axes.append(self.figure.add_subplot(111))
        self.axes[0].autoscale(axis='both', tight = False)
        self.axes[0].set_xlabel('z')
        self.axes[0].set_ylabel('SLD')
                       
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.subplots_adjust(left=0.1, right=0.95, top = 0.98)
        self.mpl_toolbar = NavigationToolbar(self, parent)
        
    def redraw_dataObjects(self, dataObjects, visible = True):
        for dataObject in dataObjects:
            if not dataObject:
                continue
            if dataObject.line2Dsld_profile and dataObject.sld_profile is not None:
               dataObject.line2Dsld_profile.set_data(dataObject.sld_profile[0], dataObject.sld_profile[1])
               dataObject.line2Dsld_profile.set_visible(visible)
        self.draw()
        
    def add_dataObject(self, dataObject):
        if dataObject.line2Dsld_profile is None and dataObject.sld_profile is not None:
            dataObject.line2Dsld_profile = self.axes[0].plot(dataObject.sld_profile[0],
                                            dataObject.sld_profile[1],
                                               linestyle='-',
                                                lw = 2,
                                                 label = 'sld_' + dataObject.name)[0]
            
            if dataObject.graph_properties['line2Dsld_profile_properties']:
                artist.setp(dataObject.line2Dsld_profle, **dataObject.graph_properties['line2Dsld_profile_properties'])
                
        self.draw()
        
    def removeTraces(self):
        while len(self.axes[0].lines):
            del self.axes[0].lines[0]
        
        self.draw()
        
class EmittingStream(QtCore.QObject):
    #a class for rewriting stdout to a console window
    textWritten = QtCore.Signal(str)
    
    def write(self, text):
        self.textWritten.emit(str(text))