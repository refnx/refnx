from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow

import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.artist
import pyplatypus.dataset.DataStore as DataStore
import pyplatypus.analysis.reflect as reflect
import os.path
from copy import deepcopy
import pickle
import math

class MyMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.errorHandler = QtGui.QErrorMessage()
        self.dataStore = DataStore.DataStore()
        self.dataStore.dataChanged.connect(self.dataObjects_visibilityChanged)
        self.current_dataset = None
        self.modelStore = DataStore.ModelStore()
        self.modifyGui()
        
        parameters = np.array([1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 25, 3.47, 0, 3])
        fitted_parameters = np.array([1,2,3,4,5,6,7,8,9, 10, 11])

        tempq = np.linspace(0.008, 0.5, num = 1000)
        tempr = np.ones_like(tempq)
        tempe = np.zeros_like(tempq)
        tempdq  = np.copy(tempq) * 5 / 100.
        dataTuple = (tempq, tempr, tempe, tempdq)
        
        self.current_dataset = None
        self.theoretical = DataStore.dataObject(dataTuple = dataTuple)

        theoreticalmodel = DataStore.Model(parameters=parameters, fitted_parameters = fitted_parameters)
        self.modelStore.addModel(theoreticalmodel, 'theoretical')
        self.baseModel = DataStore.BaseModel(self.modelStore.models['theoretical'])
        self.layerModel = DataStore.LayerModel(self.modelStore.models['theoretical'])


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
    
    def pants(self):
        print "Viewport entered"
        
    def __saveState(self, f):
        state = [self.dataStore, self.modelStore, self.current_dataset.name]
        pickle.dump(self.state, f, -1)

    def dataObjects_visibilityChanged(self, arg_1, arg_2):
        if arg_1.row() or arg_1.column() < 0:
            return
        
        name = self.dataStore.names[arg_1.row()]
        dataObject = self.dataStore.dataObjects[name]
        if dataObject.line2D is not None:
            dataObject.line2D.set_visible(dataObject.visible)
        self.redraw_dataObject_graphs([dataObject], visible = dataObject.visible)

    @QtCore.Slot(QtGui.QDropEvent)
    def dropEvent(self, event):
        m = event.mimeData()
        urls = m.urls()
        for url in urls:
            try:
                dataObject = self.dataStore.loadDataObject(url.toLocalFile())
                self.add_dataObject_to_gui(dataObject)
            except Exception:
                #try loading a model file.
                self.loadModel(url.toLocalFile())
                

                
    @QtCore.Slot(QtGui.QDragEnterEvent)
    def dragEnterEvent(self, event):
        m = event.mimeData()
        if m.hasUrls():
            event.acceptProposedAction()
              
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
            self.add_dataObject_to_gui(dataObject)
            
    def add_dataObject_to_gui(self, dataObject):
        lineInstance = self.reflectivitygraphs.axes[0].plot(dataObject.W_q,
                                                                 dataObject.W_ref,
                                                                   markersize=5,
                                                                    marker='o',
                                                                     linestyle='',
                                                                      label = dataObject.name)
        dataObject.line2D = lineInstance[0]
        self.reflectivitygraphs.draw()
        if self.dataStore.rowCount() == 1:
            self.ui.dataset_comboBox.setCurrentIndex(0)

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
    def on_actionLoad_Model_triggered(self):
        #load a model
        modelFileName, ok = QtGui.QFileDialog.getOpenFileName(self,  caption = 'Select Model File')
        if not ok:
            return
        
        self.loadModel(modelFileName)
            
    def loadModel(self, fileName):
        themodel = DataStore.Model()
        
        with open(fileName, 'Ur') as f:
            themodel.load(f)

        modelName = os.path.basename(fileName)

        self.modelStore.addModel(themodel, os.path.basename(modelName))
        if self.ui.model_comboBox.count() == 1:
            self.ui.model_comboBox.setCurrentIndex(-1)
    
    @QtCore.Slot()
    def on_actionRefresh_Datasets_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.dataStore.refresh()
        for key in self.dataStore.dataObjects:
            dataObject = self.dataStore.dataObjects[key]
            if dataObject.line2D:
                dataObject.line2D.set_data(dataObject.W_q, dataObject.W_ref)
        self.reflectivitygraphs.draw()
 
                       
    @QtCore.Slot()
    def on_do_fit_button_clicked(self):
        """
            you should do a fit
        """
        if self.current_dataset is None:
            return
        
        theoreticalmodel = self.modelStore.models['theoretical']
        
        self.ui.statusbar.showMessage('fitting')
        self.current_dataset.do_a_fit(theoreticalmodel)
        self.ui.statusbar.clearMessage()
        
        newmodel = DataStore.Model(parameters = theoreticalmodel.parameters, fitted_parameters = theoreticalmodel.fitted_parameters)
                
        self.modelStore.addModel(newmodel, 'coef_' + self.current_dataset.name)

#TODO Update gui when fit has finished        
#        self.gui_from_parameters(theoreticalmodel.parameters, theoreticalmodel.fitted_parameters)
        
        
        if self.current_dataset.line2Dfit is None:
            self.current_dataset.line2Dfit = self.reflectivitygraphs.axes[0].plot(self.current_dataset.W_q,
                                                  self.current_dataset.fit,
                                                   linestyle='-',
                                                    lw = 2,
                                                     label = 'fit_' + self.current_dataset.name)[0]

        if self.current_dataset.line2Dsld_profile is None:
            self.current_dataset.line2Dsld_profile = self.sldgraphs.axes[0].plot(self.current_dataset.sld_profile[0],
                                                  self.current_dataset.sld_profile[1],
                                                   linestyle='-',
                                                    lw = 2,
                                                     label = 'sld_' + self.current_dataset.name)[0]
        
        if self.current_dataset.line2Dresiduals is None:
            self.current_dataset.line2Dresiduals = self.reflectivitygraphs.axes[1].plot(self.current_dataset.W_q,
                                                  self.current_dataset.residuals,
                                                   linestyle='-',
                                                    lw = 2,
                                                     label = 'residuals_' + self.current_dataset.name)[0]
                                                     
        if self.ui.model_comboBox.findText('coef_' + self.current_dataset.name) < 0:
            self.ui.model_comboBox.setCurrentIndex(self.ui.model_comboBox.findText('coef_' + self.current_dataset.name))
        self.update_gui_modelChanged()
        self.redraw_dataObject_graphs([self.current_dataset], visible = self.current_dataset.visible)
        
                      
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
            if model.parameters is not None and model.fitted_parameters is not None:
                self.modelStore.models['theoretical'].parameters =  model.parameters[:]
                self.modelStore.models['theoretical'].fitted_parameters =  model.fitted_parameters[:]
                self.baseModel.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

#                 self.layerModel.beginInsertRows(QtCore.QModelIndex(), -1, -1)
                self.modelStore.models['theoretical'].parameters =  model.parameters[:]
                self.modelStore.models['theoretical'].fitted_parameters =  model.fitted_parameters[:]
                self.layerModel.dataChanged.emit(self.layerModel.createIndex(0,0), self.layerModel.createIndex(2 + int(model.parameters[0]),3))
#                 self.layerModel.endInsertRows()
                self.layerModel.modelReset.emit()

                self.update_gui_modelChanged()
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
            
        print theoreticalmodel.usedq

    def layerCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()
        if row == 0 and (col == 0 or col == 3):
            return

        theoreticalmodel = self.modelStore.models['theoretical']

        if row == int(theoreticalmodel.parameters[0]) + 1 and col == 0:
            return
            
        self.currentCell= {}
        self.currentCell['model'] = self.layerModel 
        self.currentCell['row'] = row
        self.currentCell['col'] = col
        
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
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
    
    def baseCurrentCellChanged(self, index):
        row = index.row()
        col = index.column()

        theoreticalmodel = self.modelStore.models['theoretical']

        if col == 0:
            return
        
        self.currentCell= {}
        self.currentCell['model'] = self.baseModel 
        self.currentCell['row'] = row
        self.currentCell['col'] = col
        
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
        self.currentCell['val'] = val
        self.currentCell['lowlim'] = lowlim
        self.currentCell['hilim'] = hilim
        self.currentCell['readyToChange'] = True
        
    @QtCore.Slot(int)
    def on_horizontalSlider_valueChanged(self, arg_1):
        try:
            c = self.currentCell

            if not c['readyToChange']:
                return

            theoreticalmodel = self.modelStore.models['theoretical']
                
            if c['row'] == 0 and c['col'] == 0:
                return
            if c['model'] is self.layerModel:
                if c['row'] == 0 and (c['col'] == 0 or c['col'] == 3):
                    return
                if c['row'] == int(theoretical.parameters[0]) + 1 and c['col'] == 0 or c['col'] == 3:
                    return
            
            val = c['lowlim'] + (arg_1 / 1000.) * math.fabs(c['lowlim'] - c['hilim'])
                        
            item = c['widget'].item(c['row'], c['col']).setText(str(val))
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
            self.currentCellChanged(c['widget'], c['row'], c['col'])
        except (ValueError, AttributeError):
            return
               
    def modifyGui(self):
        #add the plots
        self.sldgraphs = MySLDGraphs(self.ui.sld)
        self.ui.gridLayout_4.addWidget(self.sldgraphs)

        self.reflectivitygraphs = MyReflectivityGraphs(self.ui.reflectivity)        
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs)
        
        self.ui.gridLayout_5.addWidget(self.reflectivitygraphs.mpl_toolbar)
        self.ui.gridLayout_4.addWidget(self.sldgraphs.mpl_toolbar)
        
#         #add baseparams table widget info
#         header = self.ui.baseparams_tableWidget.horizontalHeader()
#         header.setResizeMode(QtGui.QHeaderView.Stretch)
# 
#         #add layerparams table widget info
#         self.ui.layerparams_tableWidget.setHorizontalHeaderLabels(['thickness', 'sld', 'iSLD', 'roughness'])
#         self.ui.layerparams_tableWidget.setVerticalHeaderLabels(['fronting', '1', 'backing'])
#         
#         header = self.ui.layerparams_tableWidget.horizontalHeader()
#         header.setResizeMode(QtGui.QHeaderView.Stretch)
#         header = self.ui.layerparams_tableWidget.verticalHeader()
#         header.setResizeMode(QtGui.QHeaderView.Stretch)
                     
    def redraw_dataObject_graphs(self, dataObjects, visible = True):
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
            if dataObject.line2Dsld_profile:
               dataObject.line2Dsld_profile.set_data(dataObject.sld_profile[0], dataObject.sld_profile[1])
               dataObject.line2Dsld_profile.set_visible(visible)
        
        self.sldgraphs.draw()    
        self.reflectivitygraphs.draw()
        
                
    def update_gui_modelChanged(self, store = False):
        theoreticalmodel = self.modelStore.models['theoretical']
        self.theoretical.evaluate_model(theoreticalmodel, store = True)

        if self.current_dataset is not None:
            energy = self.current_dataset.evaluate_chi2(theoreticalmodel)
            self.ui.lineEdit.setText(str(energy))     
        
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