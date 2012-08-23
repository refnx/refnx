from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow

import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.rcParams['backend.qt4']='PySide'

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
import pyplatypus.dataset.DataStore as DataStore
import pyplatypus.analysis.reflect as reflect
import os.path

class MyMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.errorHandler = QtGui.QErrorMessage()
        self.dataStore = DataStore.DataStore()
        self.current_dataset = None
        self.models = {}
        self.modifyGui()
        
        parameters = np.array([1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 25, 3.47, 0, 3])
        fitted_parameters = np.array([1,2,3,4,5,6,7,8,9, 10, 11])

        tempq = np.linspace(0.005, 0.5, num = 1000)
        tempr = np.ones_like(tempq)
        tempe = np.zeros_like(tempq)
        tempdq  = np.copy(tempq) * 5 / 100.
        dataTuple = (tempq, tempr, tempe, tempdq)
        
        self.current_dataset = None
        self.theoretical = DataStore.dataObject(dataTuple = dataTuple, fitted_parameters = fitted_parameters, parameters = parameters)
        self.models['theoretical'] = self.theoretical.model
        self.theoretical.evaluate_model(store = True)
        self.dataStore.addDataObject(self.theoretical)
        self.theoretical.line2Dfit = self.reflectivitygraphs.axes[0].plot(self.theoretical.W_q,
                                                   self.theoretical.fit,
                                                    linestyle='-', lw=2, label = 'theoretical')[0]
        self.theoretical.line2Dsld_profile = self.reflectivitygraphs.axes[2].plot(self.theoretical.sld_profile[0],
                                                   self.theoretical.sld_profile[1],
                                                    linestyle='-')[0]
        self.gui_from_parameters(self.theoretical.model.parameters, self.theoretical.model.fitted_parameters, resize=False)
        self.redraw_dataObject_graphs([self.theoretical])
        
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
        self.ui.dataset_comboBox.addItem(dataObject.name)

    @QtCore.Slot()
    def on_actionSave_Model_triggered(self):
        #save a model
        #which model are you saving?
        listofmodels = []
        for key in self.models:
            listofmodels.append(key)
        
        which_model, ok = QtGui.QInputDialog.getItem(self, "Which model did you want to save?", "model", listofmodels, editable=False)
        if not ok:
            return
            
        themodel = self.models[which_model]
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
        
        themodel = DataStore.Model()
        
        with open(modelFileName, 'Ur') as f:
            themodel.load(f)

        modelName = os.path.basename(modelFileName)        
        self.models[os.path.basename(modelName)] = themodel
        self.ui.model_comboBox.addItem(modelName)
        self.gui_from_parameters(themodel.parameters, themodel.fitted_parameters)
        
        
    
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
            
        self.theoretical.model.parameters, self.theoretical.model.fitted_parameters = self.gui_to_parameters()
        self.ui.statusbar.showMessage('fitting')
        self.current_dataset.do_a_fit(model = self.theoretical.model)
        self.ui.statusbar.clearMessage()
            
        self.theoretical.model.parameters = np.copy(self.current_dataset.model.parameters)
        self.models[self.current_dataset.name] = self.current_dataset.model
        
        self.gui_from_parameters(self.theoretical.model.parameters, self.theoretical.model.fitted_parameters)
        
        
        if self.current_dataset.line2Dfit is None:
            self.current_dataset.line2Dfit = self.reflectivitygraphs.axes[0].plot(self.current_dataset.W_q,
                                                  self.current_dataset.fit,
                                                   linestyle='-',
                                                    lw = 2,
                                                     label = 'fit_' + self.current_dataset.name)[0]

        if self.current_dataset.line2Dsld_profile is None:
            self.current_dataset.line2Dsld_profile = self.reflectivitygraphs.axes[2].plot(self.current_dataset.sld_profile[0],
                                                  self.current_dataset.sld_profile[1],
                                                   linestyle='-',
                                                    lw = 2,
                                                     label = 'sld_' + self.current_dataset.name)[0]
        
        
        if self.ui.model_comboBox.findText(self.current_dataset.name) < 0:
            self.ui.model_comboBox.addItem(self.current_dataset.name)
            self.ui.model_comboBox.setCurrentIndex(self.ui.model_comboBox.findText(self.current_dataset.name))
        self.update_gui_modelChanged()
        
        
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
        
        try:
            dataObject = self.dataStore.dataObjects[arg_1]
            if dataObject.model.parameters is not None and dataObject.model.fitted_parameters is not None:
                self.gui_from_parameters(dataObject.model.parameters, dataObject.model.fitted_parameters, resize = True)
                #self.update_gui_modelChanged()
        except KeyError:
            return
        except IndexError, AttributeError:
            print dataObject.model.parameters, dataObject.model.fitted_parameters
               
    @QtCore.Slot(float)
    def on_doubleSpinBox_valueChanged(self, arg_1):
        if arg_1 < 0.5:
            arg_1 = 0
            
        self.theoretical.W_qSD = arg_1 * self.theoretical.W_q / 100.

        self.update_gui_modelChanged()
                 
    @QtCore.Slot(int)
    def on_use_errors_checkbox_stateChanged(self, arg_1):
        """
        want to weight by error bars, recalculate chi2
        """
        
        print arg_1
    
    @QtCore.Slot(QtGui.QTableWidgetItem)
    def on_baseparams_tableWidget_itemChanged(self, arg_1):
        """

        """
        row = self.ui.baseparams_tableWidget.currentRow()
        col = self.ui.baseparams_tableWidget.currentColumn()
        
        if row < 0 or col < 0:
            return
        
        if row == 0 and col == 0:
            validator = QtGui.QIntValidator()
            voutput = validator.validate(arg_1.text(), 1)
               
            if voutput[0] is QtGui.QValidator.State.Acceptable and int(voutput[1]) >= 0:
                oldlayers = self.ui.layerparams_tableWidget.rowCount() - 2
                newlayers = int(voutput[1])
                if oldlayers == newlayers:
                    return
                    
                parameters = self.theoretical.model.parameters
                fitted_parameters = self.theoretical.model.fitted_parameters
                #you have to defocus from layerparams because when you rejig the layering
                #it triggers on_layerparams_tableWidget_itemChanged
                
                self.ui.layerparams_tableWidget.setCurrentCell(-1,-1)
                
                if newlayers == 0:
                    parameters = np.resize(parameters, 8)
                    fitted_parameters = np.extract(fitted_parameters < 8, fitted_parameters)
                    parameters[0] = newlayers
                else:
                    if newlayers > oldlayers:
                        title = 'Where would you like to insert the new layers'
                        maxValue = oldlayers
                        minValue = 0
                        value = 0
                    elif newlayers < oldlayers:
                        title = 'Where would you like to remove the layers from?'
                        maxValue = newlayers + 1
                        minValue = 1
                        value = 1

                    label = 'layer'                
                    insertpoint, ok = QtGui.QInputDialog.getInt(self,
                                               title,
                                                label,
                                                 value = value,
                                                  minValue = minValue,  
                                                   maxValue = maxValue)
                    if not ok:
                        self.ui.baseparams_tableWidget.item(0, 0).setText(str(oldlayers))
                        return         

                    parameters[0] = newlayers
                    if newlayers > oldlayers:
                        parameters = np.insert(parameters,
                                                [4 * insertpoint + 8] * 4 *(newlayers - oldlayers),
                                                 [0, 0, 0, 0] * (newlayers - oldlayers))
                        fitted_parameters = np.where(fitted_parameters >= 4 * insertpoint + 8,
                                  fitted_parameters + (newlayers - oldlayers) * 4,
                                     fitted_parameters)
                        fitted_parameters = np.append(fitted_parameters,
                                                 np.arange(4 * insertpoint + 8, 4 * insertpoint + 8 + (newlayers -oldlayers) * 4))
                    elif newlayers < oldlayers:
                        insertpoint -= 1
                        
                        paramslost = np.arange(4 * insertpoint + 8, 4 * insertpoint + 8 + (oldlayers - newlayers) * 4)
                        parameters = np.delete(parameters, paramslost)
                        fitted_parameters = np.array([val for val in fitted_parameters.tolist() if (val < paramslost[0] or val > paramslost[-1])])
                        fitted_parameters = np.where(fitted_parameters > paramslost[-1],
                                  fitted_parameters + (newlayers - oldlayers) * 4,
                                     fitted_parameters)
                        
                self.gui_from_parameters(parameters, fitted_parameters, resize = True)                                    
                self.theoretical.model.parameters = parameters
                self.theoretical.model.fitted_parameters = fitted_parameters
            else:
                self.errorHandler.showMessage("Number of layers must be integer > 0")
                return
        else:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(arg_1.text(), 1)
            if voutput[0] is QtGui.QValidator.State.Acceptable:
                pass
            else:
                print arg_1.text()
                self.errorHandler.showMessage("values entered must be numeric")
                return
        self.theoretical.model.parameters, self.theoretical.model.fitted_parameters = self.gui_to_parameters()
        self.update_gui_modelChanged()

 
    @QtCore.Slot(QtGui.QTableWidgetItem)
    def on_layerparams_tableWidget_itemChanged(self, arg_1):
        """
       
        """
        row = self.ui.layerparams_tableWidget.currentRow()
        col = self.ui.layerparams_tableWidget.currentColumn()
        numrows = self.ui.layerparams_tableWidget.rowCount()
        numcols = self.ui.layerparams_tableWidget.columnCount()
        if row < 0 or col < 0:
            return
            
        if (row == 0 and col == 0) or (row == numrows - 1 and col == 0) or (row == 0 and col == numcols - 1):
            arg_1.setText("")
            return
        validator = QtGui.QDoubleValidator()
        if validator.validate(arg_1.text(), 1)[0] == QtGui.QValidator.State.Acceptable:
            self.theoretical.model.parameters, self.theoretical.model.fitted_parameters = self.gui_to_parameters()
            self.update_gui_modelChanged()
        else:
            print arg_1.text(), row, col
            self.errorHandler.showMessage("values entered must be numeric")
            return

    def modifyGui(self):
        #add the plots
        self.reflectivitygraphs = MyReflectivityGraphs(self.ui.centralwidget)
        self.ui.gridLayout_3.addWidget(self.reflectivitygraphs)
        self.reflectivitygraphs.axes[0].set_xlabel('Q')
        self.reflectivitygraphs.axes[0].set_ylabel('R')        
        self.reflectivitygraphs.axes[1].set_xlabel('Q')
        self.reflectivitygraphs.axes[1].set_ylabel('residual')
        self.reflectivitygraphs.axes[2].set_xlabel('z')
        self.reflectivitygraphs.axes[2].set_ylabel('SLD')
        self.ui.gridLayout_3.addWidget(self.reflectivitygraphs.mpl_toolbar)
        
        #add baseparams table widget info
        self.ui.baseparams_tableWidget.setHorizontalHeaderLabels(['number of layers', 'scale', 'background'])
        header = self.ui.baseparams_tableWidget.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)

        #add layerparams table widget info
        numrows = self.ui.layerparams_tableWidget.rowCount()
        numcols = self.ui.layerparams_tableWidget.columnCount()
        self.ui.layerparams_tableWidget.setHorizontalHeaderLabels(['thickness', 'sld', 'iSLD', 'roughness'])
        self.ui.layerparams_tableWidget.setVerticalHeaderLabels(['fronting', '1', 'backing'])
        
        header = self.ui.layerparams_tableWidget.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)
        header = self.ui.layerparams_tableWidget.verticalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)

    def gui_to_parameters(self):
        baseparams = [0, 1, 6]

        numlayers = int(float(self.ui.baseparams_tableWidget.item(0,0).text()))
        parameters = np.zeros(4 * numlayers + 8)
        parameters[0] = numlayers
        fitted_parameters = []
        parameters[1] = float(self.ui.baseparams_tableWidget.item(0, 1).text())
        if not self.ui.baseparams_tableWidget.item(0, 1).checkState():
            fitted_parameters.append(1)         
        parameters[6] = float(self.ui.baseparams_tableWidget.item(0, 2).text())
        if not self.ui.baseparams_tableWidget.item(0, 2).checkState():
            fitted_parameters.append(6)

        parameters[2] = float(self.ui.layerparams_tableWidget.item(0, 1).text())
        if not self.ui.layerparams_tableWidget.item(0, 1).checkState():
            fitted_parameters.append(2)

        parameters[3] = float(self.ui.layerparams_tableWidget.item(0, 2).text())
        if not self.ui.layerparams_tableWidget.item(0, 2).checkState():
            fitted_parameters.append(3)
        
        parameters[4] = float(self.ui.layerparams_tableWidget.item(numlayers + 1, 1).text())
        if not self.ui.layerparams_tableWidget.item(numlayers + 1, 1).checkState():
            fitted_parameters.append(4)

        parameters[5] = float(self.ui.layerparams_tableWidget.item(numlayers + 1, 2).text())
        if not self.ui.layerparams_tableWidget.item(numlayers + 1, 2).checkState():
            fitted_parameters.append(5)

        parameters[7] = float(self.ui.layerparams_tableWidget.item(numlayers + 1, 3).text())
        if not self.ui.layerparams_tableWidget.item(numlayers + 1, 3).checkState():
            fitted_parameters.append(7)

        for pidx in xrange(8, 4 * numlayers + 8):
            row = ((pidx - 8) // 4) + 1
            col = (pidx - 8) % 4
            parameters[pidx] = float(self.ui.layerparams_tableWidget.item(row, col).text())
            if not self.ui.layerparams_tableWidget.item(row, col).checkState():
                fitted_parameters.append(pidx)
        return parameters, np.array(fitted_parameters)
                        
    def gui_from_parameters(self, parameters, fitted_parameters, resize = False):
        baseparamsrow = self.ui.baseparams_tableWidget.currentRow()
        baseparamscol = self.ui.baseparams_tableWidget.currentColumn()

        layerparamsrow = self.ui.layerparams_tableWidget.currentRow()
        layerparamscol = self.ui.layerparams_tableWidget.currentColumn()
        
        self.ui.layerparams_tableWidget.setCurrentCell(-1, -1)
        self.ui.baseparams_tableWidget.setCurrentCell(-1, -1)
        
        baseparams = [0, 1, 6]
        numlayers = int(parameters[0])
        parameters[0] = numlayers
        
        checked = [QtCore.Qt.Checked] * np.size(parameters, 0)
        for val in fitted_parameters:
            checked[val] = QtCore.Qt.Unchecked
        
        self.ui.layerparams_tableWidget.setRowCount(numlayers + 2) 
        #set fronting and backing first
        idx = 0
        wi = QtGui.QTableWidgetItem('')
        self.ui.layerparams_tableWidget.setItem(0, 0, wi)
        wi = QtGui.QTableWidgetItem('')
        self.ui.layerparams_tableWidget.setItem(0, 3, wi)
        wi = QtGui.QTableWidgetItem('')
        self.ui.layerparams_tableWidget.setItem(numlayers + 1, 0, wi)

        wi = QtGui.QTableWidgetItem(str(parameters[2]))
        wi.setCheckState(checked[2])
        self.ui.layerparams_tableWidget.setItem(0, 1, wi)
        wi = QtGui.QTableWidgetItem(str(parameters[3]))
        wi.setCheckState(checked[3])
        self.ui.layerparams_tableWidget.setItem(0, 2, wi)
        
        wi = QtGui.QTableWidgetItem(str(parameters[4]))
        wi.setCheckState(checked[4])
        self.ui.layerparams_tableWidget.setItem(numlayers + 1, 1, wi)
        wi = QtGui.QTableWidgetItem(str(parameters[5]))
        wi.setCheckState(checked[5])
        self.ui.layerparams_tableWidget.setItem(numlayers + 1, 2, wi)

        wi = QtGui.QTableWidgetItem(str(parameters[7]))
        wi.setCheckState(checked[7])
        self.ui.layerparams_tableWidget.setItem(numlayers + 1, 3, wi)
        
        for pidx in xrange(8, 4 * numlayers + 8):
            wi = QtGui.QTableWidgetItem(str(parameters[pidx]))
            wi.setCheckState(checked[pidx])
            row = ((pidx - 8) // 4) + 1
            col = (pidx - 8) % 4
            self.ui.layerparams_tableWidget.setItem(row, col, wi)

        labels = [str(val) for val in xrange(1, numlayers + 1)]
        labels.append('backing')
        labels.insert(0, 'fronting')
        self.ui.layerparams_tableWidget.setVerticalHeaderLabels(labels)
        
        for cidx in xrange(3):
            wi = QtGui.QTableWidgetItem(str(parameters[baseparams[cidx]]))
            wi.setCheckState(checked[baseparams[cidx]])
            self.ui.baseparams_tableWidget.setItem(0, cidx, wi)
                
        self.ui.layerparams_tableWidget.setCurrentCell(layerparamsrow, layerparamscol)
        self.ui.baseparams_tableWidget.setCurrentCell(baseparamsrow, baseparamscol)

    def redraw_dataObject_graphs(self, dataObjects):
        for dataObject in dataObjects:
            if not dataObject:
                continue
            if dataObject.line2D:
               dataObject.line2D.set_data(dataObject.W_q, dataObject.W_ref)
            if dataObject.line2Dfit:
               dataObject.line2Dfit.set_data(dataObject.W_q, dataObject.fit)
            if dataObject.line2Dsld_profile:
                dataObject.line2Dsld_profile.set_data(dataObject.sld_profile[0], dataObject.sld_profile[1])
            
        self.reflectivitygraphs.draw()
                
    def update_gui_modelChanged(self, store = False):
        self.theoretical.evaluate_model(store = True)
        if self.current_dataset is not None:
            energy = self.current_dataset.evaluate_chi2(model = self.theoretical.model)
            self.ui.lineEdit.setText(str(energy))     
        
        self.redraw_dataObject_graphs([self.theoretical, self.current_dataset])
        
            
class MyReflectivityGraphs(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1,1,1), edgecolor=(0,0,0))
        #reflectivity graph
        self.axes = []
        self.axes.append(self.figure.add_subplot(211))
        self.axes[0].autoscale(axis='both', tight = False)
        self.axes[0].set_xlabel('Q')
        self.axes[0].set_ylabel('R')
        self.axes[0].set_yscale('log')
        
        #residual plot
        #, sharex=self.axes[0]
        self.axes.append(self.figure.add_subplot(312))
        self.axes[1].set_visible(False)
        self.axes[1].set_xlabel('Q')
        self.axes[1].set_ylabel('residual')

        #SLD plot
        self.axes.append(self.figure.add_subplot(212))
        self.axes[2].autoscale(axis='both', tight = False)
        self.axes[2].set_xlabel('z')
        self.axes[2].set_ylabel('SLD')

                       
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.subplots_adjust(left=0.1, right=0.95, top = 0.98)
        self.mpl_toolbar = NavigationToolbar(self, parent)

    def visibility_of_plots(self, true_false_triplet):
        """
            if you want to show an individual plot in the window you can use this method to select which
            ones are shown
            true_false_triplet should be a tuple (True or False, True or False, True or False)
            True = display, False = Hide
            true_false_triplet[0] = reflectivity plot
            true_false_triplet[1] = residuals plot
            true_false_triplet[2] = sld plot
        """
        numrows = 0
        for truth in true_false_triplet:
            if truth is True:
                numrows += 1

        upto = 1    
        for idx, val in enumerate(self.axes):
            val.set_visible(true_false_triplet[idx])
            if true_false_triplet[idx] is True:
                val.change_geometry(numrows, 1, upto)            
                upto += 1
            
                            
            