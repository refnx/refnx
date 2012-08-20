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

class MyMainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.errorHandler = QtGui.QErrorMessage()
        self.dataStore = DataStore.DataStore()
        self.modifyGui()

        parameters = np.array([1, 1.0, 0, 0, 2.07, 0, 1e-7, 3, 205, 3.47, 0, 3])
        fitted_parameters = np.array([1,2,3,4,5,6,7,8,9, 10, 11])

        tempq = np.linspace(0.005, 0.5, num = 500)
        tempr = np.ones_like(tempq)
        tempe = np.zeros_like(tempq)
        tempdq  = np.copy(tempq) * 5 / 100.
        dataTuple = (tempq, tempr, tempe, tempdq)
        
        self.theoretical_model = DataStore.dataObject(dataTuple = dataTuple, fitted_parameters = fitted_parameters, parameters = parameters)
        self.theoretical_model.evaluate()
        self.theoretical_model.W_ref = self.theoretical_model.fit
        self.dataStore.addDataObject(self.theoretical_model)
        self.theoretical_model.line2Dfit = self.reflectivitygraphs.axes[0].plot(self.theoretical_model.W_q,
                                                   self.theoretical_model.fit,
                                                    linestyle='-', label = 'theoretical')[0]
        self.theoretical_model.line2Dsld_profile = self.reflectivitygraphs.axes[2].plot(self.theoretical_model.sld_profile[0],
                                                   self.theoretical_model.sld_profile[1],
                                                    linestyle='-')[0]
        self.gui_from_parameters(self.theoretical_model.parameters, self.theoretical_model.fitted_parameters)
        self.theoretical_model.update(parameters, fitted_parameters)
        self.redraw_dataObject_graphs(self.theoretical_model)
        
    @QtCore.Slot()
    def on_actionLoad_Data_triggered(self):
        """
            you load data
        """ 
        theFiles = QtGui.QFileDialog.getOpenFileNames(self,  caption = 'Select Reflectivity Files')[0]
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
    def on_actionRefresh_Datasets_triggered(self):
        """
            you are refreshing existing datasets
        """
        self.dataStore.refresh()
        for key in self.dataStore.dataObjects:
            dataObject = self.dataStore.dataObjects[key]
            dataObject.line2D.set_data(dataObject.W_q, dataObject.W_ref)
        self.reflectivitygraphs.draw()
 
                       
    @QtCore.Slot()
    def on_do_fit_button_clicked(self):
        """
            you should do a fit
        """
#        print self.reflectivitygraphs.axes[0].lines[0].get_marker()
        self.reflectivitygraphs.visibility_of_plots((True, True, True))
        self.reflectivitygraphs.draw()
        
    @QtCore.Slot(unicode)
    def on_dataset_comboBox_currentIndexChanged(self, arg_1):
        """
        dataset to be fitted changed, must update chi2
        """
        print arg_1
       
    @QtCore.Slot(unicode)
    def on_model_comboBox_currentIndexChanged(self, arg_1):
        """
        model selection changed, update view with parameters from model.
        """
        print arg_1
         
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
        
        if row == 0 and col == 0:
            #increase/decrease number of layers
            validator = QtGui.QIntValidator()
            voutput = validator.validate(arg_1.text(), 1)
                        
            if voutput[0] is QtGui.QValidator.State.Acceptable and voutput[1] >= 0:
                #update model
                pass
            else:
                self.errorHandler.showMessage("Number of layers must be integer > 0")
        else:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(arg_1.text(), 1)
            if voutput[0] is QtGui.QValidator.State.Acceptable:
                #update model
                pass
            else:
                self.errorHandler.showMessage("values entered must be numeric")
        if row == 0 and col == 0:
            #perhaps you have to insert rows.
            pass
        
    	self.theoretical_model.parameters = self.gui_to_parameters()
        self.theoretical_model.update(self.theoretical_model.parameters, self.theoretical_model.fitted_parameters)     
        self.redraw_dataObject_graphs(self.theoretical_model)

        
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
            #update model
            pass
        else:
            self.errorHandler.showMessage("values entered must be numeric")
            
        self.theoretical_model.parameters = self.gui_to_parameters()
        self.theoretical_model.update(self.theoretical_model.parameters, self.theoretical_model.fitted_parameters)     
        self.redraw_dataObject_graphs(self.theoretical_model)

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
        
        parameters[1] = float(self.ui.baseparams_tableWidget.item(0, 1).text())
        parameters[6] = float(self.ui.baseparams_tableWidget.item(0, 2).text())
        
        parameters[2] = float(self.ui.layerparams_tableWidget.item(0, 1).text())
        parameters[3] = float(self.ui.layerparams_tableWidget.item(0, 2).text())
        parameters[4] = float(self.ui.layerparams_tableWidget.item(numlayers + 1, 1).text())
        parameters[5] = float(self.ui.layerparams_tableWidget.item(numlayers + 1, 2).text())
        parameters[7] = float(self.ui.layerparams_tableWidget.item(numlayers + 1, 3).text())
        
        for pidx in xrange(8, 4 * numlayers + 8):
            row = ((pidx - 8) // 4) + 1
            col = (pidx - 8) % 4
            parameters[pidx] = float(self.ui.layerparams_tableWidget.item(row, col).text())

        return parameters
                        
    def gui_from_parameters(self, parameters, fitted_parameters):
        baseparams = [0, 1, 6]
        numlayers = int(parameters[0])
        parameters[0] = numlayers

        checked = [QtCore.Qt.Checked] * np.size(parameters, 0)
        for val in fitted_parameters:
            checked[val] = QtCore.Qt.Unchecked

        for cidx in xrange(3):
            wi = QtGui.QTableWidgetItem(parameters[baseparams[cidx]])
            wi.setText(str(parameters[baseparams[cidx]]))
            wi.setCheckState(checked[baseparams[cidx]])
            self.ui.baseparams_tableWidget.setItem(0, cidx, wi)

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
        
    def redraw_dataObject_graphs(self, dataObject):
        if dataObject.line2D:
           dataObject.line2D.set_data(dataObject.W_q, dataObject.W_ref)
        if dataObject.line2Dfit:
           dataObject.line2Dfit.set_data(dataObject.W_q, dataObject.fit)
        if dataObject.line2Dsld_profile:
            dataObject.line2Dsld_profile.set_data(dataObject.sld_profile[0], dataObject.sld_profile[1])
            
        self.reflectivitygraphs.draw()
                

            
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
            
                            
            