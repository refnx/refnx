from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow

from numpy import arange, sin, pi
import random
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

    @QtCore.Slot()
    def on_actionLoad_Data_triggered(self):
        """
            you load data
        """ 
        theFiles = QtGui.QFileDialog.getOpenFileNames(self,  caption = 'Select Reflectivity Files')[0]
        for file in theFiles:
            dataObject = self.dataStore.loadDataObject(file)
            lineInstance = self.reflectivitygraphs.axes[0].plot(dataObject.W_q,
                                                                 dataObject.W_ref,
                                                                   markersize=5,
                                                                    marker='o',
                                                                     linestyle='',
                                                                      label = dataObject.name)
            dataObject.line2D = lineInstance[0]
            
            self.reflectivitygraphs.draw()
 
 
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
        print self.reflectivitygraphs.axes[0].lines[0].get_marker()
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

        initvals = ["1", "1.0", "1e-7"]
        for cidx in xrange(3):
            wi = QtGui.QTableWidgetItem(initvals[cidx])
            if cidx:
                wi.setCheckState(QtCore.Qt.Unchecked)
            self.ui.baseparams_tableWidget.setItem(0, cidx, wi)

        #add layerparams table widget info
        numrows = self.ui.layerparams_tableWidget.rowCount()
        numcols = self.ui.layerparams_tableWidget.columnCount()
        self.ui.layerparams_tableWidget.setHorizontalHeaderLabels(['thickness', 'sld', 'iSLD', 'roughness'])
        self.ui.layerparams_tableWidget.setVerticalHeaderLabels(['fronting', '1', 'backing'])
        
        initvals = [" ", "0", "0", " ", "25", "3.47", "0", "3", " ", "2.07", "0", "3"]
        idx = 0
        for ridx in xrange(self.ui.layerparams_tableWidget.rowCount()):
            for cidx in xrange(self.ui.layerparams_tableWidget.columnCount()):
                wi = QtGui.QTableWidgetItem(initvals[idx])
                if not ((ridx, cidx) == (0,0) or (ridx, cidx) == (0, 3) or (ridx, cidx) == (numrows - 1, 0)):
                    wi.setCheckState(QtCore.Qt.Unchecked)
                self.ui.layerparams_tableWidget.setItem(ridx, cidx, wi)
                idx += 1
                
        header = self.ui.layerparams_tableWidget.horizontalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)
        header = self.ui.layerparams_tableWidget.verticalHeader()
        header.setResizeMode(QtGui.QHeaderView.Stretch)
        self.ui.dataset_comboBox.addItem("theoretical")
        
class MyReflectivityGraphs(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None):
        self.figure = Figure(facecolor=(1,1,1), edgecolor=(0,0,0))
        #reflectivity graph
        self.axes = []
        self.axes.append(self.figure.add_subplot(211))
        self.axes[0].set_xlabel('Q')
        self.axes[0].set_ylabel('R')
        self.axes[0].set_yscale('log')
        
        #residual plot
        self.axes.append(self.figure.add_subplot(312, sharex=self.axes[0]))
        self.axes[1].set_visible(False)
        self.axes[1].set_xlabel('Q')
        self.axes[1].set_ylabel('residual')

        #SLD plot
        self.axes.append(self.figure.add_subplot(212))
        self.axes[2].set_xlabel('z')
        self.axes[2].set_ylabel('SLD')
                       
        # We want the axes cleared every time plot() is called
#        for ax in self.axes:
#            ax.hold(False)
        
        self.update_figure()

        #
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
        self.figure.subplots_adjust(left=0.1, right=0.95, top = 0.98)
        self.mpl_toolbar = NavigationToolbar(self, parent)

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        pass
#         l = [ random.randint(0, 10) for i in xrange(4) ]
#         self.axes[0].plot([0, 1, 2, 3], l)
#         self.axes[1].plot([1, 2, 3, 4], l)
#         self.axes[2].plot([2, 3, 4, 5], l)

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
            
                            
            