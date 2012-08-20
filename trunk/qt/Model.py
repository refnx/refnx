from __future__ import division
import numpy as np
from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow
import pyplatypus.dataset.DataStore as DataStore

def theoreticalModel_dataObject(qvals = None, qstart = 0.005, qend = 0.5, points = 100, parameters = None, fitted_parameters = None):
    dataObject = DataStore.dataObject(parameters = parameters, fitted_parameters = fitted_parameters)
    if qvals is not None:
        tempq = qvals
        tempr = np.copy(qvals)
        tempe = np.zeroes_like(tempq)
    else:
        tempq = np.linspace(qstart, qend, num = 100)
        tempr = np.ones_like(tempq)
        tempr = np.zeroes_like(tempq)
        
    dataObject.set_data((tempq, tempr, tempe))
    dataObject.name = '_theoretical_'
    
    return dataObject
    
