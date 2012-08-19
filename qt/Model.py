from __future__ import division
import numpy as np
from PySide import QtCore, QtGui
from MotofitUI import Ui_MainWindow
import pyplatypus.dataset.DataStore as DataStore

def theoreticalModel_dataObject(qvals = None, qstart = 0.005, qend = 0.5, points = 100):
    dataObject = DataStore.dataObject()
    if qvals is not None:
        tempq = qvals
        tempr = np.copy(qvals)
    else:
        tempq = np.linspace(qstart, qend, num = 100)
        tempr = np.ones_like(tempq)
        
    dataObject.set_data((tempq, tempr))
    dataObject.name = '_theoretical_'
    
    return dataObject
    


class theoreticalModel(object):
    def __init__(self, qvals = None, qstart = 0.005, qend = 0.5):
        pass
        
    def gui_to_parameters(self):
        pass
        
    def parameters_to_gui(self):
        pass
    