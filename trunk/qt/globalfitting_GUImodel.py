from __future__ import division
from PySide import QtCore, QtGui
import pyplatypus.analysis.fitting as fitting
import pyplatypus.analysis.globalfitting as globalfitting

class GlobalFitting_DataModel(QtCore.QAbstractTableModel):
    def __init__(self, parent = None):
        super(GlobalFitting_DataModel, self).__init__(parent)
                
    def rowCount(self, parent = QtCore.QModelIndex()):
        return 0
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 0
            
    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        pass
                
    def flags(self, index):
        return  (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        return None               
                    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        return None
                

class GlobalFitting_ParamModel(QtCore.QAbstractTableModel):
    def __init__(self, parent = None):
        super(GlobalFitting_ParamModel, self).__init__(parent)
                
    def rowCount(self, parent = QtCore.QModelIndex()):
        return 0
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 0
            
    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        pass
                
    def flags(self, index):
        return  (QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        return None               
                    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        return None