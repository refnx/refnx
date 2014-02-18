from __future__ import division
from PySide import QtCore, QtGui
import numpy as np

class LimitsModel(QtCore.QAbstractTableModel):
    def __init__(self, parameters, fitted_parameters, limits, parent = None):
        super(LimitsModel, self).__init__(parent)
        self.parameters = np.copy(parameters)
        self.fitted_parameters = np.unique(fitted_parameters)
        self.limits = np.copy(limits)
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return 4
        
    def rowCount(self, parent = QtCore.QModelIndex()):
        return np.size(self.fitted_parameters)
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False
            
        row = index.row()
        col = index.column()
                    
        uniquevals = np.unique(self.fitted_parameters)
        if role == QtCore.Qt.DisplayRole:
            if col == 0:
                return str(self.fitted_parameters[row])
            if col == 1:
                return str(self.parameters[self.fitted_parameters[row]])
            if col == 2:
                return str(self.limits[0, self.fitted_parameters[row]])
            if col == 3:
                return str(self.limits[1, self.fitted_parameters[row]])
                
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None
    
        if orientation == QtCore.Qt.Vertical:
            return ''            
        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'parameter'
            if section == 1:
                return 'value'
            if section == 2:
                return 'lower limit'
            if section == 3:
                return 'upper limit'
        return None

    def flags(self, index):
        row = index.row()
        col = index.column()
        
        if col == 0 or col == 1:
        	return QtCore.Qt.NoItemFlags
        
    	return (QtCore.Qt.ItemIsEditable |
    	           QtCore.Qt.ItemIsEnabled |
    	            QtCore.Qt.ItemIsSelectable)
    	            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()
    
        if not index.isValid():
            return False
                    
        if col < 0 or col > 3:
            return False
                                                
        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)
            if voutput[0] == QtGui.QValidator.State.Acceptable:
                if col == 2:
                    self.limits[0, self.fitted_parameters[row]] = voutput[1]
                if col == 3:
                    self.limits[1, self.fitted_parameters[row]] = voutput[1]                    
            else:
                return False
            
        self.dataChanged.emit(index, index)
        return True