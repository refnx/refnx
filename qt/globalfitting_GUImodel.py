from __future__ import division
from PySide import QtCore, QtGui
import pyplatypus.analysis.fitting as fitting
import pyplatypus.analysis.globalfitting as globalfitting
import numpy as np

class GlobalFitting_DataModel(QtCore.QAbstractTableModel):
    def __init__(self, parent = None):
        super(GlobalFitting_DataModel, self).__init__(parent)
        self.numdatasets = 0
        self.dataset_names = []
        self.numparams = []
        self.fitplugins = []
        self.linkages = []
        
    def rowCount(self, parent = QtCore.QModelIndex()):
        val = 3
        if len(self.numparams):
            val += max(self.numparams)
        return val
        
    def columnCount(self, parent = QtCore.QModelIndex()):
        return len(self.dataset_names)
                
    def flags(self, index):
        row = index.row()
        col = index.column()
        
        if row == 0:
            return False
        if row == 1:
           return (QtCore.Qt.ItemIsEditable |
                   QtCore.Qt.ItemIsEnabled |
                   QtCore.Qt.ItemIsSelectable)        
        if row == 2:
           return (QtCore.Qt.ItemIsEditable |
                   QtCore.Qt.ItemIsEnabled |
                   QtCore.Qt.ItemIsSelectable)

        if row < self.numparams[col] + 3:
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        
        return False
                
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()

        if role == QtCore.Qt.EditRole:
            if row == 1:
                self.fitplugins[col] = value   
            if row == 2:
                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)

                if voutput[0] is QtGui.QValidator.State.Acceptable and int(voutput[1]) >= 0:
                    val = int(voutput[1])
                    
                    if val > max(self.numparams):
                        currentrows = max(self.numparams)
                        self.beginInsertRows(QtCore.QModelIndex(),
                                             self.rowCount(),
                                             3 + val - 1)
                        self.endInsertRows()
                    else:
                    #you might have to shrink the number of parameters
                        numparams_copy = list(self.numparams)
                        numparams_copy[col] = val
                        if max(numparams_copy) < max(self.numparams):
                            self.beginRemoveRows(QtCore.QModelIndex(),
                                                 max(numparams_copy),
                                                 max(self.numparams) - 1)
                            
                            self.endRemoveRows()
                                                            
                    self.numparams[col] = val
        
        self.dataChanged.emit(index, index)
        return True
        
    def link_selection(self, indices):
        
        #first convert indices to entries like 'd0:p1'
        indices_list = list()
        for index in indices:
            row = index.row() - 3
            col = index.column()
            if row > -1:
                indices_list.append('d' + str(col) + ':p' + str(row))
            
        indices_list.sort()
        
        #if there is only one entry, then there is no linkage
        if len(indices_list) < 2:
            return
    
    def is_already_linked(self, parameter):
        for linkage in self.linkages:
            if parameter in linkage:
                return True
                
        return False        
        
    def unlink_parameter(self, parameter):
        pass
    
    def add_DataSet(self, dataset):
        if dataset in self.dataset_names:
            return
            
        self.beginInsertColumns(QtCore.QModelIndex(),
                                self.numdatasets,
                                self.numdatasets)
        self.insertColumns(self.numdatasets, 1)
        self.endInsertColumns()
        self.numdatasets += 1
        self.numparams.append(0)
        self.fitplugins.append('')
        self.dataset_names.append(dataset)
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False
        
        if not len(self.dataset_names):
            return None
            
        row = index.row()
        col = index.column()
                    
        if role == QtCore.Qt.DisplayRole:
            if row == 0:
                return self.dataset_names[col]
            if row == 1:
                return self.fitplugins[col]
            if row == 2:
                return self.numparams[col]
            
            if row < self.numparams[col] + 3:
                return 'd' + str(col) + ':p' + str(row - 3)
                
    def generate_linkage_matrix(self):
        pass
                                    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            if section == 0:
                return 'dataset'
            if section == 1:
                return 'fitting plugin'
            if section == 2:
                return 'number of parameters'
        return None
        

class FitPluginItemDelegate(QtGui.QStyledItemDelegate):
    def __init__(self, plugins, parent=None):
        super(FitPluginItemDelegate, self).__init__(parent)
        self.plugins = plugins
        
    def set_plugins(self, plugins):
        self.plugins = plugins
        
    def createEditor(self, parent, option, index):
        return PluginComboBox(self.plugins, parent)
    
class PluginComboBox(QtGui.QComboBox):
    def __init__(self, plugins, parent=None):
            super(PluginComboBox, self).__init__(parent)
            for plugin in plugins:
                self.addItem(plugin['name'])


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