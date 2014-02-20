from __future__ import division
from PySide import QtCore, QtGui
import pyplatypus.analysis.fitting as fitting
import pyplatypus.analysis.globalfitting as globalfitting
import pyplatypus.analysis.reflect as reflect
import numpy as np
import pyplatypus.analysis.model as model
import re


def position_in_linkage(linkages, parameter):
    for idx, linkage in enumerate(linkages):
        if parameter in linkage:
            return idx
    return -1
    
def is_linked(linkages, parameter):
    idx = position_in_linkage(linkages, parameter)                
    if idx == -1:
        return (False, False, '')
    #now test to see if it's a master linkage
    if linkages[idx].index(parameter) == 0:
        return (True, True, parameter)
    else:
        return (True, False, linkages[idx][0])
                
def RC_to_parameter(row, col):
    return 'd' + str(col) + ':p' + str(row)

def parameter_to_RC(parameter):
    regex = re.compile("[a-z]([0-9]*):[a-z]([0-9]*)")
    r = regex.search(parameter)
    groups = r.groups()
    col = int(groups[0])
    row = int(groups[1])

    return (row, col)
    
def generate_linkage_matrix(linkages, numparams):
    #initialise linkage matrix
    totalparams = np.insert(np.cumsum(numparams), 0, 0)
    
    linkage_matrix = np.zeros([len(numparams), max(numparams)],
                              dtype='int64')
    linkage_matrix -= 1
   
    cumulative_param = 0
    linkages.sort()

    for i in range(len(numparams)):
        for j in range(numparams[i]):
            isLinked, isMaster, master_link = is_linked(linkages,
                                           RC_to_parameter(j, i))
            if isLinked and isMaster is False:
                p, d = parameter_to_RC(master_link)
                linkage_matrix[i, j] = linkage_matrix[d, p]
            else:
                linkage_matrix[i, j] = cumulative_param
                cumulative_param += 1
 
    return linkage_matrix
    
    
class GlobalFitting_DataModel(QtCore.QAbstractTableModel):
    
    added_DataSet = QtCore.Signal(unicode)
    #which dataset did you remove
    removed_DataSet = QtCore.Signal(int)
    #new number of params, which dataset
    added_params = QtCore.Signal(int, int)
    #new number of params, which dataset
    removed_params = QtCore.Signal(int, int)
    #changed linkages
    changed_linkages = QtCore.Signal(list)
    #resized table
    resized_rows = QtCore.Signal(int, int)
    #changed fitplugins
    changed_fitplugin = QtCore.Signal(list)
    
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
        if row == 1 or row == 2:
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
                self.changed_fitplugin.emit(self.fitplugins)
            if row == 2:
                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)

                if voutput[0] is QtGui.QValidator.State.Acceptable and \
                              int(voutput[1]) >= 0:
                    
                    val = int(voutput[1])
                    if self.fitplugins[col] == 'default':
                        if (val - 8) % 4:
                            msgBox = QtGui.QMessageBox()
                            msgBox.setText('The default fitting plugin is reflectivity. The number of parameters must be 4N+8.')
                            msgBox.exec_()
                            return False
                    
                    oldparams = self.numparams[col]
                 

                    if val > max(self.numparams):
                        #this change causes the table to grow
                        self.numparams[col] = val                    
                        currentrows = max(self.numparams)
                        self.beginInsertRows(QtCore.QModelIndex(),
                                             self.rowCount(),
                                             3 + val - 1)
                        self.endInsertRows()
                        self.resized_rows.emit(currentrows, val)
                    elif val < max(self.numparams):
                        #there is at least one other parameter vector bigger
                        self.numparams[col] = val                        
                    else:
                        #val == max(self.numparams)
                        #this may be the largest, but another may be the same
                        #you might have to shrink the number of parameters
                        numparams_copy = list(self.numparams)
                        numparams_copy[col] = val
                        if max(numparams_copy) < max(self.numparams):
                            #it was the biggest, now we have to shrink
                            self.beginRemoveRows(QtCore.QModelIndex(),
                                                 max(numparams_copy),
                                                 max(self.numparams) - 1)                            
                            self.endRemoveRows()
                            self.numparams[col] = val
                            self.resized_rows.emit(max(self.numparams),
                                                    max(numparams_copy))
                        else:
                            #there was one other that was just as big,
                            #don't shrink
                            self.numparams[col] = val
                            
                    if oldparams > val:
                        for row in range(val, self.numparams[col]):
                            self.unlink_parameter(RC_to_parameter(row, col))              
                        self.removed_params.emit(val, col)
                    elif oldparams < val:
                        self.added_params.emit(val, col)
                        
        
        self.dataChanged.emit(index, index)
        return True
        
    def convert_indices_to_parameter_list(self, indices):
        #first convert indices to entries like 'd0:p1'
        parameter_list = list()
        for index in indices:
            row = index.row() - 3
            col = index.column()
            if row > -1:
                parameter_list.append('d' + str(col) + ':p' + str(row))
        
        return parameter_list    
    
    def link_selection(self, indices):        
        parameter_list = self.convert_indices_to_parameter_list(indices)
        
        #if there is only one entry, then there is no linkage to add.
        if len(parameter_list) < 2:
            return
            
        parameter_list.sort()
        parameter_list.reverse()
        
        for parameter in parameter_list:
            if is_linked(self.linkages, parameter)[0]:
                self.unlink_parameter(parameter) 
        
        #now link the parameters
        parameter_list.sort()
        self.linkages.append(parameter_list)
        
        self.remove_single_linkages()
        self.modelReset.emit()
        self.changed_linkages.emit(self.linkages)
        
    def unlink_selection(self, indices):
        parameter_list = self.convert_indices_to_parameter_list(indices)
                                
        parameter_list.sort()
        parameter_list.reverse()
        
        for parameter in parameter_list:
            if is_linked(self.linkages, parameter)[0]:
                self.unlink_parameter(parameter) 
        
        self.remove_single_linkages()
        self.modelReset.emit()
        self.changed_linkages.emit(self.linkages)
                
    def unlink_parameter(self, parameter):
        #find out which entry in the linkage list it is.
        #It should only be in there once.
        idx = position_in_linkage(self.linkages, parameter)
        isLinked, isMaster, master_link = is_linked(self.linkages, parameter)
        
        if isLinked:
            linkage = self.linkages[idx]
            del(linkage[linkage.index(parameter)])
                
        #if the parameter was a 'master link', then delete all other linkages
        if isMaster:
            del(self.linkages[idx])
        
        self.changed_linkages.emit(self.linkages)
    
    def remove_single_linkages(self):
        for idx, linkage in enumerate(self.linkages):
            if len(linkage) == 1:
                del[self.linkages[idx]]
    
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
        self.fitplugins.append('default')
        self.dataset_names.append(dataset)
        self.added_DataSet.emit(dataset)
                
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
                parameter = RC_to_parameter(row - 3, col)
                isLinked, isMaster, master_link = \
                            is_linked(self.linkages, parameter)
                if isLinked and isMaster is False:
                    idx = position_in_linkage(self.linkages, parameter)
                    return 'linked: ' + master_link
                
                return parameter
                                    
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
        ComboBox = QtGui.QComboBox(parent)
        pluginlist = [plugin['name'] for plugin in self.plugins]
        ComboBox.addItems(pluginlist)
        return ComboBox
    
    
class GlobalFitting_ParamModel(QtCore.QAbstractTableModel):
    def __init__(self, parent = None):
        super(GlobalFitting_ParamModel, self).__init__(parent)
        self.numdatasets = 0
        self.dataset_names = []
        self.numparams = []
        self.fitplugins = []
        self.linkages = []
        self.models = []
    
    def changed_fitplugin(self, fitplugins):
        self.fitplugins = fitplugins
        
    def changed_linkages(self, linkages):
        self.linkages = linkages
        for linkage in linkages:
            masterParam, masterDataSet = parameter_to_RC(linkage[0])
            for link in linkage:
                param, which_dataset = parameter_to_RC(link)
                self.models[which_dataset].parameters[param] = \
                    self.models[masterDataSet].parameters[masterParam]
                    
    def added_DataSet(self, name):
        self.numdatasets += 1
        self.dataset_names.append(name)
        self.numparams.append(0)
        self.fitplugins.append('default')
        self.models.append(model.Model(np.zeros_like([])))
        self.beginInsertColumns(QtCore.QModelIndex(),
                                self.numdatasets - 1,
                                self.numdatasets - 1)
        self.endInsertColumns()
        
    def removed_DataSet(self, which_dataset):
        self.numdatasets -= 1
        del(self.dataset_names[which_dataset])
        self.beginRemoveColumns(QtCore.QModelIndex(),
                                which_dataset,
                                which_dataset)
        del[self.models[which_dataset]]
        self.endRemoveColumns()
        
    def added_params(self, newparams, which_dataset):
        oldparams = self.numparams[which_dataset]
        self.numparams[which_dataset] = newparams
        model = self.models[which_dataset]
        model.parameters = \
                 np.resize(model.parameters,
                           newparams)
                           
        model.parameters[oldparams:] = 0.
        model.fitted_parameters = \
                np.append(model.fitted_parameters,
                          np.arange(oldparams, newparams))
                          
        start = self.createIndex(oldparams, which_dataset)
        finish = self.createIndex(newparams, which_dataset)
        self.dataChanged.emit(start, finish)
        
    def removed_params(self, newparams, which_dataset):
        oldparams = self.numparams[which_dataset]
        self.numparams[which_dataset] = newparams
        self.models[which_dataset].parameters = \
                 np.resize(self.models[which_dataset].parameters,
                           newparams)
        start = self.createIndex(newparams, which_dataset)
        self.dataChanged.emit(start, start)
        
    def resized_rows(self, oldrows, newrows):
        self.modelReset.emit()
                
    def rowCount(self, parent = QtCore.QModelIndex()):
        if len(self.numparams):
            return max(self.numparams)
        return 0
            
    def columnCount(self, parent = QtCore.QModelIndex()):
        return self.numdatasets
            
    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        pass
                
    def flags(self, index):
        row = index.row()
        col = index.column()
        
        if row > self.numparams[col] - 1:
            return False
        
        parameter = RC_to_parameter(row, col)
        isLinked, isMaster, master_link = is_linked(self.linkages, parameter)
                    
        theflags = (QtCore.Qt.ItemIsUserCheckable |
                    QtCore.Qt.ItemIsEnabled)
    	           
        if isLinked and isMaster is False:
            pass
        else:
            theflags = (theflags |
                       QtCore.Qt.ItemIsEditable |
                       QtCore.Qt.ItemIsSelectable)

    	return theflags
    	            
    def setData(self, index, value, role = QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()

        parameter = RC_to_parameter(row, col)
        isLinked, isMaster, master_link = is_linked(self.linkages, parameter)
        
        model = self.models[col]
        
        if role == QtCore.Qt.CheckStateRole:    
            #if the default plugin is the reflectivity one,
            #don't allow the value to be changed. 
            #this is the number of layers
            if row == 0 and self.fitplugins[col] == 'default':
                model.fitted_parameters = np.append(model.fitted_parameters,
                                                    0)

            if value == QtCore.Qt.Checked:
                model.fitted_parameters = np.delete(model.fitted_parameters,
                                       np.where(model.fitted_parameters == row))
            else:
                model.fitted_parameters = np.append(model.fitted_parameters,
                                                    row)
        
        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)
            if voutput[0] is not QtGui.QValidator.State.Acceptable:
                return False

            val = voutput[1]
            
            if row == 0 and self.fitplugins[col] == 'default':
                testparams = np.copy(model.parameters)
                testparams[0] = float(int(val))
                if not reflect.is_proper_Abeles_input(testparams):
                    msgBox = QtGui.QMessageBox()
                    msgBox.setText('The default fitting plugin for this model is reflectivity. This parameter must be (numparams - 8)/4.')
                    msgBox.exec_()
                    return False

    
            model.parameters[row] = val
            if isMaster and isLinked:
                linkage = self.linkages[position_in_linkage(self.linkages,
                                                            parameter)]
                for link in linkage:
                    row, col = parameter_to_RC(link)
                    self.models[col].parameters[row] = val
                
        self.dataChanged.emit(index, index)
        return True
                
    def data(self, index, role=QtCore.Qt.DisplayRole):
        row = index.row()
        col = index.column()
        
        if row > self.numparams[col] - 1:
            return None
            
        parameter = RC_to_parameter(row, col)
        isLinked, isMaster, master_link = is_linked(self.linkages, parameter)
        model = self.models[col]
        
        if role == QtCore.Qt.DisplayRole:
            if row < self.numparams[col]:
                return str(model.parameters[row])
        
        if role == QtCore.Qt.CheckStateRole:
            if row == 0 and self.fitplugins[col] == 'default':
                return QtCore.Qt.Checked
            if isLinked and isMaster is False:
                return None
            if row in model.fitted_parameters:
                return QtCore.Qt.Unchecked
            else:
                return QtCore.Qt.Checked
               
        return None               
                    
    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Horizontal:
            return self.dataset_names[section]
            
        if orientation == QtCore.Qt.Vertical:
            return int(section)

        return None
        