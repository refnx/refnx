from __future__ import division
from PySide import QtCore, QtGui
import refnx.analysis.curvefitter as curvefitter
from refnx.analysis import CurveFitter, GlobalFitter
import refnx.analysis.reflect as reflect
import numpy as np
from lmfit import Parameters
import re


def is_linked(linkages, parameter):
    """
    Given a set of linkages and a parameter determine whether the parameter
    is linked, is a master link, and what the master link is.

    Parameters
    ----------
    linkages: sequence
        The linkages for the analysis. 'd0p1:d1p2' would link parameter 2
        of dataset 1 with parameter 1 of dataset 0.
    parameter: str
        The parameter of interest

    Returns
    -------
    (is_linked, is_master, master_link): bool, bool, str
    """
    is_linked, is_master, master_link = False, False, ''

    if not len(linkages):
        return is_linked, is_master, master_link

    dataset, param = parameter_to_CR(parameter)

    for linkage in linkages:
        d_master, p_master, d_slave, p_slave = linkage_to_CR(linkage)

        if dataset == d_master and param == p_master:
            is_linked = True
            is_master = True
            master_link = 'd%dp%d' % (dataset, param)
            break
        elif dataset == d_slave and param == p_slave:
            is_linked = True
            master_link = 'd%dp%d' % (d_master, p_slave)
            break

    return is_linked, is_master, master_link


def RC_to_parameter(row, col):
    return 'd%dp%d' % (col, row)


def linkage_to_CR(linkage):
    """
    Get the dataset and parameter numbers for a linkage

    Parameters
    ----------
    linkage: str
        linkage specified as 'dNpM:dApB'

    Returns
    -------
    A tuple (N, M, A, B)
    """

    linkage_regex = re.compile('d([0-9]+)p([0-9]+):d([0-9]+)p([0-9]+)')
    linkage_search = linkage_regex.search(linkage)
    if linkage_search is not None:
        vals = [int(val) for val in linkage_search.groups()]
        d_master, p_master, d_slave, p_slave = vals
        return d_master, p_master, d_slave, p_slave
    else:
        return ValueError("Parameter needs to be in form 'dNpM'")


def parameter_to_CR(parameter):
    """
    Get the dataset and parameter number for a given parameter

    Parameters
    ----------
    parameter: str
        Parameter specified as 'dNpM'

    Returns
    -------
    A tuple (N, M)
    """

    parameter_regex = re.compile('d([0-9]+)p([0-9]+)')
    par_search = parameter_regex.search(parameter)
    if par_search:
        return (int(par_search.groups()[0]),
                int(par_search.groups()[1]))
    else:
        return ValueError("Parameter needs to be in form 'dNpM'")


class GlobalFitting_Settings(object):
    # a lightweight class to enable pickling of the globalfitting state
    def __init__(self):
        self.ndatasets = 0
        self.dataset_names = []
        self.nparams = []
        self.fitplugins = []
        self.linkages = []
        self.parameters = []


class GlobalFitting_DataModel(QtCore.QAbstractTableModel):

    added_DataSet = QtCore.Signal(unicode)
    # which dataset did you remove
    removed_DataSet = QtCore.Signal(int)
    # new number of params, which dataset
    added_params = QtCore.Signal(int, int)
    # new number of params, which dataset
    removed_params = QtCore.Signal(int, int)
    # changed linkages
    changed_linkages = QtCore.Signal(list)
    # resized table
    resized_rows = QtCore.Signal(int, int)
    # changed fitplugins
    changed_fitplugin = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(GlobalFitting_DataModel, self).__init__(parent)
        self.gf_settings = GlobalFitting_Settings()

    def rowCount(self, parent=QtCore.QModelIndex()):
        val = 3
        if len(self.gf_settings.nparams):
            val += max(self.gf_settings.nparams)
        return val

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.gf_settings.ndatasets

    def flags(self, index):
        row = index.row()
        col = index.column()

        if row == 0:
            return False
        if row == 1 or row == 2:
            return (QtCore.Qt.ItemIsEditable |
                    QtCore.Qt.ItemIsEnabled |
                    QtCore.Qt.ItemIsSelectable)

        if row < self.gf_settings.nparams[col] + 3:
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        return False

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()

        if role == QtCore.Qt.EditRole:
            if row == 1:
                self.gf_settings.fitplugins[col] = value
                self.changed_fitplugin.emit(self.gf_settings.fitplugins)
            if row == 2:
                validator = QtGui.QIntValidator()
                voutput = validator.validate(value, 1)

                if (voutput[0] is QtGui.QValidator.State.Acceptable
                    and int(voutput[1]) >= 0):

                    val = int(voutput[1])
                    if self.gf_settings.fitplugins[col] == 'default':
                        if (val - 8) % 4:
                            msgBox = QtGui.QMessageBox()
                            msgBox.setText(
                                'The default fitting plugin is reflectivity.'
                                ' The number of parameters must be 4N+8.')
                            msgBox.exec_()
                            return False

                    oldparams = self.gf_settings.nparams[col]

                    if val > max(self.gf_settings.nparams):
                        # this change causes the table to grow
                        self.gf_settings.nparams[col] = val
                        currentrows = max(self.gf_settings.nparams)
                        self.beginInsertRows(QtCore.QModelIndex(),
                                             oldparams + 3,
                                             3 + val - 1)
                        self.endInsertRows()
                        self.resized_rows.emit(currentrows, val)
                    elif val < max(self.gf_settings.nparams):
                        # there is at least one other parameter vector bigger
                        self.gf_settings.nparams[col] = val
                    else:
                        #val == max(self.numparams)
                        # this may be the largest, but another may be the same
                        # you might have to shrink the number of parameters
                        numparams_copy = list(self.gf_settings.nparams)
                        numparams_copy[col] = val
                        if max(numparams_copy) < max(self.gf_settings.nparams):
                            # it was the biggest, now we have to shrink
                            self.beginRemoveRows(QtCore.QModelIndex(),
                                             max(numparams_copy),
                                             max(self.gf_settings.nparams) - 1)
                            self.endRemoveRows()
                            self.gf_settings.nparams[col] = val
                            self.resized_rows.emit(
                                        max(self.gf_settings.nparams),
                                        max(numparams_copy))
                        else:
                            # there was one other that was just as big,
                            # don't shrink
                            self.gf_settings.nparams[col] = val

                    if oldparams > val:
                        for row in range(val, self.gf_settings.nparams[col]):
                            self.unlink_parameter(RC_to_parameter(row, col))
                        self.removed_params.emit(val, col)
                    elif oldparams < val:
                        self.added_params.emit(val, col)

        self.dataChanged.emit(index, index)
        return True

    def convert_indices_to_parameter_list(self, indices):
        # first convert indices to entries like 'd0p1'
        parameter_list = list()
        for index in indices:
            row = index.row() - 3
            col = index.column()
            if row > -1:
                parameter_list.append('d%dp%d' % (col, row))

        return parameter_list

    def link_selection(self, indices):
        parameter_list = self.convert_indices_to_parameter_list(indices)

        # if there is only one entry, then there is no linkage to add.
        if len(parameter_list) < 2:
            return

        parameter_list.sort()
        parameter_list.reverse()

        for parameter in parameter_list:
            if is_linked(self.gf_settings.linkages, parameter)[0]:
                self.unlink_parameter(parameter)

        # now link the parameters
        parameter_list.sort()
        master_link = parameter_list[0]
        for parameter in parameter_list[1:]:
            link = master_link + ':' + parameter
            self.gf_settings.linkages.append(link)

        self.modelReset.emit()
        self.changed_linkages.emit(self.gf_settings.linkages)

    def unlink_selection(self, indices):
        parameter_list = sorted(
            self.convert_indices_to_parameter_list(indices))

        parameter_list.reverse()

        for parameter in parameter_list:
            if is_linked(self.gf_settings.linkages, parameter)[0]:
                self.unlink_parameter(parameter)

        self.modelReset.emit()
        self.changed_linkages.emit(self.gf_settings.linkages)

    def unlink_parameter(self, parameter):
        # remove all entries that contain the parameter
        linkages = self.gf_settings.linkages

        isLinked, isMaster, master_link = is_linked(linkages,
                                                    parameter)
        if not isLinked:
            return

        param_regex = re.compile(parameter)
        linkages_to_remove = filter(param_regex.search,
                                    linkages)
        new_linkages = [val for val in linkages if val
                        not in linkages_to_remove]

        self.gf_settings.linkages = new_linkages
        self.changed_linkages.emit(self.gf_settings.linkages)

    def add_DataSet(self, dataset):
        if dataset in self.gf_settings.dataset_names:
            return

        self.beginInsertColumns(QtCore.QModelIndex(),
                                self.gf_settings.ndatasets,
                                self.gf_settings.ndatasets)
        self.insertColumns(self.gf_settings.ndatasets, 1)
        self.endInsertColumns()
        self.gf_settings.ndatasets += 1
        self.gf_settings.nparams.append(0)
        self.gf_settings.fitplugins.append('default')
        self.gf_settings.dataset_names.append(dataset)
        self.added_DataSet.emit(dataset)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False

        if not len(self.gf_settings.dataset_names):
            return None

        row = index.row()
        col = index.column()

        if role == QtCore.Qt.DisplayRole:
            if row == 0:
                return self.gf_settings.dataset_names[col]
            if row == 1:
                return self.gf_settings.fitplugins[col]
            if row == 2:
                return self.gf_settings.nparams[col]

            if row < self.gf_settings.nparams[col] + 3:
                parameter = RC_to_parameter(row - 3, col)
                isLinked, isMaster, master_link = \
                    is_linked(self.gf_settings.linkages, parameter)
                if isLinked and isMaster is False:
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
        pluginlist = self.plugins.keys()
        ComboBox.addItems(pluginlist)
        return ComboBox


class GlobalFitting_ParamModel(QtCore.QAbstractTableModel):

    def __init__(self, parent=None):
        super(GlobalFitting_ParamModel, self).__init__(parent)
        self.gf_settings = GlobalFitting_Settings()

    def changed_fitplugin(self, fitplugins):
        self.gf_settings.fitplugins = fitplugins

    def changed_linkages(self, linkages):
        self.gf_settings.linkages = linkages
        for linkage in linkages:
            d_master, p_master, d_slave, p_slave = linkage_to_CR(linkage)
            # TODO update parameters

    def added_DataSet(self, dataset):
        if dataset in self.gf_settings.dataset_names:
            return

        self.gf_settings.ndatasets += 1
        self.gf_settings.nparams.append(0)
        self.gf_settings.fitplugins.append('default')
        self.gf_settings.dataset_names.append(dataset)
        self.gf_settings.parameters.append(Parameters())
        self.beginInsertColumns(QtCore.QModelIndex(),
                                self.gf_settings.ndatasets - 1,
                                self.gf_settings.ndatasets - 1)
        self.endInsertColumns()

    def removed_DataSet(self, which_dataset):
        linkages = self.gf_settings.linkages

        # remove all the linkages referring to this dataset.
        for i in range(len(linkages - 1), -1, -1):
            linkage = linkages[i]
            d_master, p_master, d_slave, p_slave = linkage_to_CR(linkage)
            if which_dataset in [d_master, d_slave]:
                del(linkages[i])

        self.gf_settings.ndatasets -= 1
        del(self.gf_settings.dataset_names[which_dataset])
        self.beginRemoveColumns(QtCore.QModelIndex(),
                                which_dataset,
                                which_dataset)
        del(self.gf_settings.parameters[which_dataset])
        self.endRemoveColumns()

    def added_params(self, newparams, which_dataset):
        oldparams = self.gf_settings.nparams[which_dataset]
        self.gf_settings.nparams[which_dataset] = newparams
        params = self.gf_settings.parameters[which_dataset]
        for i in range(oldparams, newparams):
            params.add_many(('d%dp%d' % (which_dataset, i),
                             0,
                             True,
                             None,
                             None,
                             None))

        start = self.createIndex(oldparams, which_dataset)
        finish = self.createIndex(newparams, which_dataset)
        self.dataChanged.emit(start, finish)

    def removed_params(self, newparams, which_dataset):
        oldparams = self.gf_settings.nparams[which_dataset]
        self.gf_settings.nparams[which_dataset] = newparams
        params = self.gf_settings.parameters[which_dataset]
        names_to_remove = params.keys()[newparams:]
        map(params.pop, names_to_remove)

        start = self.createIndex(newparams, which_dataset)
        self.dataChanged.emit(start, start)

    def resized_rows(self, oldrows, newrows):
        self.modelReset.emit()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if len(self.gf_settings.nparams):
            return max(self.gf_settings.nparams)
        return 0

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.gf_settings.ndatasets

    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        pass

    def flags(self, index):
        row = index.row()
        col = index.column()

        if row > self.gf_settings.nparams[col] - 1:
            return False

        parameter = RC_to_parameter(row, col)
        isLinked, isMaster, master_link = is_linked(self.gf_settings.linkages,
                                                    parameter)

        theflags = (QtCore.Qt.ItemIsUserCheckable |
                    QtCore.Qt.ItemIsEnabled)

        if isLinked and isMaster is False:
            pass
        else:
            theflags = (theflags
                        | QtCore.Qt.ItemIsEditable
                        | QtCore.Qt.ItemIsSelectable)

        return theflags

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()

        parameter = RC_to_parameter(row, col)
        isLinked, isMaster, master_link = is_linked(self.gf_settings.linkages,
                                                    parameter)

        params = self.gf_settings.parameters[col]
        name = params.keys()[row]

        if role == QtCore.Qt.CheckStateRole:
            # if the default plugin is the reflectivity one,
            # don't allow the value to be changed.
            # this is the number of layers
            if row == 0 and self.gf_settings.fitplugins[col] == 'default':
                param[name].vary = False

            if value == QtCore.Qt.Checked:
                params[name].vary = False
            else:
                params[name].vary = True

        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            voutput = validator.validate(value, 1)
            if voutput[0] is not QtGui.QValidator.State.Acceptable:
                return False

            val = float(voutput[1])

            if row == 0 and self.gf_settings.fitplugins[col] == 'default':
                if not reflect.is_proper_Abeles_input(
                                    curvefitter.values(params)):
                    msgBox = QtGui.QMessageBox()
                    msgBox.setText(
                        "The default fitting plugin for this model is"
                        " reflectivity. This parameter must be "
                        "(numparams - 8)/4.")
                    msgBox.exec_()
                    return False

            params[name].value = val
            if isMaster and isLinked:
                for linkage in self.gf_settings.linkages:
                    d_master, p_master, d_slave, p_slave = linkage_to_CR(
                                                                    linkage)
                    if d_master == col and p_master == row:
                        slave_params = self.gf_settings.parameters[col]
                        name = slave_params.keys()[p_slave]
                        slave_params[name].value = val

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        row = index.row()
        col = index.column()

        if row > self.gf_settings.nparams[col] - 1:
            return None

        parameter = RC_to_parameter(row, col)
        isLinked, isMaster, master_link = is_linked(self.gf_settings.linkages,
                                                    parameter)
        params = self.gf_settings.parameters[col]
        name = params.keys()[row]

        if role == QtCore.Qt.DisplayRole:
            if row < self.gf_settings.nparams[col]:
                return str(params[name].value)

        if role == QtCore.Qt.CheckStateRole:
            if row == 0 and self.gf_settings.fitplugins[col] == 'default':
                return QtCore.Qt.Checked
            if isLinked and isMaster is False:
                return None
            if params[name].vary:
                return QtCore.Qt.Unchecked
            else:
                return QtCore.Qt.Checked

        return None

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Horizontal:
            return self.gf_settings.dataset_names[section]

        if orientation == QtCore.Qt.Vertical:
            return int(section)

        return None
