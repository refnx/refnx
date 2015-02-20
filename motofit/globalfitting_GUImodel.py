from __future__ import division
from PySide import QtCore, QtGui
import refnx.analysis.curvefitter as curvefitter
from refnx.analysis import CurveFitter, GlobalFitter
import refnx.analysis.reflect as reflect
import numpy as np
from lmfit import Parameters
import re


def is_linked(linkages, dataset, parameter):
    """
    Given a set of linkages and a parameter determine whether the parameter
    is linked, is a master link, and what the master link is.

    Parameters
    ----------
    linkages: sequence
        The linkages for the analysis. 'd1:abc=d2:def' would link parameter
        `def` of dataset 2 with parameter `abc` of dataset 1.
    dataset: integer
        The dataset index the parameter belongs to.
    parameter: str
        The parameter name of interest.

    Returns
    -------
    (is_linked, is_master, master_link): bool, bool, str
    """
    is_linked, is_master, master_link = False, False, ''

    if not len(linkages):
        return is_linked, is_master, master_link

    for linkage in linkages:
        d_master, p_master, d_slave, p_slave = linkage_to_CR(linkage)

        if dataset == d_master and p_master == parameter:
            is_linked = True
            is_master = True

        if d_slave == dataset and parameter == p_slave:
            is_linked = True
            master_link = 'd%d:%s'%(d_master, p_master)

    return is_linked, is_master, master_link


def RC_to_parameter(row, col):
    return 'd%dp%d' % (col, row)


def linkage_ref_to_CR(ref):
    dp_string = 'd([0-9]+):([0-9a-zA-Z_]+)'
    ref_regex = re.compile(dp_string)

    ref_search = ref_regex.search(ref)

    if ref_regex is not None:
        vals = ref_search.groups()
        d_master = int(vals[0])
        p_master = vals[1]

        return d_master, p_master

    return None


def linkage_to_CR(linkage):
    """
    Get the dataset and parameter numbers for a linkage

    Parameters
    ----------
    linkage: str
        linkage specified as 'dN:abc=dM:def'

    Returns
    -------
    A tuple (N, M, A, B)
    """
    dp_string = 'd([0-9]+):([0-9a-zA-Z_]+)'
    linkage_regex = re.compile(dp_string + '=' + dp_string)

    linkage_search = linkage_regex.search(linkage)
    if linkage_search is not None:
        vals = linkage_search.groups()
        d_master = int(vals[2])
        p_master = vals[3]
        d_slave = int(vals[0])
        p_slave = vals[1]
        return d_master, p_master, d_slave, p_slave
    else:
        return ValueError("linkage needs to be in form 'dN:abc=dM:def'")


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

    data_model_changed = QtCore.Signal()

    def __init__(self, gf_settings, parent=None):
        super(GlobalFitting_DataModel, self).__init__(parent)
        self.gf_settings = gf_settings

    def rowCount(self, parent=QtCore.QModelIndex()):
        val = 2
        if len(self.gf_settings.nparams):
            val += max(self.gf_settings.nparams)
        return val

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.gf_settings.ndatasets

    def flags(self, index):
        row = index.row()
        col = index.column()

        if row in [0, 1]:
            return (QtCore.Qt.ItemIsEditable |
                    QtCore.Qt.ItemIsEnabled |
                    QtCore.Qt.ItemIsSelectable)

        if row < self.gf_settings.nparams[col] + 2:
            return QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        return False

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        row = index.row()
        col = index.column()
        params = self.gf_settings.parameters[col]
        names = params.keys()

        if role == QtCore.Qt.EditRole:
            if row == 0:
                self.gf_settings.fitplugins[col] = value
            if row == 1:
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
                                             oldparams + 2,
                                             2 + val - 1)
                        self.endInsertRows()
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
                        else:
                            # there was one other that was just as big,
                            # don't shrink
                            self.gf_settings.nparams[col] = val

                    if oldparams > val:
                        for row in range(val, oldparams):
                            self.unlink_parameter(col, RC_to_parameter(row, col))
                            # remove parameters

                        names_to_remove = params.keys()[val: oldparams]
                        map(params.pop, names_to_remove)

                    elif oldparams < val:
                        # add parameters
                        fitter = self.gf_settings.fitplugins[col]
                        if hasattr(fitter, 'parameter_names'):
                            new_names = fitter.parameter_names(nparams=val)
                            new_names = new_names[oldparams:]
                        else:
                            new_names = ['p%d' % i for i in range(oldparams,
                                                                  val)]

                        for i in range(val - oldparams):
                            params.add_many((new_names[i], 0, True, None, None,
                                             None))

        self.dataChanged.emit(index, index)
        self.data_model_changed.emit()
        return True

    def convert_indices_to_parameter_list(self, indices):
        # first convert indices to entries like 'd0p1'
        parameter_list = list()
        for index in indices:
            row = index.row() - 2
            col = index.column()
            params = self.gf_settings.parameters[col]
            names = params.keys()
            name = names[row]
            if row > -1:
                parameter_list.append((col, name))

        return parameter_list

    def link_selection(self, indices):
        parameter_list = self.convert_indices_to_parameter_list(indices)

        # if there is only one entry, then there is no linkage to add.
        if len(parameter_list) < 2:
            return

        for dataset, parameter in parameter_list:
            isLinked, isMaster, master_link = is_linked(
                              self.gf_settings.linkages,
                              dataset,
                              parameter)
            if isLinked or isMaster:
                self.unlink_parameter(dataset, parameter)

        # now link the parameters
        master_link = parameter_list[0]
        for dataset, parameter in parameter_list[1:]:
            link = 'd%d:%s=d%d:%s' % (dataset, parameter,
                                    master_link[0], master_link[1])
            self.gf_settings.linkages.append(link)

        self.modelReset.emit()

    def unlink_selection(self, indices):
        parameter_list = sorted(
            self.convert_indices_to_parameter_list(indices))

        for dataset, parameter in parameter_list:
            if is_linked(self.gf_settings.linkages,
                         dataset,
                         parameter)[0]:
                self.unlink_parameter(dataset, parameter)

        self.modelReset.emit()

    def unlink_parameter(self, dataset, parameter):
        # remove all entries that contain the parameter
        linkages = self.gf_settings.linkages

        isLinked, isMaster, master_link = is_linked(linkages,
                                                    dataset,
                                                    parameter)
        if not isLinked:
            return

        param_regex = re.compile('d%d:%s' % (dataset, parameter))
        linkages_to_remove = filter(param_regex.search,
                                    linkages)
        new_linkages = [val for val in linkages if val
                        not in linkages_to_remove]

        self.gf_settings.linkages = new_linkages

    def add_DataSet(self, dataset):
        if dataset in self.gf_settings.dataset_names:
            return

        self.beginInsertColumns(QtCore.QModelIndex(),
                                self.gf_settings.ndatasets,
                                self.gf_settings.ndatasets)
        self.insertColumns(self.gf_settings.ndatasets, 1)
        self.gf_settings.ndatasets += 1
        self.gf_settings.nparams.append(0)
        self.gf_settings.parameters.append(Parameters())
        self.gf_settings.fitplugins.append('default')
        self.gf_settings.dataset_names.append(dataset)
        self.endInsertColumns()

        self.data_model_changed.emit()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return False

        if not len(self.gf_settings.dataset_names):
            return None

        row = index.row()
        col = index.column()
        parameters = self.gf_settings.parameters[col]
        names = parameters.keys()

        if role == QtCore.Qt.DisplayRole:
            if row == 0:
                return self.gf_settings.fitplugins[col]
            if row == 1:
                return self.gf_settings.nparams[col]

            if row < self.gf_settings.nparams[col] + 2:
                name = names[row - 2]
                isLinked, isMaster, master_link = \
                    is_linked(self.gf_settings.linkages, col, name)
                if isLinked and isMaster is False:
                    return '%s is linked: %s' %(name, master_link)

                return name

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            if section == 0:
                return 'fitting plugin'
            if section == 1:
                return 'number of parameters'

        if orientation == QtCore.Qt.Horizontal:
            if self.gf_settings.ndatasets:
                return self.gf_settings.dataset_names[section]

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

    def __init__(self, gf_settings, parent=None):
        super(GlobalFitting_ParamModel, self).__init__(parent)
        self.gf_settings = gf_settings

    def data_model_changed(self):
        self.modelReset.emit()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if len(self.gf_settings.nparams):
            return max(self.gf_settings.nparams)
        return 0

    def columnCount(self, parent=QtCore.QModelIndex()):
        return self.gf_settings.ndatasets * 2

    def insertRows(self, position, rows=1, index=QtCore.QModelIndex()):
        pass

    def flags(self, index):
        row = index.row()
        col = index.column()
        which_dataset = col // 2

        if row > self.gf_settings.nparams[which_dataset] - 1:
            return False

        if (col % 2) == 0:
            return False

        params = self.gf_settings.parameters[which_dataset]
        name = params.keys()[row]

        isLinked, isMaster, master_link = is_linked(self.gf_settings.linkages,
                                                    which_dataset,
                                                    name)

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
        which_dataset = col // 2

        params = self.gf_settings.parameters[which_dataset]
        name = params.keys()[row]
        param = params[name]
        isLinked, isMaster, master_link = is_linked(self.gf_settings.linkages,
                                                    which_dataset,
                                                    name)

        # 0, 2, 4, ... are the names of the parameters
        if not (col % 2):
            return False

        if role == QtCore.Qt.CheckStateRole:
            # if the default plugin is the reflectivity one,
            # don't allow the value to be changed.
            # this is the number of layers
            if row == 0 and self.gf_settings.fitplugins[which_dataset] == 'default':
                params[name].vary = False

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

            if row == 0 and self.gf_settings.fitplugins[which_dataset] == 'default':
                if (len(params) - 8) / 4 != val:
                    msgBox = QtGui.QMessageBox()
                    msgBox.setText(
                        "The default fitting plugin for this model is"
                        " reflectivity. This parameter must be "
                        "(numparams - 8)/4.")
                    msgBox.exec_()
                    return False

            param.value = val
            if isMaster and isLinked:
                for linkage in self.gf_settings.linkages:
                    d_master, p_master, d_slave, p_slave = linkage_to_CR(
                                                                    linkage)
                    if d_master == which_dataset and p_master == name:
                        slave_params = self.gf_settings.parameters[d_slave]
                        slave_params[p_slave].value = val

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        row = index.row()
        col = index.column()
        which_dataset = col // 2

        if row > self.gf_settings.nparams[which_dataset] - 1:
            return None

        params = self.gf_settings.parameters[which_dataset]
        name = params.keys()[row]

        isLinked, isMaster, master_link = is_linked(self.gf_settings.linkages,
                                                    which_dataset,
                                                    name)

        if role == QtCore.Qt.DisplayRole:
            if not (col % 2):
                return name
            if isLinked and isMaster is False:
                d_master, p_master = linkage_ref_to_CR(master_link)
                master_params = self.gf_settings.parameters[d_master]
                return str(master_params[p_master].value)
            if row < self.gf_settings.nparams[which_dataset]:
                return str(params[name].value)

        if role == QtCore.Qt.CheckStateRole:
            if col % 2 == 0:
                return None
            if row == 0 and self.gf_settings.fitplugins[which_dataset] == 'default':
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
            which_dataset = section // 2
            if section % 2:
                return self.gf_settings.dataset_names[which_dataset]
            else:
                return 'names'
        if orientation == QtCore.Qt.Vertical:
            return int(section)

        return None
