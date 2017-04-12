from __future__ import division
from PyQt5 import QtCore, QtGui
from refnx.reflect import ReflectModel, Slab, SLD


class BaseModel(QtCore.QAbstractTableModel):
    """
    a model for displaying in a QtGui.QTableView
    """

    def __init__(self, model, parent=None):
        super(BaseModel, self).__init__(parent)
        self.model = model

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 1

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2

    def flags(self, index):
        return (QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsUserCheckable |
                QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid():
            return False
        row = index.row()
        col = index.column()

        par = [self.model.scale, self.model.bkg][col]
        validator_limits = (0, 1000000000., 15)

        if row != 0 or col < 0 or col > 1:
            return False

        if role == QtCore.Qt.CheckStateRole:
            if value == QtCore.Qt.Checked:
                par.vary = True
            else:
                par.vary = False

        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            if col == 0:
                validator.setRange(*validator_limits)

            voutput = validator.validate(value, 1)
            if voutput[0] == QtGui.QValidator.Acceptable:
                par.value = float(voutput[1])
            else:
                return False

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None

        col = index.column()

        if index.row() != 0 or col < 0 or col > 1:
            return None

        par = [self.model.scale, self.model.bkg][col]

        if role == QtCore.Qt.DisplayRole:
            return str(par.value)

        if role == QtCore.Qt.CheckStateRole:
            if par.vary:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Horizontal:
            if section == 0:
                return 'scale factor'
            if section == 1:
                return 'background'
        return None


class LayerModel(QtCore.QAbstractTableModel):
    """
        a model for displaying in a QtGui.QTableView
    """
    def __init__(self, model, parent=None):
        super(LayerModel, self).__init__(parent)
        self.model = model

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.model.structure)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 5

    def flags(self, index):
        nlayers = len(self.model.structure) - 2
        row = index.row()
        col = index.column()

        if row == 0 and col in [0, 3, 4]:
            return QtCore.Qt.NoItemFlags

        if row == nlayers + 1 and col in [0, 4]:
            return QtCore.Qt.NoItemFlags

        return (QtCore.Qt.ItemIsEditable |
                QtCore.Qt.ItemIsUserCheckable |
                QtCore.Qt.ItemIsEnabled |
                QtCore.Qt.ItemIsSelectable)

    def rename_layers(self):
        structure = self.model.structure
        for idx, layer in enumerate(structure):
            layer.thick.name = '%s - thick' % idx
            layer.rough.name = '%s - rough' % idx
            layer.vfsolv.name = '%s - volfrac solvent' % idx
            layer.sld.real.name = '%s - sld' % idx
            layer.sld.imag.name = '%s - isld' % idx

    def add_layer(self, insertpoint):
        structure = self.model.structure

        slab = SLD(0.)(0., 3.)
        structure.insert(insertpoint + 1, slab)

        self.beginInsertRows(QtCore.QModelIndex(), insertpoint + 1,
                             insertpoint + 1)
        self.rename_layers()
        self.endInsertRows()
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def remove_layer(self, which_layer):
        structure = self.model.structure

        self.beginRemoveRows(QtCore.QModelIndex(), which_layer, which_layer)
        structure.pop(which_layer)
        self.rename_layers()
        self.endRemoveRows()
        # need to tell all listeners that we've removed a layer
        # (e.g. limits_model, force plots to be updated, etc)
        self.dataChanged.emit(QtCore.QModelIndex(), QtCore.QModelIndex())

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        structure = self.model.structure

        row = index.row()
        col = index.column()

        slab = structure[row]
        par = [slab.thick, slab.sld.real, slab.sld.imag,
               slab.rough, slab.vfsolv][col]

        validator_limits = [(0, 1000000000., 15),
                            (-100000000., 100000000., 15),
                            (-100000000., 100000000., 15),
                            (0, 100000000., 15),
                            (0, 1., 15)]

        if not index.isValid():
            return False

        if col < 0 or col > 4:
            return False

        if role == QtCore.Qt.CheckStateRole:
            if value == QtCore.Qt.Checked:
                par.vary = True
            else:
                par.vary = False

        if role == QtCore.Qt.EditRole:
            validator = QtGui.QDoubleValidator()
            validator.setRange(*validator_limits[col])
            voutput = validator.validate(value, 1)
            if voutput[0] == QtGui.QValidator.Acceptable:
                par.value = float(voutput[1])
            else:
                return False

        self.dataChanged.emit(index, index)
        return True

    def data(self, index, role=QtCore.Qt.DisplayRole):
        structure = self.model.structure
        nlayers = len(structure) - 2

        if not index.isValid():
            return False

        row = index.row()
        col = index.column()
        slab = structure[row]
        par = [slab.thick, slab.sld.real, slab.sld.imag,
               slab.rough, slab.vfsolv][col]

        if row == 0 and col in [0, 3, 4]:
            return None

        if row == nlayers + 1 and col in [0, 4]:
            return None

        if role == QtCore.Qt.DisplayRole:
            return str(par.value)

        if role == QtCore.Qt.CheckStateRole:
            if par.vary:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        """ Set the headers to be displayed. """
        if role != QtCore.Qt.DisplayRole:
            return None

        if orientation == QtCore.Qt.Vertical:
            nlayers = len(self.model.structure) - 2
            if section == 0:
                return 'fronting'
            elif section == nlayers + 1:
                return 'backing'
            else:
                return str(section)

        if orientation == QtCore.Qt.Horizontal:
            headers = ['thickness', 'SLD / 1e-6', 'iSLD / 1e-6',
                       'roughness', 'vfsolv']
            return headers[section]

        return None
