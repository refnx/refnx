from copy import deepcopy
import os.path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from refnx._lib import flatten
from refnx.dataset import ReflectDataset
from refnx.reflect._app.dataobject import DataObject
from refnx.reflect._app.datastore import DataStore
from refnx.reflect import Slab, LipidLeaflet, SLD


def component_class(component):
    """
    Give the node type for a given component
    """
    if isinstance(component, Slab):
        return SlabNode
    elif isinstance(component, LipidLeaflet):
        return LipidLeafletNode


class Node(object):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        self._data = data
        self._children = []
        # the parent node
        self._parent = parent

        # the qabstractitemmodel
        self._model = model

    def insertChild(self, i, item):
        item._model = self._model
        self._children.insert(i, item)

    def appendChild(self, item):
        item._model = self._model
        self._children.append(item)

    def popChild(self, row):
        return self._children.pop(row)

    def child(self, row):
        return self._children[row] if row < len(self._children) else None

    def childCount(self):
        return len(self._children)

    def columnCount(self):
        # name, value, sigma, lower, upper, expr
        # vary will be displayed in a checkbox with value
        return 6

    def data(self, column, role=QtCore.Qt.DisplayRole):
        return None

    def setData(self, col, val, role=QtCore.Qt.EditRole):
        return False

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        return flags

    def parent(self):
        return self._parent

    def row(self):
        if self._parent:
            # this index method is searching a *list* for an item, and isn't
            # meant to produce a QModelIndex
            v = self._parent._children.index(self)
            return v

        return 0

    def _row_parent(self):
        """
        Work out which row of the parent the current node lies in,
        returning the row and the parent.
        """
        return self.row(), self._parent

    def hierarchy(self):
        """
        The top down node hierarchy up to, and including, this node.
        """
        h = [self]
        rp = self._row_parent()
        while rp[1] is not None:
            h.append(rp[1])
            rp = rp[1]._row_parent()
        h.reverse()
        return h

    def row_indices(self):
        hierarchy = self.hierarchy()
        indices = []
        for node in hierarchy:
            indices.append(node.row())
        return indices

    def descendants(self):
        # yield all of the descendants of this node
        for el in self._children:
            yield el
            if el._children:
                yield from el.descendants()

    @property
    def index(self):
        """
        The QModelIndex for this node
        """
        # first work out the hierarchy of the nodes right back to the root
        # node, working out which row of the parent each node is in.
        # uses recursion
        a = []
        rp = self._row_parent()
        while rp[1] is not None:
            a.append(rp)
            rp = rp[1]._row_parent()

        # once you've got the hierarchy then you have to reverse it, and create
        # a model index right back down again.
        a.reverse()
        idx = QtCore.QModelIndex()
        for v in a:
            idx = self._model.index(v[0], 0, idx)
        return idx


class ParNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super(ParNode, self).__init__(data, model, parent)

    @property
    def parameter(self):
        return self._data

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column in [1, 3, 4]:
            flags |= QtCore.Qt.ItemIsEditable

        if column == 1:
            flags |= QtCore.Qt.ItemIsUserCheckable

        return flags

    def data(self, column, role=QtCore.Qt.DisplayRole):
        p = self.parameter
        d = [p.name, p.value, p.stderr, p.bounds.lb, p.bounds.ub,
             repr(p.constraint)]

        if (role == QtCore.Qt.CheckStateRole):
            if column != 1:
                return None
            if p.vary:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        if role == QtCore.Qt.DisplayRole and column < len(d):
            return d[column]

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        p = self.parameter

        # we want to use a checkbox to say if a parameter is varying
        if role == QtCore.Qt.CheckStateRole and column == 1:
            try:
                p.vary = value == QtCore.Qt.Checked
            except RuntimeError:
                # can't try and hold a parameter that has a constraint
                False

            return True

        # parse and fill out parameter values/limits
        if role == QtCore.Qt.EditRole:
            if type(value) is str:
                validator = QtGui.QDoubleValidator()
                voutput = validator.validate(value, 1)
            else:
                voutput = [QtGui.QValidator.Acceptable, float(value)]

            if voutput[0] == QtGui.QValidator.Acceptable:
                if column == 1:
                    p.value = float(voutput[1])
                elif column == 3:
                    p.bounds.lb = float(voutput[1])
                elif column == 4:
                    p.bounds.ub = float(voutput[1])
            else:
                return False

        return True


class PropertyNode(Node):
    # an object that displays/edits some attribute of its parent node
    # it is not a ParNode.
    def __init__(self, data, model, parent=QtCore.QModelIndex(),
                 validators=()):
        super(PropertyNode, self).__init__(data, model, parent)
        # here self._data is the attribute name
        self.attribute_type = type(getattr(parent._data, data))
        self.validators = validators

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 1:
            flags |= QtCore.Qt.ItemIsEditable

            if self.attribute_type is bool:
                flags |= QtCore.Qt.ItemIsUserCheckable

        return flags

    def data(self, column, role=QtCore.Qt.DisplayRole):
        d = getattr(self._parent._data, self._data)
        if (role == QtCore.Qt.CheckStateRole and column == 1 and
                self.attribute_type is bool):

            if d:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        if role == QtCore.Qt.DisplayRole and column == 1:
            return d
        if role == QtCore.Qt.DisplayRole and column == 0:
            return self._data

    def columnCount(self):
        return 2

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        if (role == QtCore.Qt.CheckStateRole and column == 1 and
                self.attribute_type is bool):

            if value == QtCore.Qt.Checked:
                setattr(self._parent._data, self._data, True)
            else:
                setattr(self._parent._data, self._data, False)
            return True

        # parse and fill out parameter values/limits
        if role == QtCore.Qt.EditRole and len(self.validators) and column == 1:
            for validator in self.validators:
                voutput = validator.validate(value, 1)
                if voutput[0] == QtGui.QValidator.Acceptable:
                    setattr(self._parent._data, self._data, voutput[1])
                    return True

        return False


class ComponentNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super(ComponentNode, self).__init__(data, model, parent)

        for par in flatten(data.parameters):
            pn = ParNode(par, model, parent=self)
            self.appendChild(pn)

    @property
    def component(self):
        return self._data

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 0:
            flags |= QtCore.Qt.ItemIsEditable

        return flags

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None
        return self._data.name

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return False

        if column:
            return False

        self.component.name = value
        self._model.dataChanged.emit(self.index, self.index)
        return True


class SlabNode(ComponentNode):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super(SlabNode, self).__init__(data, model, parent)

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 0:
            flags |= QtCore.Qt.ItemIsEditable

        return flags

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None
        return self._data.name

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return False

        if column:
            return False

        self.component.name = value
        self._model.dataChanged.emit(self.index, self.index)
        return True


class StructureNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super(StructureNode, self).__init__(data, model, parent)
        for component in data:
            self.appendChild(
                component_class(component)(component, model, self))

    @property
    def structure(self):
        return self._data

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None
        # structures name
        return 'structure'

    def remove_component(self, row):
        self._model.beginRemoveRows(self.index, row, row)
        self.structure.pop(row)
        self.popChild(row)
        self._model.endRemoveRows()

    def insert_component(self, row, component):
        n = component_class(component)(component, self._model, self)

        self._model.beginInsertRows(self.index, row, row)
        self.insertChild(row, n)
        self.structure.insert(row, component)
        self._model.endInsertRows()


class ReflectModelNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super(ReflectModelNode, self).__init__(data, model, parent=parent)

        n = ParNode(data.scale, model, self)
        self.appendChild(n)
        n = ParNode(data.bkg, model, self)
        self.appendChild(n)
        n = ParNode(data.dq, model, self)
        self.appendChild(n)
        n = StructureNode(data.structure, model, self)
        self.appendChild(n)

    def childCount(self):
        # scale, bkg, res, structure
        return 4

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None
        # structures name
        return 'model'


class DatasetNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super(DatasetNode, self).__init__(data, model, parent)

    @property
    def dataset(self):
        return self._data

    def refresh(self):
        self.dataset.refresh()

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        return flags

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if role == QtCore.Qt.DisplayRole:
            if column == 0:
                return self._data.name
            return None


class DataObjectNode(Node):
    def __init__(self, data_object, model, parent=QtCore.QModelIndex()):
        super(DataObjectNode, self).__init__(data_object, model, parent=parent)

        self.chi2 = float(np.nan)
        self.visible = data_object.graph_properties.visible

        n = DatasetNode(data_object.dataset, model, self)
        self.appendChild(n)

        # append the ReflectModel.
        if data_object.model is not None:
            # already existing
            n = ReflectModelNode(data_object.model, model, self)
            self.appendChild(n)

    @property
    def data_object(self):
        return self._data

    def refresh(self):
        self.child(0).refresh()
        # recalculate number of points
        index = self.parent().index
        idx = self._model.index(self.row(), 1, index)
        idx1 = self._model.index(self.row(), 1, index)

        self._model.dataChanged.emit(idx, idx1)

    def set_reflect_model(self, model):
        if model is not None:
            n = ReflectModelNode(model, self._model, self)
            # no ReflectModel, append the ReflectModel as a child
            if len(self._children) == 1:
                self._model.beginInsertRows(self.index, 1, 1)
                self.appendChild(n)
                self.data_object.model = model
                self._model.endInsertRows()
            # there is a pre-existing ReflectModel, overwrite.
            elif len(self._children) == 2:
                # remove and add
                self._model.beginRemoveRows(self.index, 1, 1)
                self.popChild(1)
                self._model.endRemoveRows()
                self._model.beginInsertRows(self.index, 1, 1)
                self.appendChild(n)
                self.data_object.model = model
                self._model.endInsertRows()

    def set_dataset(self, dataset):
        if dataset is not None:
            n = DatasetNode(dataset, self._model, parent=self)
            self.data_object.dataset = dataset
            self._children[0] = n
            index = self._children[0].index
            self._model.dataChanged.emit(index, index)

    def columnCount(self):
        return 4

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole and column == 3:
            self.visible = value == QtCore.Qt.Checked
            return True

        return True

    def data(self, column, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.CheckStateRole and column == 3:
            if self.visible:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        if role == QtCore.Qt.DisplayRole:
            if column == 0:
                return self._data.name
            elif column == 1:
                return 'points: %d' % len(self._data.dataset)
            elif column == 2:
                return 'chi2: %g' % self.chi2
            elif column == 3:
                return 'display'

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 3:
            flags |= QtCore.Qt.ItemIsUserCheckable

        return flags


class ContainerNode(Node):
    # set up with a datastore
    def __init__(self, data, model, parent=None):
        super(ContainerNode, self).__init__(data, model, parent)
        for data_object in data:
            n = DataObjectNode(data_object, model, parent=self)
            self.appendChild(n)

    @property
    def datastore(self):
        return self._data

    def load_data(self, filename):
        n_objects = len(self.datastore)
        data_object = self.datastore.load(filename)

        if data_object is not None:
            # if you load a new dataset then add it to the model
            if n_objects - len(self.datastore):
                position = self.childCount()
                self._model.beginInsertRows(self.index, position, position)
                n = DataObjectNode(data_object, self._model, parent=self)
                self.appendChild(n)
                self._model.endInsertRows()
            else:
                # if the number of datasets didn't increase, then it was
                # refreshed, update the datasetnode
                position = self.datastore.names.index(data_object.name)
                dataset_node = self.child(position).child(0).index
                self._model.dataChanged.emit(dataset_node, dataset_node)

        return data_object

    def set_data_object(self, data_object):
        if data_object is None:
            return

        # see if it's already present
        if data_object.name in self.datastore.names:
            row = self.datastore.names.index(data_object.name)
            self.child(row).set_dataset(data_object.dataset)
            self.child(row).set_reflect_model(data_object.model)
        else:
            position = self.childCount()
            self._model.beginInsertRows(self.index, position, position)
            self.datastore.data_objects[data_object.name] = data_object
            n = DataObjectNode(data_object, self._model, parent=self)
            self.appendChild(n)
            self._model.endInsertRows()

    def remove_data_object(self, name):
        try:
            row = self.datastore.names.index(name)
        except ValueError:
            return

        self._model.beginRemoveRows(self.index, row, row)
        # remove from the backing datastore and pop from the nodelist
        self.datastore.remove_dataset(name)
        self.popChild(row)
        self._model.endRemoveRows()

    def refresh(self):
        for data_object_node in self._children:
            data_object_node.refresh()


class TreeModel(QtCore.QAbstractItemModel):
    def __init__(self, data, parent=None):
        super(TreeModel, self).__init__(parent)
        self._rootnode = ContainerNode(data, self)
        self._data = data

    @property
    def datastore(self):
        return self._data

    def columnCount(self, parent):
        if parent.isValid():
            return parent.internalPointer().columnCount()
        else:
            return self._rootnode.columnCount()

    def data(self, index, role):
        if not index.isValid():
            return None

        item = index.internalPointer()
        return item.data(index.column(), role=role)

    def setData(self, index, value, role=QtCore.Qt.EditRole):
        if not index.isValid():
            return None

        item = index.internalPointer()
        col = index.column()

        # devolve setting of data to Node
        ok = item.setData(col, value, role=role)
        if ok:
            self.dataChanged.emit(index, index, [role])

        return ok

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.NoItemFlags

        # we want the parameter value, lb, ub values to be editable
        item = index.internalPointer()

        return item.flags(index.column())

    def headerData(self, section, orientation, role):
        headers = ["Name", "value", 'sigma', 'lb', 'ub', 'constraint']
        if (orientation == QtCore.Qt.Horizontal and
                role == QtCore.Qt.DisplayRole):
            return headers[section]

        return None

    def index(self, row, column, parent=QtCore.QModelIndex()):
        if parent is None:
            parent = QtCore.QModelIndex()

        if not parent.isValid():
            parentItem = self._rootnode
        else:
            parentItem = parent.internalPointer()

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        else:
            return QtCore.QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QtCore.QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self._rootnode:
            return QtCore.QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)

    def rowCount(self, parent=QtCore.QModelIndex()):
        if not parent.isValid():
            parentItem = self._rootnode
        else:
            parentItem = parent.internalPointer()

        return parentItem.childCount()

    def rebuild(self):
        self._rootnode = ContainerNode(self._data, self)
        self.modelReset.emit()

    def load_data(self, filename):
        data_object = self._rootnode.load_data(filename)
        return data_object

    def remove_data_object(self, name):
        self._rootnode.remove_data_object(name)

    def data_object_row(self, name):
        # returns a row in the rootnode given a name
        try:
            return self.datastore.names.index(name)
        except ValueError:
            return None

    def data_object_node(self, name):
        # returns a DataObjectNode given a name
        row = self.data_object_row(name)
        if row is not None:
            return self._rootnode.child(row)
        else:
            return None

    def node_from_row_indices(self, row_indices):
        # retrieve a descendant node given a series of row indices
        # the first entry in row_indices should always be 0/None.
        node = self._rootnode
        for idx in row_indices[1:]:
            node = node.child(idx)

        return node

    def snapshot(self, snapshot_name):
        original = self.datastore['theoretical']
        dataset = ReflectDataset()
        dataset.data = (original.dataset.x,
                        original.model.model(original.dataset.x,
                                             x_err=dataset.x_err))
        dataset.name = snapshot_name

        new_model = deepcopy(original.model)
        new_model.name = snapshot_name

        # if the snapshot already exists then overwrite it.
        if snapshot_name in self.datastore.names:
            row = self.datastore.index(snapshot_name)
            self._rootnode.child[row].set_dataset(dataset)
            self._rootnode.child[row].set_reflect_model(new_model)
        else:
            # otherwise you have to add it.
            data_object = DataObject(dataset)
            data_object.model = new_model
            self._rootnode.set_data_object(data_object)

        return data_object

    def refresh(self):
        self._rootnode.refresh()


class TreeFilter(QtCore.QSortFilterProxyModel):
    def __init__(self, tree_model, parent=None):
        super(TreeFilter, self).__init__(parent)
        self.tree_model = tree_model

    def filterAcceptsRow(self, row, index):
        idx = self.tree_model.index(row, 0, index)

        if not idx.isValid():
            return False

        item = idx.internalPointer()
        if isinstance(item, DatasetNode):
            return False

        # filter out resolution parameter if the dataset has x_err
        if (isinstance(item, ParNode) and
                isinstance(item.parent(), ReflectModelNode)):
            # find the dataset
            data_object_node = find_data_object(item.index)
            dataset = data_object_node.data_object.dataset
            if dataset.x_err is not None and item.row() == 2:
                return False

        # filter out parameters for the fronting/backing media
        if (isinstance(item, ParNode) and
                isinstance(item.parent(), SlabNode)):

            # component
            parent = item.parent()
            struc = parent.parent()
            component_loc = parent.row()
            if component_loc == 0 and row in [0, 3, 4]:
                return False
            if component_loc == len(struc.structure) - 1 and row in [0, 4]:
                return False

        return True


def find_data_object(index):
    if not index.isValid():
        return

    item = index.internalPointer()
    hierarchy = item.hierarchy()
    data_object_node = [i for i in hierarchy if
                        isinstance(i, DataObjectNode)]
    return data_object_node[0]


###############################################################################
class LipidLeafletNode(ComponentNode):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super(LipidLeafletNode, self).__init__(data, model, parent)

        prop_node = PropertyNode('reverse_monolayer', model, parent=self)
        self.appendChild(prop_node)

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 0:
            flags |= QtCore.Qt.ItemIsEditable

        return flags

    def data(self, column, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None
        return self._data.name

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return False

        if column:
            return False

        self.component.name = value
        self._model.dataChanged.emit(self.index, self.index)
        return True
