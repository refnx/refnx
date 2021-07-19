from copy import deepcopy
import os.path
import pickle
from operator import itemgetter

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from refnx._lib import flatten, unique
from refnx.analysis import Parameter, Parameters, possibly_create_parameter
from refnx.dataset import ReflectDataset
from refnx.reflect._app.dataobject import DataObject
from refnx.reflect._app.datastore import DataStore
from refnx.reflect import (
    Slab,
    LipidLeaflet,
    SLD,
    ReflectModel,
    MixedReflectModel,
    Spline,
    Stack,
)


def component_class(component):
    """
    Give the node type for a given component
    """
    if isinstance(component, Slab):
        return SlabNode
    elif isinstance(component, LipidLeaflet):
        return LipidLeafletNode
    elif isinstance(component, Spline):
        return SplineNode
    elif isinstance(component, Stack):
        return StackNode


class Node:
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
        # TreeModel emits the dataChanged Signal, not the node.
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
        super().__init__(data, model, parent)

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
        if role == QtCore.Qt.CheckStateRole:
            if column == 1:
                p = self.parameter
                if p.vary:
                    return QtCore.Qt.Checked
                else:
                    return QtCore.Qt.Unchecked
            else:
                return None
        elif role == QtCore.Qt.ToolTipRole:
            if column in [1, 2]:
                p = self.parameter
                return getattr(p, "units", None)
            elif column == 3:
                return "Lower limit for the parameter"
            elif column == 4:
                return "Upper limit for the parameter"

        if role == QtCore.Qt.DisplayRole:
            p = self.parameter
            d = [
                p.name,
                float(p.value),
                p.stderr,
                p.bounds.lb,
                p.bounds.ub,
                repr(p.constraint),
            ]
            return d[column]

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        # we want to use a checkbox to say if a parameter is varying
        if role == QtCore.Qt.CheckStateRole and column == 1:
            try:
                p = self.parameter
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
                p = self.parameter
                if column == 1:
                    p.value = float(voutput[1])
                elif column == 3:
                    p.bounds.lb = float(voutput[1])
                elif column == 4:
                    p.bounds.ub = float(voutput[1])
            else:
                return False

        return True


class ParametersNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super().__init__(data, model, parent)
        for p in data:
            if isinstance(p, Parameters):
                n = ParametersNode(p, model, self)
            if isinstance(p, Parameter):
                n = ParNode(p, model, self)
            self.appendChild(n)

    @property
    def parameters(self):
        return self._data

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled

        return flags

    def data(self, column, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if column == 0:
                name = self._data.name
                if not name:
                    name = "Parameters"
                return name

        return None


class PropertyNode(Node):
    # an object that displays/edits some attribute of its parent node
    # it is not a ParNode.
    def __init__(
        self, data, model, parent=QtCore.QModelIndex(), validators=()
    ):
        super().__init__(data, model, parent)
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
        if (
            role == QtCore.Qt.CheckStateRole
            and column == 1
            and self.attribute_type is bool
        ):
            d = getattr(self._parent._data, self._data)

            if d:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        if role == QtCore.Qt.DisplayRole and column == 1:
            return getattr(self._parent._data, self._data)
        if role == QtCore.Qt.DisplayRole and column == 0:
            return self._data

    def columnCount(self):
        return 2

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        if (
            role == QtCore.Qt.CheckStateRole
            and column == 1
            and self.attribute_type is bool
        ):

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
                    setattr(
                        self._parent._data,
                        self._data,
                        self.attribute_type(voutput[1]),
                    )
                    return True

        return False


class ComponentNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex(), flat=True):
        """
        Parameters
        ----------
        flat : bool
            If `flat is True`, then this superclass will flatten out all
            the parameters in a model and append them as child items.
        """
        super().__init__(data, model, parent)

        if flat:
            for par in flatten(data.parameters):
                pn = ParNode(par, model, parent=self)
                self.appendChild(pn)
        else:
            for p in data.parameters:
                if isinstance(p, Parameters):
                    n = ParametersNode(p, model, self)
                if isinstance(p, Parameter):
                    n = ParNode(p, model, self)
                self.appendChild(n)

    @property
    def component(self):
        return self._data

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 0:
            flags |= QtCore.Qt.ItemIsEditable

        # say that you want the Components to be draggable
        flags |= QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled

        return flags

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None
        if role == QtCore.Qt.DisplayRole:
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
        super().__init__(data, model, parent)
        for component in data:
            self.appendChild(
                component_class(component)(component, model, self)
            )

    @property
    def structure(self):
        return self._data

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None
        # structures name
        return "structure"

    def remove_component(self, row):
        self._model.beginRemoveRows(self.index, row, row)

        # remove all dependent parameters (either in this dataset or elsewhere)
        # this has to be done before it's popped from the Structure.
        self._model.unlink_dependent_parameters(self.child(row))

        self.structure.pop(row)
        self.popChild(row)

        self._model.endRemoveRows()

    def move_component(self, src, dst):
        if src == dst or dst == src + 1:
            return

        self._model.beginMoveRows(self.index, src, src, self.index, dst)

        # swap in the underlying data
        strc = self._data
        children = self._children

        c = strc[src]
        cn = children[src]

        strc.insert(dst, c)
        children.insert(dst, cn)

        if src < dst:
            strc.pop(src)
            children.pop(src)
        else:
            strc.pop(src + 1)
            children.pop(src + 1)

        self._model.endMoveRows()

    def insert_component(self, row, component):
        n = component_class(component)(component, self._model, self)

        self._model.beginInsertRows(self.index, row, row)
        self.insertChild(row, n)
        self.structure.insert(row, component)
        self._model.endInsertRows()


# the child note where structures start appearing in a ReflectModelNode
STRUCT_OFFSET = 4


class ReflectModelNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super().__init__(data, model, parent=parent)

        # pointwise/constant dq/q choice is kept in the ReflectModel
        if data.dq_type == "constant":
            self.constantdq_q = True
        elif data.dq_type == "pointwise":
            self.constantdq_q = False

        # deal with scale factors
        if isinstance(data, ReflectModel):
            n = ParNode(data.scale, model, self)
        elif isinstance(data, MixedReflectModel):
            n = ParametersNode(data.scales, model, self)
        self.appendChild(n)

        n = ParNode(data.bkg, model, self)
        self.appendChild(n)
        n = ParNode(data.dq, model, self)
        self.appendChild(n)
        n = ParNode(data.q_offset, model, self)
        self.appendChild(n)

        if isinstance(data, ReflectModel):
            n = StructureNode(data.structure, model, self)
            self.appendChild(n)
        elif isinstance(data, MixedReflectModel):
            for structure in data.structures:
                n = StructureNode(structure, model, self)
                self.appendChild(n)

    @property
    def structures(self):
        return [
            self.child(i)._data
            for i in range(STRUCT_OFFSET, self.childCount())
        ]

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole and column == 1:
            if self.constantdq_q:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        if role == QtCore.Qt.DisplayRole:
            if column == 0:
                # structures name
                return "model"
            elif column == 1:
                return "dq/q const. smearing?"

        return None

    def remove_structure(self, row):
        # don't remove the last structure
        if row < STRUCT_OFFSET or len(self.structures) < 2:
            return

        # you have to be a mixedreflectmodel because there's more than one
        # structure
        data_object_node = find_data_object(self.index)
        data_object = data_object_node.data_object
        orig_model = data_object.model
        structures = orig_model.structures
        scales = orig_model.scales

        # going down to a single reflectmodel
        if len(structures) == 2:
            self._model.beginRemoveRows(self.index, row, row)
            # remove all dependent parameters (either in this dataset or
            # elsewhere)
            # do this before the Structure is popped.
            self._model.unlink_dependent_parameters(self.child(row))

            structures.pop(row - STRUCT_OFFSET)
            scales.pop(row - STRUCT_OFFSET)
            sf = possibly_create_parameter(scales[0], name="scale")
            new_model = ReflectModel(
                structures[0], scale=sf, bkg=orig_model.bkg, dq=orig_model.dq
            )
            data_object.model = new_model
            self.popChild(row)
            self._model.endRemoveRows()
            n = ParNode(sf, self._model, self)
            self._model.beginInsertRows(self.index, 0, 0)
            self.insertChild(0, n)
            self._model.endInsertRows()
            self._model.beginRemoveRows(self.index, 1, 1)
            self.popChild(1)
            self._model.endRemoveRows()
            # data_object_node.set_reflect_model(new_model)
            return

        # you're not down to a single structure, so there must have been more
        # than 2 structures
        self._model.beginRemoveRows(self.index, row, row)

        # remove all dependent parameters (either in this dataset or elsewhere)
        # do this before the Structure is popped.
        self._model.unlink_dependent_parameters(self.child(row))

        # pop the structure and scale factor nodes
        self.popChild(row)
        structures.pop(row - STRUCT_OFFSET)
        scales.pop(row - STRUCT_OFFSET)
        self._model.endRemoveRows()

        self._model.beginRemoveRows(
            self.child(0).index, row - STRUCT_OFFSET, row - STRUCT_OFFSET
        )
        self.child(0).popChild(row - STRUCT_OFFSET)
        self._model.endRemoveRows()

    def insert_structure(self, row, structure):
        n = StructureNode(structure, self._model, self)

        data_object_node = find_data_object(self.index)
        data_object = data_object_node.data_object
        orig_model = data_object.model

        if len(self.structures) == 1:
            self._model.beginInsertRows(self.index, row, row)
            new_structures = [self.structures[0], structure]
            new_model = MixedReflectModel(
                new_structures, bkg=orig_model.bkg, dq=orig_model.dq
            )
            data_object.model = new_model
            data_object_node.set_reflect_model(
                new_model, constdq_q=self.constantdq_q
            )
            return

        # already a mixed model
        # we can't insert at a lower place than the 4rd row
        row = max(row, 4)
        self._model.beginInsertRows(self.index, row, row)

        # insert the structure
        orig_model.structures.insert(row - STRUCT_OFFSET, structure)
        v = 1 / len(orig_model.structures)
        sf = possibly_create_parameter(v, name="scale")
        orig_model.scales.insert(row - STRUCT_OFFSET, sf)

        self.insertChild(row, n)
        self._model.endInsertRows()

        # insert a scale factor
        self._model.beginInsertRows(
            self.child(0).index, row - STRUCT_OFFSET, row - STRUCT_OFFSET
        )
        # add a scale factor
        n = ParNode(sf, self._model, self.child(0))
        self.child(0).insertChild(row - STRUCT_OFFSET, n)
        self._model.endInsertRows()

    def setData(self, column, value, role=QtCore.Qt.EditRole):
        # currently this only deals with constant dq/q
        if role == QtCore.Qt.CheckStateRole and column == 1:
            self.constantdq_q = value == QtCore.Qt.Checked
            data_object_node = find_data_object(self.index)
            data_object = data_object_node.data_object
            data_object.constantdq_q = self.constantdq_q

            # need to not let the dq parameter vary if there's resolution
            # information in the dataset
            if not self.constantdq_q:
                data_object_node.data_object.model.dq_type = "pointwise"
                data_object_node.data_object.model.dq.vary = False
            else:
                data_object_node.data_object.model.dq_type = "constant"

            lindex = self._model.index(2, 0, self.index)
            rindex = self._model.index(2, 4, self.index)
            self._model.dataChanged.emit(lindex, rindex)

            return True

        return True

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 1:
            flags |= QtCore.Qt.ItemIsUserCheckable

        return flags


class DatasetNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super().__init__(data, model, parent)

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
        super().__init__(data_object, model, parent=parent)

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

    def set_reflect_model(self, model, constdq_q=True):
        if model is not None:
            n = ReflectModelNode(model, self._model, self)
            n.constantdq_q = constdq_q

            # no ReflectModel, append the ReflectModel as a child
            if len(self._children) == 1:
                self._model.beginInsertRows(self.index, 1, 1)
                self.appendChild(n)
                self.data_object.model = model
                self.data_object.constantdq_q = constdq_q
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
                self.data_object.constantdq_q = constdq_q
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
        if role == QtCore.Qt.CheckStateRole and column == 1:
            self.visible = value == QtCore.Qt.Checked
            return True

        return True

    def data(self, column, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.CheckStateRole and column == 1:
            if self.visible:
                return QtCore.Qt.Checked
            else:
                return QtCore.Qt.Unchecked

        if role == QtCore.Qt.DisplayRole:
            if column == 0:
                return self._data.name
            elif column == 1:
                return "display"
            elif column == 2:
                return "points: %d" % len(self._data.dataset)
            elif column == 3:
                return "chi2: %g" % self.chi2
        elif role == QtCore.Qt.ToolTipRole:
            if column == 1:
                return "Show or hide the dataset from the graphs"
            elif column == 3:
                return (
                    "((y<sub>i,data</sub>-y<sub>i,model</sub>)"
                    "/y<sub>i,err</sub>)<sup>2</sup> summed over all "
                    "datapoints"
                )

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 1:
            flags |= QtCore.Qt.ItemIsUserCheckable

        # want to drop dataobject onto currently_fitting
        flags |= QtCore.Qt.ItemIsDragEnabled
        return flags


class ContainerNode(Node):
    # set up with a datastore
    def __init__(self, data, model, parent=None):
        super().__init__(data, model, parent)
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
                # refreshed.
                position = self.datastore.names.index(data_object.name)
                index = self.index
                idx = self._model.index(position, 1, index)
                idx1 = self._model.index(position, 3, index)

                self._model.dataChanged.emit(idx, idx1)

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

        # remove all dependent parameters (either in this dataset or elsewhere)
        # do this before the data_object is popped.
        self._model.unlink_dependent_parameters(
            self._model.data_object_node(name)
        )

        # remove from the backing datastore and pop from the nodelist
        self.datastore.remove_dataset(name)
        self.popChild(row)

        self._model.endRemoveRows()

    def refresh(self):
        for data_object_node in self._children:
            data_object_node.refresh()


class TreeModel(QtCore.QAbstractItemModel):
    """
    Parameters
    ----------
    data: refnx.reflect._app.datastore.DataStore
    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
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
        headers = ["Name", "value", "sigma", "lb", "ub", "constraint"]
        if (
            orientation == QtCore.Qt.Horizontal
            and role == QtCore.Qt.DisplayRole
        ):
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

    def supportedDropActions(self):
        return QtCore.Qt.MoveAction

    def mimeTypes(self):
        return ["application/vnd.treeviewdragdrop.list"]

    def mimeData(self, indexes):
        index_info = []
        mimedata = QtCore.QMimeData()
        for index in indexes:
            if index.isValid():
                node = index.internalPointer()
                info = {"name": None, "indices": None}
                if isinstance(node, Node):
                    info["indices"] = node.row_indices()
                if isinstance(node, DataObjectNode):
                    info["name"] = node.data_object.name
                index_info.append(info)

        s = pickle.dumps(index_info)
        mimedata.setData("application/vnd.treeviewdragdrop.list", s)
        return mimedata

    def dropMimeData(self, data, action, row, column, parent):
        if action == QtCore.Qt.IgnoreAction:
            return True

        if not data.hasFormat("application/vnd.treeviewdragdrop.list"):
            return False

        # what the destination was.
        host_node = parent.internalPointer()

        if not (
            isinstance(host_node, ComponentNode)
            or isinstance(host_node, StackNode)
        ):
            return False

        # host_structure could be a Structure OR a Stack
        host_structure_node = host_node.parent()
        host_structure = host_structure_node._data
        ba = data.data("application/vnd.treeviewdragdrop.list")
        index_info = pickle.loads(ba)
        dragged_nodes = list(
            unique(
                self.node_from_row_indices(i["indices"]) for i in index_info
            )
        )

        # order the dragged nodes
        src_rows = [dn.row() for dn in dragged_nodes]
        d = sorted(zip(src_rows, dragged_nodes), key=itemgetter(0))
        dragged_nodes = [i[1] for i in d]

        # add to the destination in reverse
        dragged_nodes.reverse()

        for dragged_node in dragged_nodes:
            # can't drag a fronting/backing medium in a Structure
            src_structure_node = dragged_node.parent()
            src_structure = src_structure_node._data

            if isinstance(
                src_structure_node, StructureNode
            ) and dragged_node.row() in [
                0,
                len(src_structure) - 1,
            ]:
                continue

            # if (isinstance(dragged_node, SplineNode) and
            #         isinstance(host_structure_node, StackNode)):
            #     continue

            # figure out what the destination is.
            dst_row = host_node.row() + 1
            if dst_row == len(host_structure) and isinstance(
                host_structure_node, StructureNode
            ):
                return False

            if src_structure_node is host_structure_node:
                # moving within the same structure
                src_row = dragged_node.row()
                host_structure_node.move_component(src_row, dst_row)
            else:
                # move the Component between Structures.
                c = deepcopy(dragged_node._data)
                src_row = dragged_node.row()
                src_structure_node.remove_component(src_row)
                host_structure_node.insert_component(dst_row, c)

        return True

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
        original = self.datastore["theoretical"]
        dataset = ReflectDataset()
        dataset.data = (
            original.dataset.x,
            original.model.model(original.dataset.x, x_err=dataset.x_err),
        )
        dataset.name = snapshot_name

        new_model = deepcopy(original.model)
        new_model.name = snapshot_name

        # if the snapshot already exists then overwrite it.
        if snapshot_name in self.datastore.names:
            row = self.data_object_row(snapshot_name)
            self._rootnode.child(row).set_dataset(dataset)
            self._rootnode.child(row).set_reflect_model(new_model)
            data_object = self.data_object_node(snapshot_name).data_object
        else:
            # otherwise you have to add it.
            data_object = DataObject(dataset)
            data_object.model = new_model
            self._rootnode.set_data_object(data_object)

        return data_object

    def refresh(self):
        self._rootnode.refresh()

    def unlink_dependent_parameters(self, node):
        # if a dataset, or component, is removed which contains a 'master'
        # parameter, this method iterates through all dataobjects and removes
        # all constraints for parameters (on all datasets) that were linked to
        # those master parameters.

        # all the data_object nodes
        dons = self._rootnode._children

        if isinstance(node, DataObjectNode):
            # removing a dataset
            o_params = node.data_object.model.parameters.flattened()
        elif isinstance(node, ComponentNode):
            o_params = node.component.parameters.flattened()
        elif isinstance(node, StructureNode):
            o_params = node.structure.parameters.flattened()
        elif isinstance(node, StackNode):
            o_params = node.stack.parameters.flattened()

        # iterate through dataobjects and see if any parameters depend on
        # any of these model parameters
        for don in dons:
            do = don.data_object
            model_pars = do.model.parameters.flattened()
            for mp in model_pars:
                if set(mp.dependencies()).intersection(o_params):
                    mp.constraint = None
            # TODO: investigate whether a dataChanged emit is necessary.


class TreeFilter(QtCore.QSortFilterProxyModel):
    def __init__(self, tree_model, parent=None, only_fitted=False):
        super(TreeFilter, self).__init__(parent)
        self.tree_model = tree_model
        # only_fitted means that only the entries that are varying will be
        # displayed.
        self.only_fitted = only_fitted
        self._fitted_datasets = []

    def filterAcceptsRow(self, row, index):
        idx = self.tree_model.index(row, 0, index)

        if not idx.isValid():
            return False

        item = idx.internalPointer()
        if isinstance(item, DatasetNode):
            return False

        # filter out resolution parameter if the dataset has x_err
        # and constant dq/q wasn't requested.
        if isinstance(item.parent(), ReflectModelNode) and isinstance(
            item, ParNode
        ):
            data_object_node = find_data_object(item.index)
            dataset = data_object_node.data_object.dataset
            constantdq_q = data_object_node.data_object.constantdq_q
            # hard-coded the row for dq/q
            if (
                item.row() == 2
                and not constantdq_q
                and dataset.x_err is not None
            ):
                return False

        # filter out parameters for the fronting/backing media
        if isinstance(item.parent(), SlabNode) and isinstance(item, ParNode):

            # component
            parent = item.parent()
            struc = parent.parent()
            if not isinstance(struc, StructureNode):
                return True

            component_loc = parent.row()
            if component_loc == 0 and row in [0, 2, 3, 4]:
                return False
            if component_loc == len(struc.structure) - 1 and row in [0, 4]:
                return False

        # only_fitted means that only varying parameters should be displayed
        if self.only_fitted:
            # you have to be in the currently fitting list to be shown
            dataset = find_data_object(idx).data_object
            if dataset.name not in self._fitted_datasets:
                return False
            if isinstance(item, ParNode) and item._data.vary is False:
                return False

        return True


def find_data_object(index):
    if not index.isValid():
        return

    item = index.internalPointer()
    hierarchy = item.hierarchy()
    data_object_node = [i for i in hierarchy if isinstance(i, DataObjectNode)]
    return data_object_node[0]


###############################################################################
# Classes for the Node structure of different Components

###############################################################################
class SlabNode(ComponentNode):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super().__init__(data, model, parent)


###############################################################################
class LipidLeafletNode(ComponentNode):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super().__init__(data, model, parent)

        prop_node = PropertyNode("reverse_monolayer", model, parent=self)
        self.appendChild(prop_node)


###############################################################################
class SplineNode(ComponentNode):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super().__init__(data, model, parent, flat=False)
        prop_node = PropertyNode("zgrad", model, parent=self)
        self.appendChild(prop_node)

        validator = QtGui.QDoubleValidator()
        validator.setBottom(0)
        prop_node = PropertyNode(
            "microslab_max_thickness",
            model,
            parent=self,
            validators=(validator,),
        )
        prop_node.attribute_type = float
        self.appendChild(prop_node)


###############################################################################
class StackNode(Node):
    def __init__(self, data, model, parent=QtCore.QModelIndex()):
        super().__init__(data, model, parent)

        # append the number of repeats
        pn = ParNode(data.repeats, model, parent=self)
        self.appendChild(pn)

        for component in data:
            self.appendChild(
                component_class(component)(component, model, self)
            )

    @property
    def stack(self):
        return self._data

    def flags(self, column):
        flags = QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
        if column == 0:
            flags |= QtCore.Qt.ItemIsEditable

        # say that you want the Components to be draggable
        flags |= QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled

        return flags

    def data(self, column, role=QtCore.Qt.EditRole):
        if role == QtCore.Qt.CheckStateRole:
            return None

        if column > 0:
            return None

        if role == QtCore.Qt.DisplayRole:
            return self._data.name

    def remove_component(self, row):
        strc = self._data
        if len(strc) == 1:
            # can't remove the last component in a stack.
            return

        self._model.beginRemoveRows(self.index, row, row)
        # the first row in the GUI model is the repeats parnode, the second is
        # the first component
        self.stack.pop(row - 1)
        self.popChild(row)
        self._model.endRemoveRows()

    def move_component(self, src, dst):
        if src == dst or dst == src + 1:
            return

        self._model.beginMoveRows(self.index, src, src, self.index, dst)

        # swap in the underlying data
        strc = self._data
        children = self._children

        c = strc[src - 1]
        cn = children[src]

        strc.insert(dst - 1, c)
        children.insert(dst, cn)

        if src < dst:
            strc.pop(src - 1)
            children.pop(src)
        else:
            strc.pop(src)
            children.pop(src + 1)

        self._model.endMoveRows()

    def insert_component(self, row, component):
        n = component_class(component)(component, self._model, self)

        self._model.beginInsertRows(self.index, row, row)
        self.insertChild(row, n)
        self.stack.insert(row, component)
        self._model.endInsertRows()
