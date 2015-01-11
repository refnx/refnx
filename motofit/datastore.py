from __future__ import division
import refnx.dataset.reflectdataset as reflectdataset
from refnx.analysis.curvefitter import CurveFitter
from lmfit import Parameters
from copy import deepcopy
import os.path
import pickle
from collections import OrderedDict
from graphproperties import GraphProperties

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

def zipper(dir, zip):
    root_len = len(os.path.abspath(dir))
    for root, dirs, files in os.walk(dir):
        archive_root = os.path.abspath(root)[root_len:]
        for f in files:
            fullpath = os.path.join(root, f)
            archive_name = os.path.join(archive_root, f)
            zip.write(fullpath, archive_name)


class DataStore(object):
    '''
        A container for storing datasets
    '''

    def __init__(self):
        super(DataStore, self).__init__()
        self.datasets = OrderedDict()

    def __getitem__(self, key):
        if key in self.datasets:
            return self.datasets[key]
        return None

    def __iter__(self):
        for key in self.datasets:
            yield self.datasets[key]

    def __len__(self):
        return len(self.datasets)

    @property
    def names(self):
        return list(self.datasets.keys())

    def add(self, dataset):
        dataset.graph_properties = GraphProperties()
        self.datasets[dataset.name] = dataset

    def load(self, filename):
        if os.path.basename(filename) in self.names:
            self.datasets[os.path.basename(filename)].refresh()
            return None

        dataset = reflectdataset.ReflectDataset()
        with open(filename, 'Ur') as f:
            dataset.load(f)
            self.add(dataset)

        return dataset

    def remove_dataset(self, name):
        self.datasets.pop(name)

    def refresh(self):
        for dataset in self.datasets:
            self.datasets[dataset].refresh()


class ParametersStore(object):
    '''
        a container for storing lmfit.Parameters
    '''

    def __init__(self):
        super(ParametersStore, self).__init__()
        self.parameters = OrderedDict()
        self.displayOtherThanReflect = False

    def __getitem__(self, key):
        return self.parameters[key]

    def __setitem__(self, key, value):
        if isinstance(value, Parameters):
            self.parameters[key] = value

    def __iter__(self):
        for key in self.parameters:
            yield self.parameters[key]

    def __len__(self):
        return len(self.parameters)

    @property
    def names(self):
        return list(self.parameters.keys())

    def save(self, name, fname):
        params = self[name]
        with open(fname, 'wb') as f:
            pickle.dump(params, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            p = pickle.load(f)

        if isinstance(p, Parameters):
            name = os.path.basename(fname)
            name = os.path.splitext(name)[0]
            print(name)
            self.add(p, name)

    def add(self, parameters, name):
        self.parameters[name] = parameters

    def snapshot(self, name, snapshot_name):
        parameters = self.parameters[name]
        self.parameters[snapshot_name] = deepcopy(parameters)


class MinimizersStore(object):
    '''
        a container for storing lmfit.Minimizers
    '''

    def __init__(self):
        super(MinimizersStore, self).__init__()
        self.minimizers = OrderedDict()

    def __getitem__(self, key):
        return self.minimizers[key]

    def __setitem__(self, key, value):
        if not isinstance(value, CurveFitter):
            raise ValueError
        else:
            self.minimizers[key] = value

    def __iter__(self):
        for key in self.minimizers:
            yield self.minimizers[key]

    def __len__(self):
        return len(self.minimizers)

    @property
    def names(self):
        return list(self.minimizers.keys())
