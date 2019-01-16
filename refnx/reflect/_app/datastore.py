import os.path
from collections import OrderedDict

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import numpy as np
from refnx.reflect import SLD, Structure, ReflectModel
from refnx.dataset import ReflectDataset

from .dataobject import DataObject


def zipper(dir, zip):
    root_len = len(os.path.abspath(dir))
    for root, dirs, files in os.walk(dir):
        archive_root = os.path.abspath(root)[root_len:]
        for f in files:
            fullpath = os.path.join(root, f)
            archive_name = os.path.join(archive_root, f)
            zip.write(fullpath, archive_name)


class DataStore(object):
    """
    A container for storing datasets
    """

    def __init__(self):
        super(DataStore, self).__init__()
        self.data_objects = OrderedDict()

        # create the default theoretical dataset
        q = np.linspace(0.005, 0.5, 1000)
        r = np.empty_like(q)
        dataset = ReflectDataset()
        dataset.data = (q, r)
        dataset.name = 'theoretical'
        air = SLD(0, name='fronting')
        sio2 = SLD(3.47, name='1')
        si = SLD(2.07, name='backing')
        structure = air(0, 0) | sio2(15, 3.) | si(0, 3.)
        model = ReflectModel(structure, name='theoretical')
        self.add(dataset)
        self['theoretical'].model = model

    def __getitem__(self, key):
        if key in self.data_objects:
            return self.data_objects[key]
        return None

    def __iter__(self):
        for key in self.data_objects:
            yield self.data_objects[key]

    def __len__(self):
        return len(self.data_objects)

    @property
    def names(self):
        return list(self.data_objects.keys())

    def add(self, dataset):
        do = DataObject(dataset)

        # load was probably a failure
        if not len(do.dataset):
            return None

        # don't overwrite the data object if present, just refresh it
        if self[do.name] is not None:
            do.refresh()
        else:
            self.data_objects[do.name] = do

        return do

    def load(self, filename):
        dataset = ReflectDataset(filename)
        data_object = self.add(dataset)
        return data_object

    def remove_dataset(self, name):
        self.data_objects.pop(name)

    def refresh(self):
        for data_object in self:
            data_object.refresh()
