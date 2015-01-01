from __future__ import division
import numpy as np
import dataobject
import refnx.analysis.model as model
from copy import deepcopy, copy
import matplotlib.artist as artist
import os.path
import os
import string
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
        A container for storing dataobject.DataObjects
    '''

    def __init__(self):
        super(DataStore, self).__init__()
        self.dataObjects = {}
        self.numDataObjects = 0
        self.names = []

    def __getitem__(self, key):
        if key in self.dataObjects:
            return self.dataObjects[key]
        return None

    def __iter__(self):
        for key in self.dataObjects:
            yield self.dataObjects[key]

    def add(self, dataObject):
        self.dataObjects[dataObject.name] = dataObject
        self.names.append(dataObject.name)
        self.numDataObjects += 1

    def load(self, filename):
        if os.path.basename(filename) in self.names:
            self.dataObjects[os.path.basename(filename)].refresh()
            return None

        TdataObject = dataobject.DataObject()
        with open(filename, 'Ur') as f:
            TdataObject.load(f)

        self.add(TdataObject)

        return TdataObject

    def save_DataStore(self, folderName):
        for key in self.dataObjects.keys():
            dataObject = self.dataObjects[key]
            try:
                filename = os.path.join(folderName, dataObject.name)
            except AttributeError:
                print folderName, key

            with open(filename, 'w') as f:
                pass

            with open(filename, 'r+') as f:
                dataObject.save(f)

    def load_DataStore(self, files, clear=False):
        if clear:
            self.dataObjects.clear()
            self.names = []
            self.numDataObjects = 0

        for file in files:
            try:
                self.loadData(file)
            except IOError:
                continue

    def remove_DataObject(self, name):
        del(self.dataObjects[name])
        del(self.names[self.names.index(name)])
        self.numDataObjects -= 1

    def refresh(self):
        for dataObject in self:
            dataObject.refresh()


class ModelStore(object):

    '''
        a container for storing refnx.analysis.model.Models
    '''

    def __init__(self):
        super(ModelStore, self).__init__()
        self.models = {}
        self.names = []
        self.displayOtherThanReflect = False

    def __getitem__(self, key):
        return self.models[key]

    def __iter__(self):
        for key in self.models:
            yield self.models[key]

    def add(self, model, modelName):
        self.models[modelName] = model
        if modelName not in self.names:
            self.names.append(modelName)

    def save_ModelStore(self, folderName):
        for modelname in self.names:
            model = self.models[modelname]
            filename = os.path.join(folderName, modelname)
            with open(filename, 'wb+') as f:
                model.save(f)

    def load_ModelStore(self, files, clear=False):
        if clear:
            self.models.clear()
            self.names = []

        for file in files:
            try:
                with open(file, 'Ur') as f:
                    model = model.Model(None)
                    model.load(f)
                    self.add(model, os.path.basename(file))
            except IOError:
                # may be a directory
                continue

    def snapshot(self, name, snapshotname):
        model = self.models[name]
        snapshot = model.Model(parameters=model.parameters,
                               fitted_parameters=model.fitted_parameters,
                               limits=model.limits,
                               useerrors=model.useerrors,
                               costfunction=model.costfunction,
                               usedq=model.usedq)
        self.add(snapshot, snapshotname)
