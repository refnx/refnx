import imp
import sys
import inspect
import hashlib
import numpy as np
import pyplatypus.analysis.reflect as reflect

def loadReflectivityModule(filepath):
    #this loads all modules
    hash = hashlib.md5(filepath)
    module = imp.load_source(filepath, filepath)
    
    rfos = []
    
    members = inspect.getmembers(module, inspect.isclass)
    for member in members:
        if issubclass(member[1], reflect.ReflectivityFitObject):
            rfos.append(member)
    
    if not len(rfos):
        del sys.modules[filepath]
        return None
        
    return (module, rfos)
    
class UserDefinedModel(QtCore.QAbstractTableModel):
    def __init__(self):
        pass