from __future__ import division
import pyplatypus.dataset.reflectdataset as reflectdataset
import numpy as np
import pyplatypus.analysis.reflect as reflect
from copy import deepcopy, copy
import matplotlib.artist as artist
import os.path, os
from PySide import QtGui, QtCore
import string
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
    
    
class DataObject(reflectdataset.ReflectDataset):        
    __requiredgraphproperties = {'lw':float, 'label':str, 'linestyle':str,
       'fillstyle':str, 'marker':str, 'markersize':float, 'markeredgecolor':str,
      'markerfacecolor':str, 'zorder':int, 'color':str}
                                    
    def __init__(self, dataTuple = None, name = '_theoretical_', fname = None):
        super(DataObject, self).__init__(dataTuple = dataTuple)
        
        self.name = '_theoretical_'
        
        if fname is not None:
            with open(fname, 'Ur') as f:
                self.load(f)
        
        self.fit = None
        self.residuals = None
        
        self.chi2 = -1
        self.sld_profile = None
        
        self.line2D = None
        self.line2Dfit = None
        self.line2Dresiduals = None        
        self.line2Dsld_profile = None
        
        self.graph_properties = {'line2Dsld_profile_properties':{},
                                'line2Dresiduals_properties':{},
                                'line2Dfit_properties':{},
                                'line2D_properties':{},
                                 'visible':True}
        
    def __getstate__(self):
        self._save_graph_properties()
        d = copy(self.__dict__)
        d['line2Dfit'] = None
        d['line2D'] = None
        d['line2Dsld_profile'] = None
        d['line2Dresiduals'] = None
#        del(d['fit'])
        return d
        
    def save(self, f):
        #this will save it as XML
        super(DataObject, self).save(f)
            
        #have to add in extra bits about the fit.
        try:
            f.seek(0)
            tree = ET.ElementTree()    
            tree.parse(f)
        except Exception:
            #couldn't parse, may not be xml file.
            return
                    
        try:  
            self._save_graph_properties()
            rdata = tree.find('.//R')
            rdata.attrib = dict(list(rdata.attrib.items()) + list(self.graph_properties['line2D_properties'].items()))
         
            refdata = tree.find('.//REFdata')
            if self.fit is not None:
                fit = ET.SubElement(refdata, 'fit')
                fit.attrib = self.graph_properties['line2Dfit_properties']
                fit.text = string.translate(repr(self.fit.tolist()), None, ',[]')
                
            if self.residuals is not None:
                residuals = ET.SubElement(refdata, 'residuals')
                residuals.attrib = self.graph_properties['line2Dresiduals_properties']
                residuals.text = string.translate(repr(self.residuals.tolist()), None, ',[]')
            
            if self.sld_profile is not None:
                sld_profile = ET.SubElement(refdata, 'sld')
                sld_profile.attrib = self.graph_properties['line2Dsld_profile_properties']
                sld_profilez = ET.SubElement(sld_profile, 'z')
                sld_profilerho = ET.SubElement(sld_profile, 'rho')
                sld_profilez.text = string.translate(repr(self.sld_profile[0].tolist()), None, ',[]')
                sld_profilerho.text = string.translate(repr(self.sld_profile[1].tolist()), None, ',[]')                
            f.seek(0)
            tree.write(f)
        except Exception as inst:
            print type(inst)

    def load(self, f):
        #this will load as XML
        super(DataObject, self).load(f)
          
        #have to add in extra bits, if it was saved as XML, through this program
        try:
            f.seek(0)
            tree = ET.ElementTree()    
            tree.parse(f)
        except Exception:
            #couldn't parse, is not an xml file.
            return
                    
#         try:  
        rdata = tree.find('.//R')
        for key in rdata.attrib:
            if key in self.__requiredgraphproperties:
                self.graph_properties['line2D_properties'][key] = self.__requiredgraphproperties[key](rdata.attrib[key])

        fit = tree.find('.//fit')   
        if fit is not None:
            for key in fit.attrib:
                if key in self.__requiredgraphproperties:
                    self.graph_properties['line2Dfit_properties'][key] = self.__requiredgraphproperties[key](fit.attrib[key])
            self.fit = np.array([float(val) for val in fit.text.split()])

        residuals = tree.find('.//residuals')  
        if residuals is not None:
            for key in residuals.attrib:
                if key in self.__requiredgraphproperties:
                    self.graph_properties['line2Dresiduals_properties'][key] = self.__requiredgraphproperties[key](residuals.attrib[key])
            self.residuals = np.array([float(val) for val in residuals.text.split()])

        sld_profile = tree.find('.//sld')            
        if sld_profile:
            for key in sld_profile.attrib:
                if key in self.__requiredgraphproperties:
                    self.graph_properties['line2Dsld_profile_properties'][key] = self.__requiredgraphproperties[key](sld_profile.attrib[key])
            zed = tree.find('.//z') 
            rho = tree.find('.//rho')
            self.sld_profile = []
            self.sld_profile.append(np.array([float(val) for val in zed.text.split()]))
            self.sld_profile.append(np.array([float(val) for val in rho.text.split()]))
                
#         except Exception as inst:
#             print type(inst)

    def _save_graph_properties(self):
        if self.line2D:
            for key in self.__requiredgraphproperties:
                self.graph_properties['line2D_properties'][key] = str(artist.getp(self.line2D, key))

        if self.line2Dfit:
            for key in self.__requiredgraphproperties:
                self.graph_properties['line2Dfit_properties'][key] = str(artist.getp(self.line2Dfit, key))

        if self.line2Dresiduals:
            for key in self.__requiredgraphproperties:
                self.graph_properties['line2Dresiduals_properties'][key] = str(artist.getp(self.line2Dresiduals, key))
                            
        if self.line2Dsld_profile:
            for key in self.__requiredgraphproperties:
                self.graph_properties['line2Dsld_profile_properties'][key] = str(artist.getp(self.line2Dsld_profile, key))
        
    def do_a_fit(self, model, reflectPlugin = None):
        '''
            TODO this should be somewhat refactored into GUI code
        '''
        
        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        
        try:
            if model.usedq:
                callerInfo['dqvals'] = self.W_qSD
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass
        
        self.progressdialog = QtGui.QProgressDialog("Fit progress", "Abort", 0, 100)   
        self.progressdialog.setWindowModality(QtCore.Qt.WindowModal)
        
        if reflectPlugin is not None:
            RFO = reflectPlugin(**callerInfo)
        else:
            RFO = reflect.ReflectivityFitObject(**callerInfo)
            
        RFO.progress = self.progress
        model.parameters, model.uncertainties, self.chi2 = RFO.fit()
        
        self.progressdialog.setValue(100)
        
        self.fit = RFO.model()
        self.residuals = np.log10(self.fit/self.W_ref)
        self.sld_profile = RFO.sld_profile()
    
    def progress(self, iterations, convergence, chi2, *args):
        self.progressdialog.setValue(int(convergence * 100))
        if self.progressdialog.wasCanceled():
            return False
        else:  
            return True
                  
    def evaluate_chi2(self, model, store = False, reflectPlugin = None):
        
        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        
        try:
            if model.usedq:
                callerInfo['dqvals'] = self.W_qSD
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass
                          
        if reflectPlugin is not None:
            RFO = reflectPlugin(**callerInfo)
        else:
            RFO = reflect.ReflectivityFitObject(**callerInfo)
        
        energy = RFO.energy() / self.numpoints
        if store:
            self.chi2 = energy
                
        return energy

    def evaluate_model(self, model, store = False, reflectPlugin = None):   
            
        callerInfo = deepcopy(model.__dict__)
        callerInfo['xdata'] = self.W_q
        callerInfo['ydata'] = self.W_ref
        callerInfo['edata'] = self.W_refSD
        
        try:
            if model.usedq:
                callerInfo['dqvals'] = self.W_qSD  
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass
                
        if reflectPlugin is not None:
            RFO = reflectPlugin(**callerInfo)
        else:
            RFO = reflect.ReflectivityFitObject(**callerInfo)
                
        fit = RFO.model()
        sld_profile = RFO.sld_profile()
        if store:
            self.fit = fit
            self.residuals = fit - self.W_ref
            self.sld_profile = sld_profile

        return fit, fit - self.W_ref, sld_profile
