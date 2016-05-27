from __future__ import division
from PySide import QtGui, QtCore
import refnx.dataset.reflectdataset as reflectdataset
import refnx.analysis.reflect as reflect
import refnx.analysis.fitting as fitting
import refnx.util.ErrorProp as EP
import numpy as np
from copy import deepcopy, copy
import matplotlib.artist as artist
import os.path
import os
import string

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class DataObject(reflectdataset.ReflectDataset):

    '''
        defines a dataset
    '''

    __requiredgraphproperties = {'lw': float,
                                 'label': str,
                                 'linestyle': str,
                                 'fillstyle': str,
                                 'marker': str,
                                 'markersize': float,
                                 'markeredgecolor': str,
                                 'markerfacecolor': str,
                                 'zorder': int,
                                 'color': str,
                                 'visible': bool}

    def __init__(self, dataTuple=None,
                 name='theoretical', filename=None):
        super(DataObject, self).__init__(dataTuple=dataTuple)

        self.name = name

        if filename is not None:
            with open(filename, 'Ur') as f:
                self.load(f)
        else:
            self.filename = None

        self.fit = None
        self.residuals = None

        self.chi2 = -1
        self.sld_profile = None

        self.line2D = None
        self.line2Dfit = None
        self.line2Dresiduals = None
        self.line2Dsld_profile = None

        self.graph_properties = {'line2Dsld_profile_properties': {},
                                 'line2Dresiduals_properties': {},
                                 'line2Dfit_properties': {},
                                 'line2D_properties': {},
                                 'visible': True}

    def __getstate__(self):
        self._save_graph_properties()
        d = copy(self.__dict__)
        d['line2Dfit'] = None
        d['line2D'] = None
        d['line2Dsld_profile'] = None
        d['line2Dresiduals'] = None
#        del(d['fit'])
        return d

    def save_fit(self, filename):
        if self.fit is not None:
            with open(filename, 'wb+') as f:
                np.savetxt(f, np.column_stack((self.x, self.fit)))

    def save(self, f):
        # this will save it as XML
        super(DataObject, self).save(f)

        # have to add in extra bits about the fit.
        try:
            f.seek(0)
            tree = ET.ElementTree()
            tree.parse(f)
        except Exception:
            # couldn't parse, may not be xml file.
            return

        try:
            self._save_graph_properties()
            rdata = tree.find('.//R')
            rdata.attrib = dict(
                list(rdata.attrib.items()) + list(self.graph_properties['line2D_properties'].items()))

            refdata = tree.find('.//REFdata')
            if self.fit is not None:
                fit = ET.SubElement(refdata, 'fit')
                fit.attrib = self.graph_properties['line2Dfit_properties']
                fit.text = string.translate(
                    repr(self.fit.tolist()),
                    None,
                    ',[]')

            if self.residuals is not None:
                residuals = ET.SubElement(refdata, 'residuals')
                residuals.attrib = self.graph_properties[
                    'line2Dresiduals_properties']
                residuals.text = string.translate(
                    repr(self.residuals.tolist()),
                    None,
                    ',[]')

            if self.sld_profile is not None:
                sld_profile = ET.SubElement(refdata, 'sld')
                sld_profile.attrib = self.graph_properties[
                    'line2Dsld_profile_properties']
                sld_profilez = ET.SubElement(sld_profile, 'z')
                sld_profilerho = ET.SubElement(sld_profile, 'rho')
                sld_profilez.text = string.translate(
                    repr(self.sld_profile[0].tolist()),
                    None,
                    ',[]')
                sld_profilerho.text = string.translate(
                    repr(self.sld_profile[1].tolist()),
                    None,
                    ',[]')
            f.seek(0)
            tree.write(f)
        except Exception as inst:
            print(type(inst))

    def load(self, f):
        # this will load as XML
        super(DataObject, self).load(f)

        # have to add in extra bits, if it was saved as XML, through this
        # program
        try:
            f.seek(0)
            tree = ET.ElementTree()
            tree.parse(f)
        except Exception:
            # couldn't parse, is not an xml file.
            return

#         try:
        rdata = tree.find('.//R')
        for key in rdata.attrib:
            if key in self.__requiredgraphproperties:
                self.graph_properties['line2D_properties'][
                    key] = self.__requiredgraphproperties[key](rdata.attrib[key])

        fit = tree.find('.//fit')
        if fit is not None:
            for key in fit.attrib:
                if key in self.__requiredgraphproperties:
                    self.graph_properties['line2Dfit_properties'][
                        key] = self.__requiredgraphproperties[key](fit.attrib[key])
            self.fit = np.array([float(val) for val in fit.text.split()])

        residuals = tree.find('.//residuals')
        if residuals is not None:
            for key in residuals.attrib:
                if key in self.__requiredgraphproperties:
                    self.graph_properties[
                        'line2Dresiduals_properties'][
                        key] = self.__requiredgraphproperties[
                        key](
                        residuals.attrib[
                            key])
            self.residuals = np.array([float(val)
                                      for val in residuals.text.split()])

        sld_profile = tree.find('.//sld')
        if sld_profile:
            for key in sld_profile.attrib:
                if key in self.__requiredgraphproperties:
                    self.graph_properties[
                        'line2Dsld_profile_properties'][
                        key] = self.__requiredgraphproperties[
                        key](
                        sld_profile.attrib[
                            key])
            zed = tree.find('.//z')
            rho = tree.find('.//rho')
            self.sld_profile = []
            self.sld_profile.append(
                np.array([float(val) for val in zed.text.split()]))
            self.sld_profile.append(
                np.array([float(val) for val in rho.text.split()]))

#         except Exception as inst:
#             print type(inst)

    def _save_graph_properties(self):
        if self.line2D:
            for key in self.__requiredgraphproperties:
                self.graph_properties[
                    'line2D_properties'][
                    key] = artist.getp(
                    self.line2D,
                    key)

        if self.line2Dfit:
            for key in self.__requiredgraphproperties:
                self.graph_properties[
                    'line2Dfit_properties'][
                    key] = artist.getp(
                    self.line2Dfit,
                    key)

        if self.line2Dresiduals:
            for key in self.__requiredgraphproperties:
                self.graph_properties[
                    'line2Dresiduals_properties'][
                    key] = artist.getp(
                    self.line2Dresiduals,
                    key)

        if self.line2Dsld_profile:
            for key in self.__requiredgraphproperties:
                self.graph_properties[
                    'line2Dsld_profile_properties'][
                    key] = artist.getp(
                    self.line2Dsld_profile,
                    key)

    def do_a_fit(self, model, fitPlugin=None, method=None):
        '''
            TODO this should be somewhat refactored into GUI code
        '''

        callerInfo = deepcopy(model.__dict__)
        callerInfo['x'] = self.x
        callerInfo['y'] = self.y
        callerInfo['edata'] = self.y_err

        try:
            if model.usedq:
                callerInfo['dqvals'] = self.x_err
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass

        self.progressdialog = QtGui.QProgressDialog(
            "Fit progress",
            "Abort",
            0,
            100)
        self.progressdialog.setWindowModality(QtCore.Qt.WindowModal)

        if fitPlugin is not None:
            RFO = fitPlugin(**callerInfo)
        else:
            RFO = reflect.ReflectivityFitObject(**callerInfo)

        RFO.callback = self.callback
        model.parameters, model.uncertainties, self.chi2 = RFO.fit(
            method=method)
        model.covariance = RFO.covariance

        self.progressdialog.setValue(100)

        self.fit = RFO.model(model.parameters)
        self.residuals = RFO.residuals()

        self.sld_profile = None
        if 'sld_profile' in dir(RFO):
            self.sld_profile = RFO.sld_profile(model.parameters)

    def callback(self, xk, convergence=0.):
        try:
            self.progressdialog.setValue(int(convergence * 100))
            if self.progressdialog.wasCanceled():
                raise fitting.FitAbortedException('Fit aborted')
                return False
            else:
                return True
        except ValueError:
            return False

    def evaluate_chi2(self, model, store=False, fitPlugin=None):

        callerInfo = deepcopy(model.__dict__)
        callerInfo['x'] = self.x
        callerInfo['y'] = self.y
        callerInfo['edata'] = self.y_err

        try:
            if model.usedq:
                callerInfo['dqvals'] = self.x_err
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass

        if fitPlugin is not None:
            RFO = fitPlugin(**callerInfo)
        else:
            RFO = reflect.ReflectivityFitObject(**callerInfo)

        energy = RFO.energy() / self.numpoints
        if store:
            self.chi2 = energy

        return energy

    def evaluate_model(self, model, store=False, fitPlugin=None):

        callerInfo = deepcopy(model.__dict__)
        callerInfo['x'] = self.x
        callerInfo['y'] = self.y
        callerInfo['edata'] = self.y_err

        try:
            if model.usedq:
                callerInfo['dqvals'] = self.x_err
            else:
                del(callerInfo['dqvals'])
        except KeyError:
            pass

        if fitPlugin is not None:
            RFO = fitPlugin(**callerInfo)
        else:
            RFO = reflect.ReflectivityFitObject(**callerInfo)

        fit = RFO.model(callerInfo['parameters'])

        sld_profile = None
        if 'sld_profile' in dir(RFO):
            sld_profile = RFO.sld_profile(callerInfo['parameters'])

        if store:
            self.fit = fit
            self.residuals = (self.fit - RFO.ydata) / RFO.edata
            self.sld_profile = sld_profile

        return fit, fit - self.y, sld_profile
