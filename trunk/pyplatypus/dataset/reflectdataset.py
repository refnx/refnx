from __future__ import division
import string
import numpy as np
from time import gmtime, strftime
import pyplatypus.reduce.rebin
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os.path
import pyplatypus.util.ErrorProp as EP
from data_1D import Data_1D
 
class ReflectDataset(Data_1D):
    _template_ref_xml = """<?xml version="1.0"?>
    <REFroot xmlns="">
    <REFentry time="$time">
    <Title>$title</Title>
    <User>$user</User>
    <REFsample>
    <ID>$sample</ID>
    </REFsample>
    <REFdata axes="Qz" rank="1" type="POINT" spin="UNPOLARISED" dim="$numpoints">
    <Run filename="$datafilenumber" preset="" size="">
    </Run>
    <R uncertainty="dR">$_W_ref</R>
    <Qz uncertainty="dQz" units="1/A">$_W_q</Qz>
    <dR type="SD">$_W_refSD</dR>
    <dQz type="FWHM" units="1/A">$_W_qSD</dQz>
    </REFdata>
    </REFentry>
    </REFroot>"""

    def __init__(self, dataTuple = None, datasets = None, **kwds):
        #args should be a list of reduce objects
        super(ReflectDataset, self).__init__(dataTuple = dataTuple)
        self.datafilenumber = list()
        if datasets is not None:
            [self.add_dataset(data) for data in datasets]
            
    def add_dataset(self, reduceObj, scanpoint = 0):
        #when you add a dataset to splice only the first numspectra dimension is used.
        #the others are discarded
        self.add_data(reduceObj.get_1D_data(scanpoint = scanpoint), requires_splice = True)
        self.datafilenumber.append(reduceObj.datafilenumber)                                                    
        
    def save(self, f):
        s = string.Template(self._template_ref_xml)
        self.time = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

        #filename = 'c_PLP{:07d}_{:d}.xml'.format(self._rnumber[0], 0)

        self._W_ref = string.translate(repr(self.W_ref.tolist()), None, ',[]')
        self._W_q = string.translate(repr(self.W_q.tolist()), None, ',[]')
        self._W_refSD = string.translate(repr(self.W_refSD.tolist()), None, ',[]')
        self._W_qSD = string.translate(repr(self.W_qSD.tolist()), None, ',[]')

        thefile = s.safe_substitute(self.__dict__)
        f.write(thefile)
#        f.truncate()
        
    def load(self, f):
        try:
            tree = ET.ElementTree()
            tree.parse(f)
            qtext = tree.find('.//Qz')
            rtext = tree.find('.//R')
            drtext = tree.find('.//dR')
            dqtext = tree.find('.//dQz')
    
            qvals = [float(val) for val in qtext.text.split()]
            rvals = [float(val) for val in rtext.text.split()]
            drvals = [float(val) for val in drtext.text.split()]
            dqvals = [float(val) for val in dqtext.text.split()]
            
            self.filename = f.name
            self.name = os.path.basename(f.name)
            self.set_data((qvals, rvals, drvals, dqvals)) 
        except ET.ParseError:
            with open(f.name, 'Ur') as g:
                super(ReflectDataset, self).load(g)
        
    def rebin(self, rebinpercent = 4):
        W_q, W_ref, W_refSD, W_qSD = self.get_data()
        frac = 1. + (rebinpercent/100.)

        lowQ = (2 * W_q[0]) / ( 1. + frac)
        hiQ =  frac * (2 * W_q[-1]) / ( 1. + frac)       

        
        qq, rr, dr, dq = rebin.rebin_Q(W_q,
                                        W_ref,
                                        W_refSD,
                                        W_qSD,
                                        lowerQ = lowQ,
                                        upperQ = hiQ,
                                        rebinpercent = rebinpercent)
        qdat = qq
        rdat = rr
        drdat = dr
        qsddat = dq

        self.set_data((qdat, rdat, drdat, qsddat))


  
        
