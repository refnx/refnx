""""

    A basic representation of a 1D dataset
        
"""


from __future__ import division
import string
import numpy as np
import os.path
import pyplatypus.reduce.nsplice as nsplice
import pyplatypus.util.ErrorProp as EP

class Data_1D(object):
    def __init__(self, dataTuple = None):
    
        self.filename = None
        
        if dataTuple is not None:
            self.W_q = np.copy(dataTuple[0]).flatten()
            self.W_ref = np.copy(dataTuple[1]).flatten()
            if len(dataTuple) > 2:
                self.W_refSD = np.copy(dataTuple[2]).flatten()
            if len(dataTuple) > 3:
                self.W_qSD = np.copy(dataTuple[3]).flatten()
    
            self.numpoints = np.size(self.W_q, 0)
        
        else:
            self.W_q = np.zeros(0)
            self.W_ref = np.zeros(0)
            self.W_refSD = np.zeros(0)
            self.W_qSD = np.zeros(0)
                
            self.numpoints = 0
    
    def get_data(self):
        return (self.W_q, self.W_ref, self.W_refSD, self.W_qSD)
        
    def set_data(self, dataTuple):
        self.W_q = np.copy(dataTuple[0]).flatten()
        self.W_ref = np.copy(dataTuple[1]).flatten()
        
        if len(dataTuple) > 2:
            self.W_refSD = np.copy(dataTuple[2]).flatten()
        else:
            self.W_refSD = np.zeros(np.size(self.W_q))
        
        if len(dataTuple) > 3:
            self.W_qSD = np.copy(dataTuple[3]).flatten()
        else:
            self.W_qSD = np.zeros(np.size(self.W_q))
            
        self.numpoints = len(self.W_q)
        
    def scale(self, scalefactor = 1.):
        self.W_ref /= scalefactor
        self.W_refSD /= scalefactor
            
    def add_data(self, dataTuple, requires_splice = False):
        W_q, W_ref, W_refSD, W_qSD = self.get_data()

        aW_q, aW_ref, aW_refSD, aW_qSD = dataTuple
        
        qq = np.r_[W_q]
        rr = np.r_[W_ref]
        dr = np.r_[W_refSD]
        dq = np.r_[W_qSD]
    
        #go through and stitch them together.
        if requires_splice and self.numpoints > 1:
            scale, dscale = nsplice.get_scaling_in_overlap(qq,
                                                        rr,
                                                        dr,
                                                        aW_q,
                                                        aW_ref,
                                                        aW_refSD)
        else:
            scale = 1.
            dscale = 0.
                         
        qq = np.r_[qq, aW_q]
        dq = np.r_[dq, aW_qSD]  
        
        appendR, appendDR = EP.EPmul(aW_ref,
                                        aW_refSD,
                                            scale,
                                                dscale)
        rr = np.r_[rr, appendR]
        dr = np.r_[dr, appendDR]
        
        self.set_data((qq, rr, dr, dq))
        self.sort()

    def sort(self):
        sorted = np.argsort(self.W_q)
        self.W_q = self.W_q[:,sorted]
        self.W_ref = self.W_ref[:,sorted]
        self.W_refSD = self.W_refSD[:,sorted]
        self.W_qSD = self.W_qSD[:,sorted]

    def save(self, f):
        np.savetxt(f, np.column_stack((self.W_q, self.W_ref, self.W_refSD, self.W_qSD)))
        
    def load(self, f):
        array = np.loadtxt(f)
        self.filename = f.name
        self.name = os.path.basename(f.name)
        self.set_data(tuple(np.hsplit(array, np.size(array, 1))))
        
    def refresh(self):
        if self.filename:
            with open(self.filename) as f:
                self.load(f)
        
