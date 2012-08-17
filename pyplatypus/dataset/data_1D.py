""""

    A basic representation of a 1D datafile
        
"""


from __future__ import division
import string
import numpy as np
import os.path
import pyplatypus.util.ErrorProp as EP

class Data_1D(object):
    def __init__(self):
        self.W_q = np.zeros(0)
        self.W_ref = np.zeros(0)
        self.W_refSD = np.zeros(0)
        self.W_qSD = np.zeros(0)
            
        self.numpoints = 0
    
    def get_data(self):
        return (self.W_q, self.W_ref, self.W_refSD, self.W_qSD)
        
    def set_data(self, W_q, W_ref, *args):
        self.W_q = np.copy(W_q).flatten()
        self.W_ref = np.copy(W_ref).flatten()
        
        if len(args):
            self.W_refSD = np.copy(args[0]).flatten()
        else:
            self.W_refSD = np.zeros(np.size(self.W_q))
        
        if len(args) > 1:
            self.W_qSD = np.copy(args[1]).flatten()
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
        
        self.set_data(qq, rr, dr, dq)
        self.sort()

    def sort(self):
        sorted = np.argsort(self.W_q)
        self.W_q = self.W_q[:,sorted]
        self.W_ref = self.W_ref[:,sorted]
        self.W_refSD = self.W_refSD[:,sorted]
        self.W_qSD = self.W_qSD[:,sorted]

    def save_dat(self, f):
        np.savetxt(f, np.column_stack((self.W_q, self.W_ref, self.W_refSD, self.W_qSD)))
        
    def load_dat(self, f):
        array = np.loadtxt(f)
        self.name = os.path.basename(f)
        self.set_data(np.hsplit(array, np.size(array, 1)))
