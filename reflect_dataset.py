from __future__ import division
import string
import numpy as np
from time import gmtime, strftime
import nsplice
import rebin
import ErrorProp as EP

class Data_1D(object):
    def __init__(self, W_q, W_ref, W_refSD, W_qSD):
        self.W_q = np.copy(np.atleast_2d(W_q))
        self.W_ref = np.copy(np.atleast_2d(W_ref))
        self.W_refSD = np.copy(np.atleast_2d(W_refSD))
        self.W_qSD = np.copy(np.atleast_2d(W_qSD))
            
        self._numpoints = len(W_q[0])
    
    def getData(self):
        return (self.W_q, self.W_ref, self.W_refSD, self.W_qSD)
        
    def setData(self, W_q, W_ref, W_refSD, W_qSD):
        self.W_q = np.copy(np.atleast_2d(W_q))
        self.W_ref = np.copy(np.atleast_2d(W_ref))
        self.W_refSD = np.copy(np.atleast_2d(W_refSD))
        self.W_qSD = np.copy(np.atleast_2d(W_qSD))
            
        self._numpoints = len(self.W_q[0])
        
    def scale(self, scalefactor = 1.):
        self.W_ref /= scalefactor
        self.W_refSD /= scalefactor
            
    def add_data(self, Data_1D_obj, requires_splice = False):
        W_q, W_ref, W_refSD, W_qSD = self.getData()
        
        qq = np.r_[W_q[0]]
        rr = np.r_[W_ref[0]]
        dr = np.r_[W_refSD[0]]
        dq = np.r_[W_qSD[0]]
        #go through and stitch them together.
        if requires_splice:
            scale, dscale = nsplice.getScalingInOverlap(qq,
                                                        rr,
                                                        dr,
                                                        Data_1D_obj.W_q[0],
                                                        Data_1D_obj.W_ref[0],
                                                        Data_1D_obj.W_refSD[0])
        else:
            scale = 1.
            scale = 0.
                    
        qq = np.r_[qq, Data_1D_obj.W_q[0]]
        dq = np.r_[dq, Data_1D_obj.W_qSD[0]]
        appendR, appendDR = EP.EPmul(Data_1D_obj.W_ref[0],
                                        Data_1D_obj.W_refSD[0],
                                            scale,
                                                dscale)
        rr = np.r_[rr, appendR]
        dr = np.r_[dr, appendDR]
        
        self.setData(qq, rr, dr, dq)
        self.sort()
                            
    def sort(self):
        sorted = np.argsort(self.W_q[0])
        self.W_q = self.W_q[:,sorted]
        self.W_ref = self.W_ref[:,sorted]
        self.W_refSD = self.W_refSD[:,sorted]
        self.W_qSD = self.W_qSD[:,sorted]
        
class Reflect_Dataset_2D(object):
    __template_ref_xml = """<?xml version="1.0"?>
    <REFroot xmlns="">
      <REFentry time="$time">
        <Title>$title</Title>
        <User>$user</User>
        <REFsample>
          <ID>$sample</ID>
        </REFsample>
        <REFdata axes="Qz:Qy" rank="1" type="POINT" spin="UNPOLARISED" dim="$_numpointsz:$_numpointsy">
          <Run filename="$_rnumber" preset="" size="">
          </Run>
          <R uncertainty="dR">$_r</R>
          <Qz uncertainty="dQz" units="1/A">$_qz</Qz>
          <dR type="SD">$_dr</dR>
          <Qy type="FWHM" units="1/A">$_qy</Qy>
        </REFdata>
      </REFentry>
    </REFroot>"""
    
    def __init__(self, M_qz, M_qy, M_ref, M_refSD, rnumber):
        self.M_qz = M_qz
        self.M_ref = M_ref
        self.M_refSD = M_refSD
        self.M_qy = M_qy
        self._rnumber = rnumber
            
        self._numpointsz = np.size(M_qz, axis = 1)
        self._numpointsy = np.size(M_qz, axis = 2)
    
    def getData(self):
        return (self.M_qz, self.M_qy, self.M_ref, self.M_refSD)
       
    def setData(self, M_qz, M_qy, M_ref, M_refSD):
        self.M_qz = M_qz
        self.M_ref = M_ref
        self.M_refSD = M_refSD
        self.M_qy = M_qy
            
        self._numpointsz = np.size(M_qz, axis = 1)
        self._numpointsy = np.size(M_qz, azis = 2)
        
    def scale(self, scalefactor = 1.):
        self.M_ref /= scalefactor
        self.M_refSD /= scalefactor
        
    def write_xml(self):
        s = string.Template(self.__template_ref_xml)
        self.time = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())

        for index in xrange(len(self.M_qz)):
            filename = 'off_PLP{:07d}_{:d}.xml'.format(self._rnumber, index)
            self._r = string.translate(repr(self.M_ref[index].tolist()), None, ',[]')
            self._qz = string.translate(repr(self.M_qz[index].tolist()), None, ',[]')
            self._dr = string.translate(repr(self.M_refSD[index].tolist()), None, ',[]')
            self._qy = string.translate(repr(self.M_qy[index].tolist()), None, ',[]')
            f = open(filename, 'w')
            thefile = s.safe_substitute(self.__dict__)
            f.write(thefile)
            f.truncate()
            f.close()
 
class Reflect_Dataset(Data_1D):
    __template_ref_xml = """<?xml version="1.0"?>
    <REFroot xmlns="">
      <REFentry time="$time">
        <Title>$title</Title>
        <User>$user</User>
        <REFsample>
          <ID>$sample</ID>
        </REFsample>
        <REFdata axes="Qz" rank="1" type="POINT" spin="UNPOLARISED" dim="$_numpoints">
          <Run filename="$_rnumber" preset="" size="">
          </Run>
          <R uncertainty="dR">$_r</R>
          <Qz uncertainty="dQz" units="1/A">$_q</Qz>
          <dR type="SD">$_dr</dR>
          <dQz type="FWHM" units="1/A">$_dq</dQz>
        </REFdata>
      </REFentry>
    </REFroot>"""
    
    __allowed_attributes = ['title', 'sample', 'user', 'time', 'cmd', '_rnumber',
                              '_r', '_q', '_dr', '_dq', '_W_q', '_W_ref']

    def __init__(self, *args, **kwds):
        #*args should be a tuple of q, r, dr, dq data
        super(Reflect_Dataset, self).__init__(*args[0])
        if kwds['Reduced_Output']:
            Reduced_Output_obj = kwds['Reduced_Output']
            'title', 'sample', 'user', 'rnumber', 'dnumber'
            self.sample = Reduced_Output_obj.reflect_beam_spectrum['sample']
            self.title = Reduced_Output_obj.reflect_beam_spectrum['title']
            self.user = Reduced_Output_obj.reflect_beam_spectrum['user']
            self.time = ''
            self._rnumber = [Reduced_Output_obj.reflect_beam_spectrum['runnumber']]
        else:
            self.sample = ''
            self.title = ''
            self.user = ''
            self.time = ''
            self._rnumber = list()
            
        #sort the data as it goes in
        self.sort()
        
    def add_dataset(self, Reduced_Output_obj):
        #when you add a dataset to splice only the first numspectra dimension is used.
        #the others are discarded
        self.add_data(Data_1D(*Reduced_Output_obj.get1Ddata()), requires_splice = True)
        self._rnumber.append(Reduced_Output_obj.reflect_beam_spectrum['runnumber'])                                                    
        
    def write_xml(self):
        s = string.Template(self.__template_ref_xml)
        self.time = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
        
        W_q, W_ref, W_refSD, W_qSD = self.getData()
        
        #the data to write
        if len(self._rnumber) > 1:
            filename = 'c_PLP{:07d}_{:d}.xml'.format(self._rnumber[0], 0)
            
            self._r = string.translate(repr(W_ref[0].tolist()), None, ',[]')
            self._q = string.translate(repr(W_q[0].tolist()), None, ',[]')
            self._dr = string.translate(repr(W_refSD[0].tolist()), None, ',[]')
            self._dq = string.translate(repr(W_qSD[0].tolist()), None, ',[]')
            f = open(filename, 'w')
            thefile = s.safe_substitute(self.__dict__)
            f.write(thefile)
            f.truncate()
            f.close()
        else:
            for index in xrange(len(W_q)):
                if len(self._rnumber):
                    filename = 'PLP{:07d}_{:d}.xml'.format(self._rnumber[0], index)
                else:
                    filename = 'refXML_{:d}.xml'.format(index)
                self._r = string.translate(repr(W_ref[index].tolist()), None, ',[]')
                self._q = string.translate(repr(W_q[index].tolist()), None, ',[]')
                self._dr = string.translate(repr(W_refSD[index].tolist()), None, ',[]')
                self._dq = string.translate(repr(W_qSD[index].tolist()), None, ',[]')
                f = open(filename, 'w')
                thefile = s.safe_substitute(self.__dict__)
                f.write(thefile)
                f.truncate()
                f.close()
        
    def write_dat(self):
        W_q, W_ref, W_refSD, W_qSD = self.getData()
        
        #the data to write
        if len(self._rnumber) > 1:                
            filename = 'c_PLP{:07d}_{:d}.dat'.format(self._rnumber[0], 0)
            
            self.__r = string.translate(repr(W_ref[0].tolist()), None, ',[]')
            self.__q = string.translate(repr(W_q[0].tolist()), None, ',[]')
            self.__dr = string.translate(repr(W_refSD[0].tolist()), None, ',[]')
            self.__dq = string.translate(repr(W_qSD[0].tolist()), None, ',[]')
            
            f = open(filename, 'w')
            for q, r, dr, dq in zip(W_q[0], W_ref[0], W_refSD[0], W_qSD[0]):
                thedata = '{:g}\t{:g}\t{:g}\t{:g}\n'.format(q, r, dr, dq)
                f.write(thedata)
            f.close()
        else:
            for index in xrange(len(W_q)):
                if len(self._rnumber):
                    filename = 'PLP{:07d}_{:d}.dat'.format(self._rnumber[0], index)
                else:
                    filename = 'refXML_{:d}.dat'.format(index)
                f = open(filename, 'w')
                for q, r, dr, dq in zip(W_q[index], W_ref[index], W_refSD[index], W_qSD[index]):
                    thedata = '{:g}\t{:g}\t{:g}\t{:g}\n'.format(q, r, dr, dq)
                    f.write(thedata)
                f.close()
        
    def rebin(self, rebinpercent = 4):
        W_q, W_ref, W_refSD, W_qSD = self.getData()
        frac = 1. + (rebinpercent/100.)

        lowQ = (2 * W_q[0,0]) / ( 1. + frac)
        hiQ =  frac * (2 * W_q[0,-1]) / ( 1. + frac)       

        for index in xrange(len(W_q)):
            qq, rr, dr, dq = rebin.rebin_Q(W_q[index],
                                            W_ref[index],
                                             W_refSD[index],
                                              W_qSD[index],
                                               lowerQ = lowQ,
                                                upperQ = hiQ,
                                                 rebinpercent = rebinpercent)
            if index:
                qdat = np.vstack((qdat, qq))
                rdat = np.vstack((rdat, rr))
                drdat = np.vstack((drdat, dr))
                qsddat = np.vstack((qsdat, dq))
            else:
                qdat = qq
                rdat = rr
                drdat = dr
                qsddat = dq
            
        self.setData(qdat, rdat, drdat, qsddat)

class Reduced_Output(object):
    __allowed_attributes = ['W_q', 'W_qSD', 'W_ref','W_refSD','M_ref','M_refSD', 'M_qz', 'M_qzSD', 'M_qy',
                            'reflect_beam_spectrum', 'direct_beam_spectrum']
    def __init__(self, output):
        for key, value in output.items():
            if key in Reduced_Output.__allowed_attributes:
                object.__setattr__(self, key, value)
                #self.key = value
    
    def get1Ddata(self):
        return self.W_q, self.W_ref, self.W_refSD, self.W_qSD

    def get2Ddata(self):
        return (M_qz, M_qy, M_ref, M_refSD)     
        