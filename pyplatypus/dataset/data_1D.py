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

    def __init__(self, dataTuple=None):

        self.filename = None

        if dataTuple is not None:
            self.xdata = np.copy(dataTuple[0]).flatten()
            self.ydata = np.copy(dataTuple[1]).flatten()
            if len(dataTuple) > 2:
                self.ydataSD = np.copy(dataTuple[2]).flatten()
            if len(dataTuple) > 3:
                self.xdataSD = np.copy(dataTuple[3]).flatten()

            self.numpoints = np.size(self.xdata, 0)

        else:
            self.xdata = np.zeros(0)
            self.ydata = np.zeros(0)
            self.ydataSD = np.zeros(0)
            self.xdataSD = np.zeros(0)

            self.numpoints = 0

    @property
    def data(self):
        return (self.xdata, self.ydata, self.ydataSD, self.xdataSD)

    @data.setter
    def data(self, dataTuple):
        self.xdata = np.copy(dataTuple[0]).flatten()
        self.ydata = np.copy(dataTuple[1]).flatten()

        if len(dataTuple) > 2:
            self.ydataSD = np.copy(dataTuple[2]).flatten()
        else:
            self.ydataSD = np.ones_like(self.xdata)

        if len(dataTuple) > 3:
            self.xdataSD = np.copy(dataTuple[3]).flatten()
        else:
            self.xdataSD = np.zeros(np.size(self.xdata))

        self.numpoints = len(self.xdata)

    def scale(self, scalefactor=1.):
        self.ydata /= scalefactor
        self.ydataSD /= scalefactor

    def add_data(self, dataTuple, requires_splice=False):
        xdata, ydata, ydataSD, xdataSD = self.data()

        axdata, aydata, aydataSD, axdataSD = dataTuple

        qq = np.r_[xdata]
        rr = np.r_[ydata]
        dr = np.r_[ydataSD]
        dq = np.r_[xdataSD]

        # go through and stitch them together.
        if requires_splice and self.numpoints > 1:
            scale, dscale = nsplice.get_scaling_in_overlap(qq,
                                                           rr,
                                                           dr,
                                                           axdata,
                                                           aydata,
                                                           aydataSD)
        else:
            scale = 1.
            dscale = 0.

        qq = np.r_[qq, axdata]
        dq = np.r_[dq, axdataSD]

        appendR, appendDR = EP.EPmul(aydata,
                                     aydataSD,
                                     scale,
                                     dscale)
        rr = np.r_[rr, appendR]
        dr = np.r_[dr, appendDR]

        self.set_data((qq, rr, dr, dq))
        self.sort()

    def sort(self):
        sorted = np.argsort(self.xdata)
        self.xdata = self.xdata[:, sorted]
        self.ydata = self.ydata[:, sorted]
        self.ydataSD = self.ydataSD[:, sorted]
        self.xdataSD = self.xdataSD[:, sorted]

    def save(self, f):
        np.savetxt(
            f, np.column_stack((self.xdata,
                                self.ydata,
                                self.ydataSD,
                                self.xdataSD)))

    def load(self, f):
        array = np.loadtxt(f)
        self.filename = f.name
        self.name = os.path.basename(f.name)
        self.data = tuple(np.hsplit(array, np.size(array, 1)))

    def refresh(self):
        if self.filename:
            with open(self.filename) as f:
                self.load(f)
