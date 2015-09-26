""""
    A basic representation of a 1D dataset
"""
from __future__ import division
import numpy as np
import os.path
import refnx.reduce.nsplice as nsplice


class Data1D(object):

    def __init__(self, dataTuple=None, curvefitter=None):
        self.filename = None
        self.fit = None
        self.params = None
        self.chisqr = np.inf
        self.residuals = None

        if dataTuple is not None:
            self.xdata = np.copy(dataTuple[0]).flatten()
            self.ydata = np.copy(dataTuple[1]).flatten()
            if len(dataTuple) > 2:
                self.ydata_sd = np.copy(dataTuple[2]).flatten()
            if len(dataTuple) > 3:
                self.xdata_sd = np.copy(dataTuple[3]).flatten()

        else:
            self.xdata = np.zeros(0)
            self.ydata = np.zeros(0)
            self.ydata_sd = np.zeros(0)
            self.xdata_sd = np.zeros(0)

    @property
    def npoints(self):
        return np.size(self.ydata, 0)

    @property
    def data(self):
        return self.xdata, self.ydata, self.ydata_sd, self.xdata_sd

    @property
    def finite_data(self):
        finite_loc = np.where(np.isfinite(self.ydata))
        return (self.xdata[finite_loc],
                self.ydata[finite_loc],
                self.ydata_sd[finite_loc],
                self.xdata_sd[finite_loc])

    @data.setter
    def data(self, data_tuple):
        self.xdata = np.copy(data_tuple[0]).flatten()
        self.ydata = np.copy(data_tuple[1]).flatten()

        if len(data_tuple) > 2:
            self.ydata_sd = np.copy(data_tuple[2]).flatten()
        else:
            self.ydata_sd = np.ones_like(self.xdata)

        if len(data_tuple) > 3:
            self.xdata_sd = np.copy(data_tuple[3]).flatten()
        else:
            self.xdata_sd = np.zeros(np.size(self.xdata))

    def scale(self, scalefactor=1.):
        self.ydata /= scalefactor
        self.ydata_sd /= scalefactor

    def add_data(self, data_tuple, requires_splice=False, trim_trailing=True):
        xdata, ydata, ydata_sd, xdata_sd = self.data

        axdata, aydata, aydata_sd, axdata_sd = data_tuple

        qq = np.r_[xdata]
        rr = np.r_[ydata]
        dr = np.r_[ydata_sd]
        dq = np.r_[xdata_sd]

        # which values in the first dataset overlap with the second
        overlap_points = np.zeros_like(qq, 'bool')

        # go through and stitch them together.
        scale = 1.
        dscale = 0.
        if requires_splice and self.npoints > 1:
            scale, dscale, overlap_points = (
                nsplice.get_scaling_in_overlap(qq,
                                               rr,
                                               dr,
                                               axdata,
                                               aydata,
                                               aydata_sd))
        if not trim_trailing:
            overlap_points[:] = False

        qq = np.r_[qq[~overlap_points], axdata]
        dq = np.r_[dq[~overlap_points], axdata_sd]

        rr = np.r_[rr[~overlap_points], aydata * scale]
        dr = np.r_[dr[~overlap_points], aydata_sd * scale]

        self.data = (qq, rr, dr, dq)
        self.sort()

    def sort(self):
        sorted = np.argsort(self.xdata)
        self.xdata = self.xdata[sorted]
        self.ydata = self.ydata[sorted]
        self.ydata_sd = self.ydata_sd[sorted]
        self.xdata_sd = self.xdata_sd[sorted]

    def save(self, f):
        np.savetxt(
            f, np.column_stack((self.xdata,
                                self.ydata,
                                self.ydata_sd,
                                self.xdata_sd)))

    def save_fit(self, f):
        if self.fit is not None:
            np.savetxt(f, np.column_stack((self.xdata,
                                           self.fit)))

    def load(self, f):
        array = np.loadtxt(f)
        self.filename = f.name
        self.name = os.path.splitext(os.path.basename(f.name))[0]
        self.data = tuple(np.hsplit(array, np.size(array, 1)))

    def refresh(self):
        if self.filename:
            with open(self.filename) as f:
                self.load(f)
