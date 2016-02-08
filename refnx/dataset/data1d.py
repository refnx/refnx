""""
A basic representation of a 1D dataset
"""
from __future__ import division

import numpy as np
import os.path

from refnx.util.nsplice import get_scaling_in_overlap


class Data1D(object):
    """
    A basic representation of a 1D dataset.

    Parameters
    ----------
    data : str, file-like or tuple of np.ndarray, optional
        `data` can be a string or file-like object referring to a File to load
        the dataset from.

        Alternatively it is a tuple containing the data from which the dataset
        will be constructed. The tuple should have between 2 and 4 members.
            data[0] - x
            data[1] - y
            data[2] - standard deviation of y, y_sd
            data[3] - standard deviation of x, x_sd

        `data` must be at least two long, `x` and `y`.
        If the tuple is at least 3 long then the third member is `y_sd`.
        If the tuple is 4 long then the fourth member is `x_sd`.
        All arrays must have the same shape.

    Attributes
    ----------
    npoints
    data
    finite_data
    x : np.ndarray
        x data
    y : np.ndarray
        y data
    y_sd : np.ndarray
        uncertainties (1 standard deviation) on the y data
    x_sd : np.ndarray
        uncertainties on the x data
    """
    def __init__(self, data=None):
        self.filename = None
        self.fit = None
        self.params = None
        self.chisqr = np.inf
        self.residuals = None

        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.y_sd = np.zeros(0)
        self.x_sd = np.zeros(0)

        # if it's a file then open and load the file.
        if hasattr(data, 'read') or type(data) is str:
            self.load(data)
        elif data is not None:
            self.x = np.asfarray(data[0])
            self.y = np.asfarray(data[1])
            if len(data) > 2:
                self.y_sd = np.asfarray(data[2])
            if len(data) > 3:
                self.x_sd = np.asfarray(data[3])

    @property
    def npoints(self):
        """
        The number of points in the dataset.

        Returns
        -------
        npoints : int
            How many points in the dataset
        """
        return self.y.size

    @property
    def data(self):
        """
        The data contained within this object.

        Returns
        -------
        data_tuple : tuple
            4-tuple containing the (x, y, y_sd, x_sd) data
        """
        return self.x, self.y, self.y_sd, self.x_sd

    @property
    def finite_data(self):
        """
        Returns
        -------
        dataTuple : tuple
            4-tuple containing the (x, y, y_sd, x_sd) datapoints that are
            finite.
        """
        finite_loc = np.where(np.isfinite(self.y))
        return (self.x[finite_loc],
                self.y[finite_loc],
                self.y_sd[finite_loc],
                self.x_sd[finite_loc])

    @data.setter
    def data(self, data_tuple):
        """
        Set the data for this object from supplied data.

        Parameters
        ----------
        data_tuple : tuple
            2 to 4 member tuple containing the (x, y, y_sd, x_sd) data to
            specify the dataset. `y_sd` and `x_sd` are optional.
        """
        self.x = np.asfarray(data_tuple[0])
        self.y = np.asfarray(data_tuple[1])

        if len(data_tuple) > 2:
            self.y_sd = np.asfarray(data_tuple[2])
        else:
            self.y_sd = np.ones_like(self.y)

        if len(data_tuple) > 3:
            self.x_sd = np.asfarray(data_tuple[3])
        else:
            self.x_sd = np.zeros(np.size(self.x))
        self.sort()

    def scale(self, scalefactor=1.):
        """
        Scales the y and y_sd data by dividing by `scalefactor`.

        Parameters
        ----------
        scalefactor : float
            The scalefactor to divide by.
        """
        self.y /= scalefactor
        self.y_sd /= scalefactor

    def add_data(self, data_tuple, requires_splice=False, trim_trailing=True):
        """
        Adds more data to the dataset

        Parameters
        ----------
        data_tuple : tuple
            2 to 4 member tuple containing the (x, y, y_sd, x_sd) data to add
            to the dataset. `y_sd` and `x_sd` are optional.
        requires_splice : bool, optional
            When the new data is added to the dataset do you want to scale it
            vertically so that it overlaps with the existing data? `y` and
            `y_sd` in `data_tuple` are both multiplied by the scaling factor.
        trim_trailing : bool, optional
            When the new data is concatenated do you want to remove points from
            the existing data that are in the overlap region? This might be
            done because the datapoints in the `data_tuple` you are adding have
            have lower `y_sd` than the preceding data.
        """
        xdata, ydata, ydata_sd, xdata_sd = self.data

        axdata, aydata = data_tuple[0:2]

        if len(data_tuple) > 2:
            aydata_sd = np.asfarray(data_tuple[2])
        else:
            aydata_sd = np.ones_like(aydata)

        if len(data_tuple) > 3:
            axdata_sd = np.asfarray(data_tuple[3])
        else:
            axdata_sd = np.zeros(np.size(axdata))

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
                get_scaling_in_overlap(qq,
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
        """
        Sorts the data in ascending order
        """
        sorted = np.argsort(self.x)
        self.x = self.x[sorted]
        self.y = self.y[sorted]
        self.y_sd = self.y_sd[sorted]
        self.x_sd = self.x_sd[sorted]

    def save(self, f):
        """
        Saves the data to file. Saves the data as 4 column ASCII.

        Parameters
        ----------
        f : file-handle or string
            File to save the dataset to.
        """
        np.savetxt(
            f, np.column_stack((self.x,
                                self.y,
                                self.y_sd,
                                self.x_sd)))

    def save_fit(self, f):
        if self.fit is not None:
            np.savetxt(f, np.column_stack((self.x,
                                           self.fit)))

    def load(self, f):
        """
        Loads a dataset from file. Must be 2 to 4 column ASCII.

        Parameters
        ----------
        f : file-handle or string
            File to load the dataset from.
        """
        array = np.loadtxt(f)
        if hasattr(f, 'read'):
            fname = f.name
        else:
            fname = f

        self.filename = fname
        self.name = os.path.splitext(os.path.basename(fname))[0]

        self.data = [np.squeeze(array[:, col]) for col in range(np.size(array, 1))]

    def refresh(self):
        """
        Refreshes a previously loaded dataset.
        """
        if self.filename is not None:
            with open(self.filename) as f:
                self.load(f)
