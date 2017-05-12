""""
A basic representation of a 1D dataset
"""
from __future__ import division

import os.path
import re

import numpy as np

from refnx.util.nsplice import get_scaling_in_overlap


class Data1D(object):
    r"""
    A basic representation of a 1D dataset.

    Parameters
    ----------
    data : str, file-like or tuple of np.ndarray, optional
        `data` can be a string or file-like object referring to a File to load
        the dataset from.

        Alternatively it is a tuple containing the data from which the dataset
        will be constructed. The tuple should have between 2 and 4 members.

            - data[0] - x
            - data[1] - y
            - data[2] - uncertainties on y, y_err
            - data[3] - uncertainties of x, x_err

        `data` must be at least two long, `x` and `y`.
        If the tuple is at least 3 long then the third member is `y_err`.
        If the tuple is 4 long then the fourth member is `x_err`.
        All arrays must have the same shape.

    Attributes
    ----------
    data : tuple of np.ndarray
        The data, (x, y, y_err, x_err)
    finite_data : tuple of np.ndarray
        Data points that are finite
    x : np.ndarray
        x data
    y : np.ndarray
        y data
    y_err : np.ndarray
        uncertainties on the y data
    x_err : np.ndarray
        uncertainties on the x data
    filename : str or None
        The file the data was read from
    weighted : bool
        Whether the y data has uncertainties
    metadata : dict
        Information that should be retained with the dataset.
    """
    def __init__(self, data=None, **kwds):
        self.filename = None

        self.metadata = kwds
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.y_err = np.zeros(0)
        self.x_err = np.zeros(0)
        self.weighted = False

        # if it's a file then open and load the file.
        if hasattr(data, 'read') or type(data) is str:
            self.load(data)
        elif data is not None:
            self.x = np.array(data[0], dtype=float)
            self.y = np.array(data[1], dtype=float)
            if len(data) > 2:
                self.y_err = np.array(data[2], dtype=float)
                self.weighted = True
            else:
                self.y_err = np.ones_like(self.y, dtype=float)

            if len(data) > 3:
                self.x_err = np.array(data[3], dtype=float)
            else:
                self.x_err = np.zeros_like(self.x, dtype=float)

    def __len__(self):
        """
        the number of points in the dataset.
        """
        return self.y.size

    @property
    def data(self):
        """
        4-tuple containing the (x, y, y_err, x_err) data
        """
        return self.x, self.y, self.y_err, self.x_err

    @property
    def finite_data(self):
        """
        4-tuple containing the (x, y, y_err, x_err) datapoints that are
        finite.
        """
        finite_loc = np.where(np.isfinite(self.y))
        return (self.x[finite_loc],
                self.y[finite_loc],
                self.y_err[finite_loc],
                self.x_err[finite_loc])

    @data.setter
    def data(self, data_tuple):
        """
        Set the data for this object from supplied data.

        Parameters
        ----------
        data_tuple : tuple
            2 to 4 member tuple containing the (x, y, y_err, x_err) data to
            specify the dataset. `y_err` and `x_err` are optional.
        """
        self.x = np.array(data_tuple[0], dtype=float)
        self.y = np.array(data_tuple[1], dtype=float)
        self.weighted = False
        if len(data_tuple) > 2:
            self.y_err = np.array(data_tuple[2], dtype=float)
            self.weighted = True
        else:
            self.y_err = np.ones_like(self.y, dtype=float)

        if len(data_tuple) > 3:
            self.x_err = np.array(data_tuple[3], dtype=float)
        else:
            self.x_err = np.zeros_like(self.x, dtype=float)
        self.sort()

    def scale(self, scalefactor=1.):
        """
        Scales the y and y_err data by dividing by `scalefactor`.

        Parameters
        ----------
        scalefactor : float
            The scalefactor to divide by.
        """
        self.y /= scalefactor
        self.y_err /= scalefactor

    def add_data(self, data_tuple, requires_splice=False, trim_trailing=True):
        """
        Adds more data to the dataset

        Parameters
        ----------
        data_tuple : tuple
            2 to 4 member tuple containing the (x, y, y_err, x_err) data to add
            to the dataset. `y_err` and `x_err` are optional.
        requires_splice : bool, optional
            When the new data is added to the dataset do you want to scale it
            vertically so that it overlaps with the existing data? `y` and
            `y_err` in `data_tuple` are both multiplied by the scaling factor.
        trim_trailing : bool, optional
            When the new data is concatenated do you want to remove points from
            the existing data that are in the overlap region? This might be
            done because the datapoints in the `data_tuple` you are adding have
            have lower `y_err` than the preceding data.

        Notes
        -----
        Raises `ValueError` if there are no points in the overlap region and
        `requires_splice` was True
        """
        xdata, ydata, ydata_sd, xdata_sd = self.data

        axdata, aydata = data_tuple[0:2]

        if len(data_tuple) > 2:
            aydata_sd = np.array(data_tuple[2], dtype=float)
        else:
            aydata_sd = np.ones_like(aydata)
            self.weighted = False

        if len(data_tuple) > 3:
            axdata_sd = np.array(data_tuple[3], dtype=float)
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
        if requires_splice and len(self) > 1:
            scale, dscale, overlap_points = (
                get_scaling_in_overlap(qq,
                                       rr,
                                       dr,
                                       axdata,
                                       aydata,
                                       aydata_sd))

            if ((not np.isfinite(scale)) or (not np.isfinite(dscale)) or
                    (not np.size(overlap_points, 0))):
                raise ValueError("No points in overlap region")

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
        self.y_err = self.y_err[sorted]
        self.x_err = self.x_err[sorted]

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
                                self.y_err,
                                self.x_err)))

    def load(self, f):
        """
        Loads a dataset from file. Must be 2 to 4 column ASCII.

        Parameters
        ----------
        f : file-handle or string
            File to load the dataset from.
        """
        # see if there are header rows
        close_file = False
        try:
            g = open(f, 'rb')
            close_file = True
        except TypeError:
            # if you're already a file then you'll get a type error
            g = f
        finally:
            header_lines = 0
            for i, line in enumerate(g):
                try:
                    nums = [float(tok) for tok in
                            re.split('\s|,', line.decode('utf-8'))
                            if len(tok)]
                    if len(nums) >= 2:
                        header_lines = i
                        break
                except ValueError:
                    continue
            if close_file:
                g.close()

        self.data = np.loadtxt(f, unpack=True, skiprows=header_lines)

        if hasattr(f, 'read'):
            fname = f.name
        else:
            fname = f

        self.filename = fname
        self.name = os.path.splitext(os.path.basename(fname))[0]

    def refresh(self):
        """
        Refreshes a previously loaded dataset.
        """
        if self.filename is not None:
            with open(self.filename) as f:
                self.load(f)

    def __add__(self, other):
        """
        Adds two datasets together. Splices the data and trims data in the
        overlap region.
        """
        ret = Data1D(self.data)
        ret.add_data(other.data, requires_splice=True, trim_trailing=True)
        return ret

    def __radd__(self, other):
        """
        radd of two datasets. Splices the data and trims data in the
        overlap region.
        """
        self.add_data(other.data, requires_splice=True, trim_trailing=True)
        return self

    def __iadd__(self, other):
        """
        iadd of two datasets. Splices the data and trims data in the
        overlap region.
        """
        self.add_data(other.data, requires_splice=True, trim_trailing=True)
        return self
