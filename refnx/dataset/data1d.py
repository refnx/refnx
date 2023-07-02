""""
A basic representation of a 1D dataset
"""
import os.path
from pathlib import Path, PurePath
import re

import numpy as np
from scipy._lib._util import check_random_state
from refnx.util.nsplice import get_scaling_in_overlap
from refnx._lib import possibly_open_file


class Data1D:
    r"""
    A basic representation of a 1D dataset.

    Parameters
    ----------
    data : {str, file-like, Path, tuple of np.ndarray}, optional
        `data` can be a string, file-like, or Path object referring to a File
        to load the dataset from. The file should be plain text and have
        2 to 4 columns separated by space, comma or tab. The columns represent
        `x, y [y_err [, x_err]]`.

        Alternatively it is a tuple containing the data from which the dataset
        will be constructed. The tuple should have between 2 and 4 members.

            - data[0] - x
            - data[1] - y
            - data[2] - uncertainties on y, y_err
            - data[3] - uncertainties on x, x_err

        `data` must be at least two long, `x` and `y`.
        If the tuple is at least 3 long then the third member is `y_err`.
        If the tuple is 4 long then the fourth member is `x_err`.
        All arrays must have the same shape.

    mask : array-like
        Specifies which data points are (un)masked. Must be broadcastable
        to the y-data. `Data1D.mask = None` clears the mask. If a mask value
        equates to `True`, then the point is included, if a mask value equates
        to `False` it is excluded.

    Attributes
    ----------
    filename : {str, Path, None}
        The file the data was read from
    weighted : bool
        Whether the y data has uncertainties
    metadata : dict
        Information that should be retained with the dataset.

    """

    def __init__(self, data=None, mask=None, **kwds):
        self.filename = None
        self.name = None

        self.metadata = kwds
        self._x = np.zeros(0)
        self._y = np.zeros(0)
        self._y_err = None
        self._x_err = None
        self.weighted = False

        # if it's a file then open and load the file.
        if (
            hasattr(data, "read")
            or type(data) is str
            or isinstance(data, PurePath)
        ):
            self.load(data)
        elif isinstance(data, Data1D):
            # copy a dataset
            self.name = data.name
            self.filename = data.filename
            self.metadata = data.metadata
            self._x = data._x
            self._y = data._y
            self._y_err = data._y_err
            self._x_err = data._x_err
            self.weighted = data.weighted
            self._mask = data._mask
        elif data is not None:
            self._x = np.array(data[0], dtype=float)
            self._y = np.array(data[1], dtype=float)
            if len(data) > 2:
                self._y_err = np.array(data[2], dtype=float)
                self.weighted = True

            if len(data) > 3:
                self._x_err = np.array(data[3], dtype=float)
        if mask is not None:
            self._mask = np.broadcast_to(mask, self._y.shape)
        elif not hasattr(self, "_mask"):
            self._mask = None #set mask to none if not copied or provided.

    def __len__(self):
        """
        the number of unmasked points in the dataset.

        """
        return self.y.size

    def __str__(self):
        return "<{0}>, {1} points".format(self.name, len(self))

    def __repr__(self):
        msk = self._mask
        if np.all(self._mask):
            msk = None

        d = {"filename": self.filename, "msk": msk, "data": self.data}
        if self.filename is not None:
            return "Data1D(data={filename!r}," " mask={msk!r})".format(**d)
        else:
            return "Data1D(data={data!r}," " mask={msk!r})".format(**d)

    @property
    def x(self):
        """
        np.ndarray : x data (possibly masked)
        """
        if self._mask is not None and self._x.size:
            return self._x[self.mask]
        return self._x

    @property
    def y(self):
        """
        np.ndarray : y data (possibly masked)
        """
        if self._mask is not None and self._y.size:
            return self._y[self.mask]
        return self._y

    @property
    def x_err(self):
        """
        np.ndarray : x uncertainty (possibly masked)
        """
        if (
            self._x_err is not None
            and self._mask is not None
            and self._x_err.size
        ):
            return self._x_err[self.mask]
        return self._x_err

    @x_err.setter
    def x_err(self, x_err):
        """
        np.ndarray : y uncertainty (possibly masked)
        """
        self._x_err = x_err

    @property
    def y_err(self):
        """
        uncertainties on the y data (possibly masked)
        """
        if (
            self._y_err is not None
            and self._mask is not None
            and self._y_err.size
        ):
            return self._y_err[self.mask]
        return self._y_err

    @property
    def mask(self):
        """
        mask
        """
        if self._mask is None:
            self._mask = np.full_like(self._y, True, dtype=bool)

        return self._mask

    @mask.setter
    def mask(self, mask):
        """
        mask
        """
        if mask is None:
            mask = True

        self._mask = np.broadcast_to(mask, self._y.shape).astype(bool)

    @property
    def data(self):
        """
        4-tuple containing the (`x`, `y`, `y_err`, `x_err`) data

        """
        return self.x, self.y, self.y_err, self.x_err

    @property
    def unmasked_data(self):
        """
        4-tuple containing unmasked (x, y, y_err, x_err) data

        """
        return self._x, self._y, self._y_err, self._x_err

    @property
    def finite_data(self):
        """
        4-tuple containing the (`x`, `y`, `y_err`, `x_err`) datapoints that are
        finite.

        """
        finite_loc = np.where(np.isfinite(self.y))
        return (
            self.x[finite_loc],
            self.y[finite_loc],
            self.y_err[finite_loc],
            self.x_err[finite_loc],
        )

    @data.setter
    def data(self, data_tuple):
        """
        Set the data for this object from supplied data.

        Parameters
        ----------
        data_tuple : tuple
            2 to 4 member tuple containing the (x, y, y_err, x_err) data to
            specify the dataset. `y_err` and `x_err` are optional.

        Notes
        -----
        Clears the mask for the dataset, it will need to be reapplied.

        """
        self._x = np.array(data_tuple[0], dtype=float)
        self._y = np.array(data_tuple[1], dtype=float)
        self.weighted = False
        self._y_err = None
        self._x_err = None

        if len(data_tuple) > 2 and data_tuple[2] is not None:
            self._y_err = np.array(data_tuple[2], dtype=float)
            self.weighted = True

        if len(data_tuple) > 3 and data_tuple[3] is not None:
            self._x_err = np.array(data_tuple[3], dtype=float)

        self._mask = None
        self.sort()

    def scale(self, scalefactor=1.0):
        """
        Scales the y and y_err data by dividing by `scalefactor`.

        Parameters
        ----------
        scalefactor : float
            The scalefactor to divide by.

        """
        self._y /= scalefactor
        self._y_err /= scalefactor

    def copy(self):
        """
        Copies the dataset and parameters, including file info.
        """
        clone = Data1D(data=self) #pass Data1D Obj back to constructor to copy.
        return clone

    def add_data(self, data_tuple, requires_splice=False, trim_trailing=True):
        """
        Adds more data to the dataset.

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
        `requires_splice` was True. The added data is not masked.

        """
        x, y, y_err, x_err = self.data

        # dataset has no points, can just initialise with the tuple
        if not len(self):
            self.data = data_tuple
            return

        ax, ay = data_tuple[0:2]

        # if ((len(data_tuple) > 2 and self.y_err is None) or
        #         (len(data_tuple) == 2 and self.y_err is not None)):
        #     raise ValueError("Both the existing Data1D and the data you're"
        #                      " trying to add need to have y_err")
        #
        # if ((len(data_tuple) > 3 and self.x_err is None) or
        #         (len(data_tuple) == 3 and self.x_err is not None)):
        #     raise ValueError("Both the existing Data1D and the data you're"
        #                      " trying to add need to have x_err")

        ay_err = None
        ax_err = None

        if len(data_tuple) > 2:
            ay_err = np.array(data_tuple[2], dtype=float)

        if len(data_tuple) > 3:
            ax_err = np.array(data_tuple[3], dtype=float)

        mask2 = np.full_like(data_tuple[0], True, bool)

        # which values in the first dataset overlap with the second
        overlap_points = np.zeros_like(x, "bool")

        # go through and stitch them together.
        scale = 1.0
        dscale = 0.0
        if requires_splice and len(self) > 1:
            scale, dscale, overlap_points = get_scaling_in_overlap(
                x, y, y_err, ax, ay, ay_err
            )

            if (
                (not np.isfinite(scale))
                or (not np.isfinite(dscale))
                or (not np.size(overlap_points, 0))
            ):
                raise ValueError("No points in overlap region")

        if not trim_trailing:
            overlap_points[:] = False

        qq = np.r_[x[~overlap_points], ax]
        rr = np.r_[y[~overlap_points], ay * scale]
        overall_mask = np.r_[self.mask[~overlap_points], mask2]

        try:
            dr = np.r_[y_err[~overlap_points], ay_err * scale]
        except (TypeError, ValueError):
            if (ay_err is not None) or (y_err is not None):
                raise ValueError(
                    "Both the existing Data1D and the data you're"
                    " trying to add need to have y_err"
                )
            dr = None

        try:
            dq = np.r_[x_err[~overlap_points], ax_err]
        except (TypeError, ValueError):
            if (ax_err is not None) or (x_err is not None):
                raise ValueError(
                    "Both the existing Data1D and the data you're"
                    " trying to add need to have x_err"
                )
            dq = None

        self.data = (qq, rr, dr, dq)
        self.mask = overall_mask

        self.sort()

    def blend_data(self, other):
        """
        Similar to add_data, but instead of truncating/overlapping, data is statistically combined using weighted yerrs.
        This can be useful in combining datasets where (scaled) reflectivity at higher Q can have lower noise when measured at higher Omega.

        Parameters
        ----------
        other : Data1D
            Additional dataset. Point spacing in Q should match existing dataset, so overlapping points can be combined.
        
        Notes
        -----
        Raises `AttributeError` if different data density in overlap reigon.
        Raises `AttributeError` if neither dataset has a y_err attribute.
        Raises `ValueError` if no overlap in x; should use add_data.
        
        Returns
        -------
        Data1D: Concatenation of non-overlap and combined overlap reigons.
        """
        
        #Check for necessary y_err data:
        if self.y_err == None:
            raise AttributeError("self does not have y_err data.")
        elif other.y_err == None:
            raise AttributeError("other does not have y_err data.")
        
        # Create clone objects so data can be sorted without modifying original objects
        RD1 = self.copy()
        RD2 = other.copy()
        # Sort data, ensuring point indexing aligns.
        RD1.sort()
        RD2.sort()
        # Get Raw data
        x, y, yerr, xerr = RD1.data
        ax, ay, ayerr, axerr = RD2.data
        # Calculate overlap and scale difference
        scale, dscale, overlap_points = get_scaling_in_overlap(
            x, y, yerr, ax, ay, ayerr)
        # Raise ValueError if no overlap between datasets.
        if len(overlap_points) == 0:
            raise ValueError("No overlap between datasets. Use add_data method instead.")
        
        # Adjust new scale of second dataset
        RD2.scale(1/scale)
        bx, by, byerr, bxerr = RD2.data

        # Check overlap in points - Are Q values the same?
        # Assume not. Point spacing also goes logarithmic in x.
        Qmin = np.min(x[overlap_points])
        Qmax = np.max(x[overlap_points])
        rd1_len = np.sum(overlap_points)  # number of points overlapping
        rd2_overlap = (bx < Qmax) & (bx >= Qmin) #Use one bound with equality to ensure n-1 points if x values are identitcal.
        rd2_len = np.sum(rd2_overlap)

        # Restrict to overlap (ovlp) region for calculations
        xovlp = x[overlap_points]
        yovlp = y[overlap_points]
        yerrovlp = yerr[overlap_points]
        xerrovlp = xerr[overlap_points]
        bxovlp = bx[rd2_overlap]
        byovlp = by[rd2_overlap]
        byerrovlp = byerr[rd2_overlap]
        bxerrovlp = bxerr[rd2_overlap]

        # If spaced linearly in LogQ, points should overlap.
        # All RD2 should be within the RD1 domain.
        if rd2_len == rd1_len-1:
            # Calculate weightings for RD1 to closest relevant x (Q) value
            xdif1 = np.abs(bxovlp - xovlp[:-1])  # proximity to x0
            xdif2 = np.abs(bxovlp - xovlp[1:])  # proximity to x1

            # Calculate new interpolation of RD1 based on x-deltas to RD2 x values.
            # val = val1 * prox_to_val2 + val2*prox_to_val1 / (prox_to_val1 + prox_to_val2)
            x_new = (xdif2 * xovlp[:-1] + xdif1 * xovlp[1:]) / (xdif1 + xdif2)
            y_new = (xdif2 * yovlp[:-1] + xdif1 * yovlp[1:]) / (xdif1 + xdif2)
            # f=ax + by / (a+b), Unc(f) = 1/abs(a+b) * sqrt((a uncx)^2 + (b uncy)^2)
            xerr_new = np.sqrt(
                (xdif2 * xerrovlp[:-1])**2 + (xdif1 * xerrovlp[1:])**2) / np.abs(xdif1 + xdif2)
            yerr_new = np.sqrt(
                (xdif2 * yerrovlp[:-1])**2 + (xdif1 * yerrovlp[1:])**2) / np.abs(xdif1 + xdif2)

            # Calculate new combination of RD1 and RD2 based on yerr weightings.
            # val = val1 * yerr2 + val2 * yerr1 / (yerr1 + yerr2) --> value of smaller error dominates
            yerr_tot = yerr_new + byerrovlp
            wx = (x_new * byerrovlp + bxovlp * yerr_new) / yerr_tot
            wy = (y_new * byerrovlp + byovlp * yerr_new) / yerr_tot
            # f=(a*x + b*y) / (a+b), Unc(f) = 1/abs(a+b) * sqrt((a*uncx)^2 + (b*uncy)^2)
            wxerr = np.sqrt((xerr_new * byerrovlp)**2 +
                            (bxerrovlp * yerr_new)**2) / yerr_tot
            wyerr = np.sqrt((yerr_new * byerrovlp)**2 +
                            (byerrovlp * yerr_new)**2) / yerr_tot
            RDoverlap_contrib_data = (wx, wy, wyerr, wxerr)

            # Generate new dataset
            RD1_points = ~overlap_points
            RD2_points = ~rd2_overlap
            RD1_contrib_data = (x[RD1_points], y[RD1_points],
                                yerr[RD1_points], xerr[RD1_points])
            RD2_contrib_data = (bx[RD2_points], by[RD2_points],
                                byerr[RD2_points], bxerr[RD2_points])
            RDconcat = Data1D(data=RD1_contrib_data)
            RDconcat.add_data(RDoverlap_contrib_data)
            RDconcat.add_data(RD2_contrib_data)
            return RDconcat
        else:
            raise AttributeError(
                "Overlap points have different data density in Log(Q); Self:"+str(rd1_len) + ", Other:" + str(rd2_len) + " between Q=("+str(Qmin)+","+str(Qmax)+")")
    
    def sort(self):
        """
        Sorts the data in ascending order
        """
        sorted = np.argsort(self.x)
        self._x = self.x[sorted]
        self._y = self.y[sorted]

        if self._mask is not None:
            self._mask = self._mask[sorted]
        if self.y_err is not None:
            self._y_err = self.y_err[sorted]
        if self.x_err is not None:
            self._x_err = self.x_err[sorted]

    def save(self, f, header=None):
        """
        Saves the data to file. Saves the data as 4 column ASCII.

        Parameters
        ----------
        f : {file-like, str, Path}
            File to save the dataset to.

        """
        if header is not None:
            _header = header
        elif len(self.metadata):
            _header = str(self.metadata)
        else:
            _header = ""

        np.savetxt(
            f,
            np.column_stack((self._x, self._y, self._y_err, self._x_err)),
            header=_header,
        )

    def load(self, f):
        """
        Loads a dataset from file, and overwrites existing data.
        Must be 2 to 4 column ASCII.

        Parameters
        ----------
        f : {file-like, string, Path}
            File to load the dataset from.

        """
        # it would be nicer to simply use np.loadtxt, but this is an
        # attempt to auto ignore header lines.
        with possibly_open_file(f, "r") as g:
            lines = list(reversed(g.readlines()))
            x = list()
            y = list()
            y_err = list()
            x_err = list()

            # a marker for how many columns in the data there will be
            numcols = 0
            for i, line in enumerate(lines):
                try:
                    # parse a line for numerical tokens separated by whitespace
                    # or comma
                    nums = [
                        float(tok)
                        for tok in re.split(r"\s|,", line)
                        if len(tok)
                    ]
                    if len(nums) in [0, 1]:
                        # might be trailing newlines at the end of the file,
                        # just ignore those
                        continue
                    if not numcols:
                        # figure out how many columns one has
                        numcols = len(nums)
                    elif len(nums) != numcols:
                        # if the number of columns changes there's an issue
                        break
                    x.append(nums[0])
                    y.append(nums[1])
                    if len(nums) > 2:
                        y_err.append(nums[2])
                    if len(nums) > 3:
                        x_err.append(nums[3])
                except ValueError:
                    # you should drop into this if you can't parse tokens into
                    # a series of floats. But the text may be meta-data, so
                    # try to carry on.
                    continue

        x.reverse()
        y.reverse()
        y_err.reverse()
        x_err.reverse()

        if len(x) == 0:
            raise RuntimeError(
                "Datafile didn't appear to contain any data (or"
                " was the wrong format)"
            )

        if numcols < 3:
            y_err = None
        if numcols < 4:
            x_err = None

        self.data = (x, y, y_err, x_err)

        if hasattr(f, "read"):
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

    def plot(self, fig=None):
        """
        Plot the dataset.

        Requires matplotlib be installed.

        Parameters
        ----------
        fig: Figure instance, optional
            If `fig` is not supplied then a new figure is created. Otherwise
            the graph is created on the current axes on the supplied figure.

        Returns
        -------
        fig, ax : :class:`matplotlib.figure.Figure`, :class:`matplotlib.Axes`
            `matplotlib` figure and axes objects.

        """
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = fig.gca()

        if self.y_err is not None:
            ax.errorbar(self.x, self.y, self.y_err, label=self.name)
        else:
            ax.scatter(self.x, self.y, label=self.name)

        return fig, ax

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

    def synthesise(self, random_state=None):
        """
        Synthesise a new dataset by adding Gaussian noise onto each of the
        datapoints of the existing data.

        Returns
        -------
        dataset : :class:`Data1D`
            A new synthesised dataset
        random_state : {int, :class:`numpy.random.RandomState`, :class:`numpy.random.Generator`}
            If `random_state` is not specified the
            :class:`numpy.random.RandomState` singleton is used.
            If `random_state` is an int, a new ``RandomState`` instance is
            used, seeded with random_state.
            If `random_state` is already a ``RandomState`` or a ``Generator``
            instance, then that object is used.
            Specify `random_state` for repeatable synthesising.
        """
        if self._y_err is None:
            raise RuntimeError(
                "Can't synthesise new dataset without y_err" "uncertainties"
            )

        rng = check_random_state(random_state)
        gnoise = rng.standard_normal(size=self._y_err.shape)

        new_y = self._y + gnoise * self._y_err
        data = list(self.data)
        data.pop(1)
        data.insert(1, new_y)

        dataset = Data1D()
        dataset.data = data

        return dataset
