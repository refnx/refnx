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
        Metadata attribute is shallow copied.
        """
        myData = self.data #tuple, returns witih None values.
        copyData = [] 
        for a in myData:  # Copy arrays, and exclude None values.
            if not a is None:
                copyData.append(a.copy())
        clone = Data1D(data=tuple(copyData))
        clone.name = self.name
        clone.filename = self.filename
        clone.metadata = self.metadata
        clone.weighted = self.weighted
        clone._mask = None if self._mask is None else self._mask.copy()
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
            Additional dataset. Point spacing in Q can be different.
        
        Notes
        -----
        Overlap points are calculated for each dataset.
        The shorter overlap dataset is interpolated to match the larger overlap dataset.
        Then a weighted average of dataset values are calculated, favoring smaller y-error values.
        Errors are also propogated throughout calculations.
        
        Raises `AttributeError` if neither dataset has a y_err attribute.
        Raises `ValueError` if no overlap in x; should use add_data.
        
        Returns
        -------
        RDconcat: :class:`refnx.dataset.data1d.Data1D` New dataset with concatenation of non-overlap and combined overlap reigons.
        """
        
        #Check for necessary y_err data:
        if self.y_err is None:
            raise AttributeError("self does not have y_err data.")
        elif other.y_err is None:
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
            x, y, yerr, ax, ay, ayerr)  # note, this only finds values ax < x[-1], garanteeing x[-1] is a limit value.

        # Raise ValueError if no overlap between datasets.
        if len(overlap_points) == 0:
            raise ValueError(
                "No overlap between datasets. Use add_data method instead.")

        # Adjust new scale of second dataset
        RD2.scale(1/scale)
        bx, by, byerr, bxerr = RD2.data

        # Check overlap in points:
        #   Assume Q values are not the same,
        #   and perhaps density is different between data..
        xovlp = x[overlap_points]  # 1st obj restricted domain
        Qmin = np.min(xovlp)
        Qmax = x[-1]  # largest value.
        rd1_len = np.sum(overlap_points)  # number of points overlapping
        # Use one bound with equality to ensure n-1 points if x values are identitcal.
        rd2_overlap = (bx < Qmax) & (bx >= Qmin)
        rd2_len = np.sum(rd2_overlap)

        # Restrict objs to overlap (ovlp) region for calculations.
        yovlp = y[overlap_points]
        yerrovlp = yerr[overlap_points]
        xerrovlp = None if xerr is None else xerr[overlap_points]
        bxovlp = bx[rd2_overlap]
        byovlp = by[rd2_overlap]
        byerrovlp = byerr[rd2_overlap]
        bxerrovlp = None if bxerr is None else bxerr[rd2_overlap]

        # Instead of assuming exact point overlap, and same number of points (likely)
        # Use np.interp to interpolate any extra points not covered.
        # Check which array is smaller,
        if rd2_len >= rd1_len:
            # RD1 needs extending, or is the same length.
            s_x = xovlp  # smaller x
            s_y = yovlp
            s_xerr = xerrovlp
            s_yerr = yerrovlp
            l_x = bxovlp
            l_y = byovlp
            l_xerr = bxerrovlp
            l_yerr = byerrovlp
        elif rd2_len < rd1_len:
            # RD2 needs extending
            s_x = bxovlp  # smaller x
            s_y = byovlp
            s_xerr = bxerrovlp
            s_yerr = byerrovlp
            l_x = xovlp
            l_y = yovlp
            l_xerr = xerrovlp
            l_yerr = yerrovlp
        # then interpolate small values (s_y) onto l_x domain.
        si_x = l_x  # small-interpolated x now matches longer x.
        # si_y = np.interp(l_x, s_x, s_y) #extend s_y

        # calculate updated error values for interpolation function:
        # f=ax + by / (a+b),
        # Unc(f) = 1/abs(a+b) * sqrt((a uncx)^2 + (b uncy)^2)
        si_y = np.zeros(si_x.shape)
        si_xerr = np.zeros(si_x.shape) if not s_xerr is None else None
        si_yerr = np.zeros(si_x.shape)
        # This calc to be done for each interpolated index,
        # with quadratic factors,
        # so no point using np.interp.
        for i in range(len(si_x)):
            x0 = si_x[i]  # get the target interpolated x

            # find spacing to each x in uninterpolated list
            diff = np.abs(s_x - x0)  # find abs spacing from x0 value for all s_x.
            j = np.argmin(diff)
            # check where index lies relative in x_s
            if j == 0 and x0 <= s_x[0]:  # interpolation at/beyond left edge.
                # Set interpolation value to left edge value.
                si_y[i] = s_y[0]
                si_yerr[i] = s_yerr[0]
                if not si_xerr is None:
                    si_xerr[i] = s_xerr[0]
            # interpolation at/beyond right edge.
            elif j == len(s_x)-1 and x0 >= s_x[j]:
                # Set interpolation value to right edge value.
                si_y[i] = s_y[j]
                si_yerr[i] = s_yerr[j]
                if not si_xerr is None:
                    si_xerr[i] = s_xerr[j]
            else:
                # Interpolation between two points, calculate the deltas
                if j == 0:  # left limit edge
                    dj = 1
                elif j == len(s_x)-1:  # right limit edge
                    dj = -1
                else:  # inbetween data.
                    dj = 1 if diff[j+1] < diff[j-1] else -1
                dx1 = diff[j]
                dx2 = diff[j + dj]
                # f=(ax + by) / (a+b)
                si_y[i] = ((dx2 * s_y[j]) + (dx1 * s_y[j+dj])) / (dx1 + dx2)
                # Unc(f) = sqrt((a uncx)^2 + (b uncy)^2) / (abs(a)+abs(b)))
                si_yerr[i] = np.sqrt((dx2 * s_yerr[j])**2 +
                                    (dx1 * s_yerr[j + dj])**2) / (dx1 + dx2)
                if not si_xerr is None:
                    si_xerr[i] = np.sqrt((dx2 * s_xerr[j])**2 +
                                        (dx1 * s_xerr[j + dj])**2) / (dx1 + dx2)

        # Now that indexes are matched, peform calculations to blend datasets.
        # Use yerr weights to mix values at same x value.
        # val = val1 * yerr2 + val2 * yerr1 / (yerr1 + yerr2) --> value of smaller error dominates
        w_x = si_x
        yerr_tot = l_yerr + si_yerr  # sum of yerrs in overlap reigon
        w_y = (l_y * si_yerr + si_y * l_yerr) / (yerr_tot)
        # Just as before, uncertainty goes as:
        # f=(a*x + b*y) / (a+b), Unc(f) = 1/abs(a+b) * sqrt((a*uncx)^2 + (b*uncy)^2)
        w_yerr = np.sqrt((2*(l_yerr * si_yerr)**2)) / yerr_tot
        if not si_xerr is None:
            w_xerr = np.sqrt(
                ((si_xerr * l_yerr)**2 + (l_xerr * si_yerr)**2)) / yerr_tot

        # Generate new dataset
        RD1_points = ~overlap_points
        RD2_points = ~rd2_overlap
        if not si_xerr is None:
            RDoverlap_contrib_data = (w_x, w_y, w_yerr, w_xerr)
            RD1_contrib_data = (x[RD1_points], y[RD1_points],
                                yerr[RD1_points], xerr[RD1_points])
            RD2_contrib_data = (bx[RD2_points], by[RD2_points],
                                byerr[RD2_points], bxerr[RD2_points])
        else:
            RDoverlap_contrib_data = (w_x, w_y, w_yerr)
            RD1_contrib_data = (x[RD1_points], y[RD1_points], yerr[RD1_points])
            RD2_contrib_data = (bx[RD2_points], by[RD2_points], byerr[RD2_points])

        RDconcat = type(self)(data=RD1_contrib_data)
        RDconcat.add_data(RDoverlap_contrib_data)
        RDconcat.add_data(RD2_contrib_data)
        return RDconcat
    
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
