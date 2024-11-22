import string
import time
import re

# from datetime import datetime
from pathlib import Path, PurePath

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np

from refnx.dataset import Data1D
from refnx._lib import possibly_open_file


def load_orso(f):
    from orsopy.fileio import load_orso as _load_orso
    from orsopy.fileio import load_nexus

    try:
        return _load_orso(f)
    except Exception:
        pass
    try:
        v = load_nexus(f)
        return v
    except Exception:
        pass


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
<R uncertainty="dR">$_ydata</R>
<Qz uncertainty="dQz" units="1/A">$_xdata</Qz>
<dR type="SD">$_ydataSD</dR>
<dQz type="_FWHM" units="1/A">$_xdataSD</dQz>
</REFdata>
</REFentry>
</REFroot>"""


class ReflectDataset(Data1D):
    """
    A 1D Reflectivity dataset.
    """

    def __init__(self, data=None, **kwds):
        """
        Initialise a reflectivity dataset.

        Parameters
        ----------
        data : {str, file-like, Path, tuple of np.ndarray} optional
            `data` can be a string, file-like, or Path object referring to a File to
            load the dataset from.

            Alternatively it is a tuple containing the data from which the
            dataset will be constructed. The tuple should have between 2 and 4
            members.

                - data[0] - Q
                - data[1] - R
                - data[2] - dR
                - data[3] - dQ

            `data` must be at least two long, `Q` and `R`.
            If the tuple is at least 3 long then the third member is `dR`.
            If the tuple is 4 long then the fourth member is `dQ`.
            All arrays must have the same shape.
        """
        super().__init__(data=data, **kwds)
        self.datafilenumber = list()
        self.sld_profile = None

    def __repr__(self):
        msk = self._mask
        if np.all(self._mask):
            msk = None

        if self.filename is not None:
            return f"ReflectDataset(data={str(self.filename)!r}, mask={msk!r})"
        else:
            return f"ReflectDataset(data={self.data!r}, mask={msk!r})"

    def save_xml(self, f, start_time=0):
        """
        Saves the reflectivity data to an XML file.

        Parameters
        ----------
        f : str or file-like
            The file to write the spectrum to, or a str that specifies the file
            name
        start_time: int, optional
            Epoch time specifying when the sample started
        """
        s = string.Template(_template_ref_xml)
        self.time = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(start_time)
        )
        # self.time = time.strftime(
        # datetime.fromtimestamp(start_time).isoformat()
        # filename = 'c_PLP{:07d}_{:d}.xml'.format(self._rnumber[0], 0)

        self._ydata = repr(self.y.tolist()).strip(",[]")
        self._xdata = repr(self.x.tolist()).strip(",[]")
        self._ydataSD = repr(self.y_err.tolist()).strip(",[]")
        self._xdataSD = repr(self.x_err.tolist()).strip(",[]")

        thefile = s.safe_substitute(self.__dict__)

        with possibly_open_file(f, "wb") as g:
            if "b" in g.mode:
                thefile = thefile.encode("utf-8")

            g.write(thefile)

    def load(self, f):
        """
        Load a dataset from file. Can either be 2-4 column ascii or XML file.

        Parameters
        ----------
        f : {str, file-like, Path}
            The file to load the spectrum from, or a str that specifies the
            file name
        """
        if hasattr(f, "read") and hasattr(f, "write"):
            if hasattr(f, "name"):
                # file-like ?
                fname = f.name
            else:
                fname = ""
        else:
            fname = f
        try:
            tree = ET.ElementTree()
            tree.parse(f)

            delim = ", | |,"
            qtext = re.split(delim, tree.find(".//Qz").text)
            rtext = re.split(delim, tree.find(".//R").text)
            drtext = re.split(delim, tree.find(".//dR").text)
            dqtext = re.split(delim, tree.find(".//dQz").text)

            qvals = [float(val) for val in qtext if len(val)]
            rvals = [float(val) for val in rtext if len(val)]
            drvals = [float(val) for val in drtext if len(val)]
            dqvals = [float(val) for val in dqtext if len(val)]

            if isinstance(fname, PurePath):
                # use a PurePath, not a system specific path type
                # because Posix systems can't deal with WindowsPath
                # and vice versa. This becomes an issue when pickling.
                fname = PurePath(fname)

            self.filename = fname
            self.name = Path(fname).stem
            self.data = (qvals, rvals, drvals, dqvals)
        except ET.ParseError:
            super().load(fname)


class OrsoDataset(Data1D):
    """
    A thinly wrapped version of an ORSODataset

    Parameters
    ----------
    data : {str, file-like. Path}

    Notes
    -----
    Multiplies the resolution information contained in the fourth column
    of the ORSO dataset to convert from standard deviation to FWHM.
    """

    def __init__(self, data, **kwds):
        super().__init__(data=data, **kwds)
        self.orso = None

    def load(self, f):
        """
        Parameters
        ----------
        f : {str, file-like, Path}
            The file to load the spectrum from, or a str/Path that specifies
            the file name
        """
        if hasattr(f, "read") and hasattr(f, "write"):
            if hasattr(f, "name"):
                # file-like ?
                fname = f.name
            else:
                fname = ""
        else:
            # string-like??
            fname = str(f)

        mode = "r"
        if fname.endswith(".orb"):
            mode = "rb"

        with possibly_open_file(f, mode) as g:
            self.orso = load_orso(g)
            header = self.orso[0].info

        _data = self.orso[0].data[:, :4].T

        # figure out if data was in 1/nm or 1/angstrom
        # internally refnx uses reciprocal angstrom
        q_units = header.columns[0].unit.lower()
        if q_units == "1/nm":
            # need to divide q by 10
            _data[0] /= 10.0

            if _data.shape[0] > 3:
                _data[3] /= 10.0

        # ORSO files save resolution information as SD,
        # internally refnx uses FWHM
        if _data.shape[0] > 3:
            _data[3] *= 2.3548

        self.data = _data
        self.filename = fname
        self.name = Path(fname).stem

    def refresh(self):
        """
        Refreshes a previously loaded dataset.

        """
        # OrsoDataset needs to carry its own implementation
        # opening a binary file needs to be done with correct mode
        # vs opening a text file.
        if self.filename is not None:
            self.load(self.filename)
