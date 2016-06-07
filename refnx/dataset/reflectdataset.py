from __future__ import division
import string
from datetime import datetime
import re

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import os.path
from refnx.dataset import Data1D


class ReflectDataset(Data1D):
    """
    A 1D Reflectivity dataset.
    """
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

    def __init__(self, data=None, **kwds):
        """
        Initialise a reflectivity dataset.

        Parameters
        ----------
        data : tuple of np.ndarray
            Specify the Q, R, dR, dQ data to construct the dataset from.
        """
        super(ReflectDataset, self).__init__(data=data)
        self.datafilenumber = list()
        self.sld_profile = None

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
        s = string.Template(self._template_ref_xml)
        self.time = datetime.fromtimestamp(start_time).isoformat()
        # filename = 'c_PLP{:07d}_{:d}.xml'.format(self._rnumber[0], 0)

        self._ydata = repr(self.y.tolist()).strip(',[]')
        self._xdata = repr(self.x.tolist()).strip(',[]')
        self._ydataSD = repr(self.y_err.tolist()).strip(',[]')
        self._xdataSD = repr(self.x_err.tolist()).strip(',[]')

        thefile = s.safe_substitute(self.__dict__)

        auto_fh = None
        g = f
        if not hasattr(f, 'write'):
            auto_fh = open(f, 'wb')
            g = auto_fh

        if 'b' in g.mode:
            thefile = thefile.encode('utf-8')

        g.write(thefile)

        if auto_fh is not None:
            auto_fh.close()

    def load(self, f):
        """
        Load a dataset from file. Can either be 2-4 column ascii or XML file.

        Parameters
        ----------
        f : str or file-like
            The file to load the spectrum from, or a str that specifies the file
            name
        """
        auto_fh = None
        g = f
        if not hasattr(f, 'read'):
            auto_fh = open(f, 'rb')
            g = auto_fh
        try:
            tree = ET.ElementTree()
            tree.parse(g)

            delim = ', | |,'
            qtext = re.split(delim, tree.find('.//Qz').text)
            rtext = re.split(delim, tree.find('.//R').text)
            drtext = re.split(delim, tree.find('.//dR').text)
            dqtext = re.split(delim, tree.find('.//dQz').text)

            qvals = [float(val) for val in qtext if len(val)]
            rvals = [float(val) for val in rtext if len(val)]
            drvals = [float(val) for val in drtext if len(val)]
            dqvals = [float(val) for val in dqtext if len(val)]

            self.filename = g.name
            self.name = os.path.splitext(os.path.basename(g.name))[0]
            self.data = (qvals, rvals, drvals, dqvals)
            self.filename = g.name
        except ET.ParseError:
            g.seek(0)
            super(ReflectDataset, self).load(g)
        finally:
            if auto_fh is not None:
                auto_fh.close()
