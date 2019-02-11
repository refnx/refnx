import os.path
import json

from PyQt5 import QtCore, QtGui, QtWidgets, uic

import periodictable as pt
from refnx.reflect import Spline, SLD


pth = os.path.dirname(os.path.abspath(__file__))
UI_LOCATION = os.path.join(pth, 'ui')
SplineDialogUI = uic.loadUiType(os.path.join(UI_LOCATION,
                                          'spline.ui'))[0]


class SplineDialog(QtWidgets.QDialog, SplineDialogUI):
    def __init__(self, parent=None):
        # persistent lipid leaflet dlg
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

    def component(self):
        # return a SplineComponent
        pass