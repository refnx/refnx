from PySide import QtCore, QtGui
import SLDcalculatorUI


class SLDcalculatorView(QtGui.QDialog):
    def __init__(self, parent=None):
        super(SLDcalculatorView, self).__init__(parent)
        SLDcalculatorGUI = SLDcalculatorUI.Ui_Form()
        SLDcalculatorGUI.setupUi(self)