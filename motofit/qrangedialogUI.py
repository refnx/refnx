# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'qrangedialog.ui'
#
# Created: Thu Feb 27 16:49:51 2014
#      by: pyside-uic 0.2.15 running on PySide 1.2.1
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_qrange(object):
    def setupUi(self, qrange):
        qrange.setObjectName("qrange")
        qrange.setWindowModality(QtCore.Qt.ApplicationModal)
        qrange.resize(295, 187)
        qrange.setModal(True)
        self.gridLayout = QtGui.QGridLayout(qrange)
        self.gridLayout.setObjectName("gridLayout")
        self.buttonBox = QtGui.QDialogButtonBox(qrange)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 4, 1, 1, 1)
        self.numpnts = QtGui.QSpinBox(qrange)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.numpnts.sizePolicy().hasHeightForWidth())
        self.numpnts.setSizePolicy(sizePolicy)
        self.numpnts.setWrapping(False)
        self.numpnts.setFrame(True)
        self.numpnts.setSpecialValueText("")
        self.numpnts.setMinimum(1)
        self.numpnts.setMaximum(1000000000)
        self.numpnts.setProperty("value", 1000)
        self.numpnts.setObjectName("numpnts")
        self.gridLayout.addWidget(self.numpnts, 2, 1, 1, 1)
        self.qminlabel = QtGui.QLabel(qrange)
        self.qminlabel.setObjectName("qminlabel")
        self.gridLayout.addWidget(self.qminlabel, 0, 0, 1, 1)
        self.qmaxlabel = QtGui.QLabel(qrange)
        self.qmaxlabel.setObjectName("qmaxlabel")
        self.gridLayout.addWidget(self.qmaxlabel, 1, 0, 1, 1)
        self.numpntslabel = QtGui.QLabel(qrange)
        self.numpntslabel.setObjectName("numpntslabel")
        self.gridLayout.addWidget(self.numpntslabel, 2, 0, 1, 1)
        self.qmin = QtGui.QLineEdit(qrange)
        self.qmin.setObjectName("qmin")
        self.gridLayout.addWidget(self.qmin, 0, 1, 1, 1)
        self.qmax = QtGui.QLineEdit(qrange)
        self.qmax.setObjectName("qmax")
        self.gridLayout.addWidget(self.qmax, 1, 1, 1, 1)

        self.retranslateUi(qrange)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("accepted()"), qrange.accept)
        QtCore.QObject.connect(self.buttonBox, QtCore.SIGNAL("rejected()"), qrange.reject)
        QtCore.QMetaObject.connectSlotsByName(qrange)
        qrange.setTabOrder(self.numpnts, self.buttonBox)

    def retranslateUi(self, qrange):
        qrange.setWindowTitle(QtGui.QApplication.translate("qrange", "Set Q range", None, QtGui.QApplication.UnicodeUTF8))
        self.qminlabel.setText(QtGui.QApplication.translate("qrange", "Q min", None, QtGui.QApplication.UnicodeUTF8))
        self.qmaxlabel.setText(QtGui.QApplication.translate("qrange", "Q max", None, QtGui.QApplication.UnicodeUTF8))
        self.numpntslabel.setText(QtGui.QApplication.translate("qrange", "Number of points", None, QtGui.QApplication.UnicodeUTF8))
        self.qmin.setText(QtGui.QApplication.translate("qrange", "0.0000", None, QtGui.QApplication.UnicodeUTF8))
        self.qmax.setText(QtGui.QApplication.translate("qrange", "0.0000", None, QtGui.QApplication.UnicodeUTF8))

