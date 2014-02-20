# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SLDcalculator.ui'
#
# Created: Tue Feb 18 14:22:24 2014
#      by: pyside-uic 0.2.15 running on PySide 1.2.1
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui


class Ui_Form(object):

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.setWindowModality(QtCore.Qt.NonModal)
        Form.resize(355, 266)
        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.chemical_formula = QtGui.QLineEdit(Form)
        self.chemical_formula.setText("")
        self.chemical_formula.setObjectName("chemical_formula")
        self.gridLayout.addWidget(self.chemical_formula, 0, 0, 1, 4)
        self.use_volume = QtGui.QRadioButton(Form)
        self.use_volume.setObjectName("use_volume")
        self.gridLayout.addWidget(self.use_volume, 1, 3, 1, 1)
        self.label_5 = QtGui.QLabel(Form)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 0, 1, 3)
        self.label_6 = QtGui.QLabel(Form)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 3, 1, 1)
        self.mass_density = QtGui.QDoubleSpinBox(Form)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.mass_density.sizePolicy().hasHeightForWidth())
        self.mass_density.setSizePolicy(sizePolicy)
        self.mass_density.setDecimals(3)
        self.mass_density.setSingleStep(0.001)
        self.mass_density.setProperty("value", 1.107)
        self.mass_density.setObjectName("mass_density")
        self.gridLayout.addWidget(self.mass_density, 3, 0, 1, 3)
        self.molecular_volume = QtGui.QDoubleSpinBox(Form)
        sizePolicy = QtGui.QSizePolicy(
            QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.molecular_volume.sizePolicy().hasHeightForWidth())
        self.molecular_volume.setSizePolicy(sizePolicy)
        self.molecular_volume.setDecimals(3)
        self.molecular_volume.setMaximum(1000000000.0)
        self.molecular_volume.setSingleStep(0.01)
        self.molecular_volume.setProperty("value", 1.0)
        self.molecular_volume.setObjectName("molecular_volume")
        self.gridLayout.addWidget(self.molecular_volume, 3, 3, 1, 1)
        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 3)
        self.neutron_wavelength = QtGui.QDoubleSpinBox(Form)
        self.neutron_wavelength.setMaximum(20.0)
        self.neutron_wavelength.setSingleStep(0.01)
        self.neutron_wavelength.setProperty("value", 1.8)
        self.neutron_wavelength.setObjectName("neutron_wavelength")
        self.gridLayout.addWidget(self.neutron_wavelength, 4, 3, 1, 1)
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 5, 0, 1, 2)
        self.xray_energy = QtGui.QDoubleSpinBox(Form)
        self.xray_energy.setDecimals(3)
        self.xray_energy.setSingleStep(0.001)
        self.xray_energy.setProperty("value", 8.048)
        self.xray_energy.setObjectName("xray_energy")
        self.gridLayout.addWidget(self.xray_energy, 5, 3, 1, 1)
        self.label_2 = QtGui.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 7, 0, 1, 1)
        self.xray_SLD = QtGui.QLineEdit(Form)
        self.xray_SLD.setReadOnly(True)
        self.xray_SLD.setObjectName("xray_SLD")
        self.gridLayout.addWidget(self.xray_SLD, 7, 1, 1, 3)
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 6, 0, 1, 1)
        self.neutron_SLD = QtGui.QLineEdit(Form)
        self.neutron_SLD.setReadOnly(True)
        self.neutron_SLD.setObjectName("neutron_SLD")
        self.gridLayout.addWidget(self.neutron_SLD, 6, 1, 1, 3)
        self.use_density = QtGui.QRadioButton(Form)
        self.use_density.setChecked(True)
        self.use_density.setObjectName("use_density")
        self.gridLayout.addWidget(self.use_density, 1, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.chemical_formula, self.use_volume)
        Form.setTabOrder(self.use_volume, self.mass_density)
        Form.setTabOrder(self.mass_density, self.molecular_volume)
        Form.setTabOrder(self.molecular_volume, self.neutron_wavelength)
        Form.setTabOrder(self.neutron_wavelength, self.xray_energy)
        Form.setTabOrder(self.xray_energy, self.neutron_SLD)
        Form.setTabOrder(self.neutron_SLD, self.xray_SLD)

    def retranslateUi(self, Form):
        Form.setWindowTitle(
            QtGui.QApplication.translate(
                "Form",
                "SLD calculator",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.chemical_formula.setPlaceholderText(
            QtGui.QApplication.translate(
                "Form",
                "H[2]2O",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.use_volume.setText(
            QtGui.QApplication.translate(
                "Form",
                "molecular volume",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(
            QtGui.QApplication.translate(
                "Form",
                "Mass density (g/cc)",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(
            QtGui.QApplication.translate(
                "Form",
                "Molecular volume (Å**3)",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(
            QtGui.QApplication.translate(
                "Form",
                "Neutron Wavelength (Å)",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(
            QtGui.QApplication.translate(
                "Form",
                "Xray Energy (keV)",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(
            QtGui.QApplication.translate(
                "Form",
                "Xray SLD  ( / 1e-6 Å**-2)",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.label.setText(
            QtGui.QApplication.translate(
                "Form",
                "Neutron SLD ( / 1e-6 Å**-2)",
                None,
                QtGui.QApplication.UnicodeUTF8))
        self.use_density.setText(
            QtGui.QApplication.translate(
                "Form",
                "density",
                None,
                QtGui.QApplication.UnicodeUTF8))
