from PySide import QtCore, QtGui
import SLDcalculatorUI
import periodictable as pt
import pyparsing


class SLDcalculatorView(QtGui.QDialog):

    def __init__(self, parent=None):
        super(SLDcalculatorView, self).__init__(parent)
        self.ui = SLDcalculatorUI.Ui_Form()
        self.ui.setupUi(self)
        self.last_formula = 'H[2]2O'

    def calculate(self):
        try:
            formula = pt.formula(self.ui.chemical_formula.text(),)

        except (pyparsing.ParseException, TypeError, ValueError):
            return

        density = self.ui.mass_density.value()
        molecular_volume = self.ui.molecular_volume.value()
        neutron_wavelength = self.ui.neutron_wavelength.value()
        xray_energy = self.ui.xray_energy.value()

        if self.ui.use_volume.isChecked():
            density = formula.molecular_mass / formula.volume(
                a=molecular_volume,
                b=1,
                c=1)
            self.ui.mass_density.setValue(density)

        elif self.ui.use_density.isChecked():
            volume = formula.mass / density / \
                pt.constants.avogadro_number * 1e24
            self.ui.molecular_volume.setValue(volume)

        real, imag, mu = pt.neutron_sld(formula,
                                        density=density,
                                        wavelength=neutron_wavelength)

        self.ui.neutron_SLD.setText(
            '%.6g' %
            real +
            ' + ' +
            '%.6g' %
            imag +
            'j')

        real, imag = pt.xray_sld(formula,
                                 density=density,
                                 energy=xray_energy)

        self.ui.xray_SLD.setText('%.6g' % real + ' + ' + '%.6g' % imag + 'j')

    @QtCore.Slot(float)
    def on_mass_density_valueChanged(self, arg_1):
        self.calculate()

    @QtCore.Slot(float)
    def on_molecular_volume_valueChanged(self, arg_1):
        self.calculate()

    @QtCore.Slot(float)
    def on_neutron_wavelength_valueChanged(self, arg_1):
        self.calculate()

    @QtCore.Slot(float)
    def on_xray_energy_valueChanged(self, arg_1):
        self.calculate()

    @QtCore.Slot(unicode)
    def on_chemical_formula_textChanged(self, arg_1):
        self.calculate()
