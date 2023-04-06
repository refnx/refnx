import os.path
from qtpy import QtWidgets, uic
from qtpy.QtCore import Slot
import periodictable as pt
import pyparsing
import numpy as np


UI_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


class SLDcalculatorView(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = uic.loadUi(
            os.path.join(UI_LOCATION, "SLDcalculator.ui"), self
        )
        self.last_formula = "H[2]2O"

    def calculate(self):
        try:
            formula = pt.formula(self.ui.chemical_formula.text())

        except (pyparsing.ParseException, TypeError, ValueError, KeyError):
            # a problem was experience during parsing of the formula.
            return

        if not len(formula.atoms):
            return

        density = self.ui.mass_density.value()
        molecular_volume = self.ui.molecular_volume.value()
        neutron_wavelength = self.ui.neutron_wavelength.value()
        xray_energy = self.ui.xray_energy.value()

        if self.ui.use_volume.isChecked():
            density = formula.molecular_mass / formula.volume(
                a=molecular_volume, b=1, c=1
            )
            self.ui.mass_density.setValue(density)

        elif self.ui.use_density.isChecked():
            try:
                volume = (
                    formula.mass
                    / density
                    / pt.constants.avogadro_number
                    * 1e24
                )
            except ZeroDivisionError:
                volume = np.nan
            self.ui.molecular_volume.setValue(volume)

        try:
            real, imag, mu = pt.neutron_sld(
                formula, density=density, wavelength=neutron_wavelength
            )

            self.ui.neutron_SLD.setText(
                "%.6g" % real + " + " + "%.6g" % imag + "j"
            )
        except Exception:
            self.ui.neutron_SLD.setText("NaN")

        try:
            real, imag = pt.xray_sld(
                formula, density=density, energy=xray_energy
            )

            self.ui.xray_SLD.setText(
                "%.6g" % real + " + " + "%.6g" % imag + "j"
            )
        except Exception:
            self.ui.xray_SLD.setText("NaN")
            # sometimes the X-ray and neutron SLD calc can fail, if there are
            # no scattering factors

    @Slot(float)
    def on_mass_density_valueChanged(self, arg_1):
        self.calculate()

    @Slot(float)
    def on_molecular_volume_valueChanged(self, arg_1):
        self.calculate()

    @Slot(float)
    def on_neutron_wavelength_valueChanged(self, arg_1):
        self.calculate()

    @Slot(float)
    def on_xray_energy_valueChanged(self, arg_1):
        self.calculate()

    @Slot(str)
    def on_chemical_formula_textChanged(self, arg_1):
        self.calculate()
