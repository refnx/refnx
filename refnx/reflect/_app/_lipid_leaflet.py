from pathlib import Path
import json

from qtpy import QtCore, QtGui, QtWidgets, uic
from qtpy.QtCore import Qt

import periodictable as pt
from refnx.reflect import LipidLeaflet, SLD


pth = Path(__file__).absolute().parent
UI_LOCATION = pth / "ui"


class LipidLeafletDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        # persistent lipid leaflet dlg
        QtWidgets.QDialog.__init__(self, parent)
        # load the GUI from the ui file
        self.ui = uic.loadUi(UI_LOCATION / "lipid_leaflet.ui", self)

        dvalidator = QtGui.QDoubleValidator(-2.0e-10, 5, 6)
        self.b_h_real.setValidator(dvalidator)
        self.b_h_imag.setValidator(dvalidator)
        self.b_t_real.setValidator(dvalidator)
        self.b_t_imag.setValidator(dvalidator)
        with open(pth / "lipids.json", "r") as f:
            lipids = json.load(f)

        self.lipids = {}
        for lip in lipids:
            lipid = Lipid(**lip)
            self.lipids[lipid.name] = lipid

        self._scene = None
        self.lipid_selector.addItem("")
        self.lipid_selector.addItems(self.lipids.keys())

    @QtCore.Slot(int)
    def on_lipid_selector_currentIndexChanged(self, idx):
        text = self.ui.lipid_selector.currentText()
        if text == "":
            # clear everything
            self.condition.clear()
            self.chemical_name.setText("")
            self.total_formula.setText("")
            self.references.setText("")
            if self._scene is not None:
                self._scene.clear()

        if text not in self.lipids:
            return
        self.condition.clear()
        lipid = self.lipids[text]
        conditions = list(lipid.conditions.keys())
        self.condition.addItems(conditions)

        self.chemical_name.setText(str(lipid.chemical_name))
        self.chemical_name.setCursorPosition(0)
        self.references.setText("\n".join(lipid.references))
        self.total_formula.setText(str(lipid.formula.atoms))

        V_h, V_t = lipid.conditions[conditions[0]]
        tt = self.thick_t.value()

        self.APM.setValue(V_t / tt)
        self.V_h.setValue(V_h)
        self.V_t.setValue(V_t)

        self.display_structure()
        self.calculate()

    def resizeEvent(self, event):
        self.display_structure()

    def display_structure(self):
        # display the image in the scene
        name = self.lipid_selector.currentText()
        if name not in self.lipids:
            return

        lipid = self.lipids[name]
        pixMap = QtGui.QPixmap(str(pth / "icons" / f"{lipid.name}.png"))
        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.addPixmap(pixMap)
        self.chemical_structure.setScene(self._scene)
        self.chemical_structure.show()
        self.chemical_structure.fitInView(
            self._scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio
        )
        self._scene.update()

    @QtCore.Slot(int)
    def on_condition_currentIndexChanged(self, val):
        name = self.lipid_selector.currentText()
        text = self.condition.currentText()

        if name not in self.lipids:
            return

        lipid = self.lipids[name]
        if text in lipid.conditions:
            self.V_h.setValue(lipid.conditions[text][0])
            self.V_t.setValue(lipid.conditions[text][1])
            self.calculate()

    @QtCore.Slot(int)
    def on_radiation_currentIndexChanged(self, idx):
        self.calculate()

    @QtCore.Slot(float)
    def on_xray_energy_valueChanged(self, value):
        self.calculate()

    @QtCore.Slot(float)
    def on_APM_valueChanged(self, value):
        name = self.lipid_selector.currentText()
        if name not in self.lipids:
            return
        lipid = self.lipids[name]
        condition = self.condition.currentText()
        V_h, V_t = lipid.conditions[condition]
        self.thick_h.setValue(V_h / value)
        self.thick_t.setValue(V_t / value)

    @QtCore.Slot(float)
    def on_thick_t_valueChanged(self, value):
        name = self.lipid_selector.currentText()
        if name not in self.lipids:
            return
        lipid = self.lipids[name]
        condition = self.condition.currentText()
        V_h, V_t = lipid.conditions[condition]
        APM = V_t / value
        self.APM.setValue(APM)
        self.thick_h.setValue(V_h / APM)

    @QtCore.Slot(float)
    def on_thick_h_valueChanged(self, value):
        name = self.lipid_selector.currentText()
        if name not in self.lipids:
            return
        lipid = self.lipids[name]
        condition = self.condition.currentText()
        V_h, V_t = lipid.conditions[condition]
        APM = V_h / value
        self.APM.setValue(APM)
        self.thick_t.setValue(V_t / APM)

    @QtCore.Slot(float)
    def on_head_solvent_valueChanged(self, value):
        self.calculate()

    @QtCore.Slot(float)
    def on_tail_solvent_valueChanged(self, value):
        self.calculate()

    def calculate(self):
        name = self.lipid_selector.currentText()
        if name not in self.lipids:
            return

        lipid = self.lipids[name]
        condition = self.condition.currentText()
        if condition not in lipid.conditions:
            return

        # hard coding proton exchange to be water!
        hsolv = self.head_solvent.value()
        tsolv = self.tail_solvent.value()
        phi_h = (hsolv + 0.56) / 6.92
        phi_t = (tsolv + 0.56) / 6.92

        if self.radiation.currentText() == "neutrons":
            scatlens_h = lipid.neutron_scattering_lengths(
                condition, vf_d_solvent=phi_h
            )
            scatlens_t = lipid.neutron_scattering_lengths(
                condition, vf_d_solvent=phi_t
            )
            scatlens = (scatlens_h[0], scatlens_t[1])
        else:
            energy = self.xray_energy.value()
            scatlens = lipid.xray_scattering_lengths(condition, energy=energy)

        self.b_h_real.setText(str(scatlens[0].real))
        self.b_h_imag.setText(str(scatlens[0].imag))
        self.b_t_real.setText(str(scatlens[1].real))
        self.b_t_imag.setText(str(scatlens[1].imag))

    def component(self):
        b_h_real = self.b_h_real.text()
        b_t_real = self.b_t_real.text()
        b_h_imag = self.b_h_imag.text()
        b_t_imag = self.b_t_imag.text()

        b_t = complex(float(b_t_real), float(b_t_imag))
        b_h = complex(float(b_h_real), float(b_h_imag))

        V_h = self.V_h.value()
        V_t = self.V_t.value()
        APM = self.APM.value()

        thick_h = self.thick_h.value()
        thick_t = self.thick_t.value()

        head_solvent = self.head_solvent.value()
        tail_solvent = self.tail_solvent.value()

        leaflet = LipidLeaflet(
            APM,
            b_h,
            V_h,
            thick_h,
            b_t,
            V_t,
            thick_t,
            3,
            3,
            head_solvent=SLD(head_solvent, name="head solvent"),
            tail_solvent=SLD(tail_solvent, name="tail solvent"),
        )
        leaflet.name = "leaflet"
        return leaflet


class Lipid:
    def __init__(
        self,
        name,
        head_formula,
        tail_formula,
        head_exchangable=0,
        tail_exchangable=0,
        references=None,
        conditions=None,
        chemical_name=None,
    ):
        self.name = name
        self.chemical_name = chemical_name
        self.head_formula = head_formula
        self.tail_formula = tail_formula
        self.references = references
        self.conditions = conditions
        if conditions is None:
            self.conditions = {}
        self.head_exchangable = head_exchangable
        self.tail_exchangable = tail_exchangable

    def __repr__(self):
        s = (
            f"Lipid({self.name!r}, "
            f"{self.head_formula!r}, "
            f"{self.tail_formula!r}, "
            f"head_exchangable={self.head_exchangable!r}, "
            f"tail_exchangable={self.tail_exchangable!r}, "
            f"references={self.references!r},"
            f"conditions={self.conditions!r}, "
            f"chemical_name={self.chemical_name!r}"
        )
        return s

    def add_condition(self, descriptor, vh, vt):
        self.conditions[descriptor] = (vh, vt)

    @property
    def formula(self):
        return pt.formula(self.head_formula) + pt.formula(self.tail_formula)

    @property
    def tf(self):
        return pt.formula(self.tail_formula)

    @property
    def hf(self):
        return pt.formula(self.head_formula)

    def neutron_scattering_lengths(
        self, condition, vf_d_solvent=1, neutron_wavelength=1.8
    ):
        vh, vt = self.conditions[condition]

        hf = exchange_protons_formula(
            self.hf, self.head_exchangable, vf_d_solvent
        )
        tf = exchange_protons_formula(
            self.tf, self.tail_exchangable, vf_d_solvent
        )

        h_density = calculate_density(hf, vh)
        t_density = calculate_density(tf, vt)

        h_sld = pt.neutron_sld(
            hf, density=h_density, wavelength=neutron_wavelength
        )
        t_sld = pt.neutron_sld(
            tf, density=t_density, wavelength=neutron_wavelength
        )
        h_scatlen = complex(*h_sld[0:2]) * vh / 1e6
        t_scatlen = complex(*t_sld[0:2]) * vt / 1e6
        return h_scatlen, t_scatlen

    def xray_scattering_lengths(self, condition, energy=8.048):
        vh, vt = self.conditions[condition]
        h_density = calculate_density(self.hf, vh)
        t_density = calculate_density(self.tf, vt)

        h_sld = pt.xray_sld(self.hf, density=h_density, energy=energy)
        t_sld = pt.xray_sld(self.tf, density=t_density, energy=energy)
        h_scatlen = complex(*h_sld[0:2]) * vh / 1e6
        t_scatlen = complex(*t_sld[0:2]) * vt / 1e6
        return h_scatlen, t_scatlen


def calculate_density(formula, volume):
    return formula.molecular_mass / formula.volume(a=volume, b=1, c=1)


def convert_atoms_to_formula(atoms):
    s = []
    for k, v in atoms.items():
        s.append([v, k])
    return pt.formula(s)


def exchange_protons_formula(formula, exchangable, vf_d):
    if not exchangable or pt.H not in formula.atoms:
        return formula

    atoms = formula.atoms
    initial = atoms[pt.H]
    atoms[pt.H] = initial - exchangable * vf_d
    atoms[pt.D] = exchangable * vf_d
    return convert_atoms_to_formula(atoms)
