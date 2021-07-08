"""
Component for studying lipid membranes at an interface
"""

import numpy as np
from refnx.reflect import Component, SLD, ReflectModel, Structure
from refnx.analysis import possibly_create_parameter, Parameters, Parameter


class LipidLeaflet(Component):
    r"""
    Describes a lipid leaflet Component at an interface

    Parameters
    ----------
    APM: float or refnx.analysis.Parameter
    b_heads: float, refnx.analysis.Parameter, complex or SLD
        Sum of coherent scattering lengths of head group (Angstrom).
        When an SLD is provided it is simply an easy way to provide a complex
        value. `LipidLeaflet.b_heads_real` is set to `SLD.real`, etc.
    vm_heads: float or refnx.analysis.Parameter
        Molecular volume of head group (Angstrom**2)
    thickness_heads: float or refnx.analysis.Parameter
        Thickness of head group region (Angstrom)
    b_tails: float, refnx.analysis.Parameter, complex or SLD
        Sum of coherent scattering lengths of tail group (Angstrom).
        When an SLD is provided it is simply an easy way to provide a complex
        value. `LipidLeaflet.b_tails_real` is set to `SLD.real`, etc.
    vm_tails: float or refnx.analysis.Parameter
        Molecular volume of tail group (Angstrom**2)
    thickness_tails: float or refnx.analysis.Parameter
        Thickness of head group region (Angstrom)
    rough_head_tail: float or refnx.analysis.Parameter
        Roughness of head-tail group (Angstrom)
    rough_preceding_mono: float or refnx.analysis.Parameter
        Roughness between preceding component (in the fronting direction) and
        the monolayer (Angstrom). If `reverse_monolayer is False` then this is
        the roughness between the preceding component and the heads, if
        `reverse_monolayer is True` then this is the roughness between the
        preceding component and the tails.
    head_solvent: None, float, complex, refnx.reflect.SLD
        Solvent for the head region. If `None`, then solvation will be
        performed by the parent `Structure`, using the `Structure.solvent`
        attribute. Other options are coerced to an `SLD` object using
        `SLD(float | complex)`. A float/complex argument is the SLD of the
        solvent (10**-6 Angstrom**-2).
    tail_solvent: None, float, complex, refnx.reflect.SLD
        Solvent for the tail region. If `None`, then solvation will be
        performed by the parent `Structure`, using the `Structure.solvent`
        attribute. Other options are coerced to an `SLD` object using
        `SLD(float | complex)`. A float/complex argument is the SLD of the
        solvent (10**-6 Angstrom**-2).
    reverse_monolayer: bool, optional
        The default is to have heads closer to the fronting medium and
        tails closer to the backing medium. If `reverse_monolayer is True`
        then the tails will be closer to the fronting medium and heads
        closer to the backing medium.
    name: str, optional
        The name for the component

    Notes
    -----
    The sum of coherent scattering lengths must be in Angstroms, the volume
    must be in cubic Angstroms. This is because the SLD of a tail group is
    calculated as `b_tails / vm_tails * 1e6` to achieve the units
    10**6 Angstrom**-2.
    """

    # TODO: use SLD of head instead of b_heads, vm_heads?
    def __init__(
        self,
        apm,
        b_heads,
        vm_heads,
        thickness_heads,
        b_tails,
        vm_tails,
        thickness_tails,
        rough_head_tail,
        rough_preceding_mono,
        head_solvent=None,
        tail_solvent=None,
        reverse_monolayer=False,
        name="",
    ):
        """
        Parameters
        ----------
        apm: float or Parameter
            Area per molecule
        b_heads: float, Parameter or complex
            Sum of coherent scattering lengths of head group (Angstrom)
        vm_heads: float or Parameter
            Molecular volume of head group (Angstrom**3)
        thickness_heads: float or Parameter
            Thickness of head group region (Angstrom)
        b_tails: float, Parameter or complex
            Sum of coherent scattering lengths of tail group (Angstrom)
        vm_tails: float or Parameter
            Molecular volume of tail group (Angstrom**3)
        thickness_tails: float or Parameter
            Thickness of head group region (Angstrom)
        rough_head_tail: float or refnx.analysis.Parameter
            Roughness of head-tail group (Angstrom)
        rough_preceding_mono: float or Parameter
            Roughness between preceding component (in the fronting direction)
            and the monolayer (Angstrom). If `reverse_monolayer is False` then
            this is the roughness between the preceding component and the
            heads, if `reverse_monolayer is True` then this is the roughness
            between the preceding component and the tails.
        head_solvent: None, float, complex, SLD
            Solvent for the head region. If `None`, then solvation will be
            performed by the parent `Structure`, using the `Structure.solvent`
            attribute. Other options are coerced to an `SLD` object using
            `SLD(float | complex)`. A float/complex argument is the SLD of the
            solvent (10**-6 Angstrom**-2).
        tail_solvent: None, float, complex, SLD
            Solvent for the tail region. If `None`, then solvation will be
            performed by the parent `Structure`, using the `Structure.solvent`
            attribute. Other options are coerced to an `SLD` object using
            `SLD(float | complex)`. A float/complex argument is the SLD of the
            solvent (10**-6 Angstrom**-2).
        reverse_monolayer: bool, optional
            The default is to have heads closer to the fronting medium and
            tails closer to the backing medium. If `reverse_monolayer is True`
            then the tails will be closer to the fronting medium and heads
            closer to the backing medium.
        name: str, optional
            The name for the component
        """
        super().__init__()
        self.apm = possibly_create_parameter(
            apm, "%s - area_per_molecule" % name, units="Å**2"
        )

        if isinstance(b_heads, complex):
            self.b_heads_real = possibly_create_parameter(
                b_heads.real, name="%s - b_heads_real" % name
            )
            self.b_heads_imag = possibly_create_parameter(
                b_heads.imag, name="%s - b_heads_imag" % name
            )
        elif isinstance(b_heads, SLD):
            self.b_heads_real = b_heads.real
            self.b_heads_imag = b_heads.imag
        else:
            self.b_heads_real = possibly_create_parameter(
                b_heads, name="%s - b_heads_real" % name
            )
            self.b_heads_imag = possibly_create_parameter(
                0, name="%s - b_heads_imag" % name
            )

        self.b_heads_real.units = self.b_heads_imag.units = "Å"

        self.vm_heads = possibly_create_parameter(
            vm_heads, name="%s - vm_heads" % name, units="Å**3"
        )

        self.thickness_heads = possibly_create_parameter(
            thickness_heads, name="%s - thickness_heads" % name, units="Å"
        )

        if isinstance(b_tails, complex):
            self.b_tails_real = possibly_create_parameter(
                b_tails.real, name="%s - b_tails_real" % name
            )
            self.b_tails_imag = possibly_create_parameter(
                b_tails.imag, name="%s - b_tails_imag" % name
            )
        elif isinstance(b_tails, SLD):
            self.b_tails_real = b_tails.real
            self.b_tails_imag = b_tails.imag
        else:
            self.b_tails_real = possibly_create_parameter(
                b_tails, name="%s - b_tails_real" % name
            )
            self.b_tails_imag = possibly_create_parameter(
                0, name="%s - b_tails_imag" % name
            )
        self.b_tails_real.units = self.b_tails_imag.units = "Å"

        self.vm_tails = possibly_create_parameter(
            vm_tails, name="%s - vm_tails" % name, units="Å**3"
        )
        self.thickness_tails = possibly_create_parameter(
            thickness_tails, name="%s - thickness_tails" % name, units="Å"
        )
        self.rough_head_tail = possibly_create_parameter(
            rough_head_tail, name="%s - rough_head_tail" % name, units="Å"
        )
        self.rough_preceding_mono = possibly_create_parameter(
            rough_preceding_mono,
            name="%s - rough_fronting_mono" % name,
            units="Å",
        )

        self.head_solvent = self.tail_solvent = None
        if head_solvent is not None:
            self.head_solvent = SLD(head_solvent)
        if tail_solvent is not None:
            self.tail_solvent = SLD(tail_solvent)

        self.reverse_monolayer = reverse_monolayer
        self.name = name

    def __repr__(self):
        d = {}
        d.update(self.__dict__)
        sld_bh = SLD([self.b_heads_real, self.b_heads_imag])
        sld_bt = SLD([self.b_tails_real, self.b_tails_imag])
        d["bh"] = sld_bh
        d["bt"] = sld_bt

        s = (
            "LipidLeaflet({apm!r}, {bh!r}, {vm_heads!r}, {thickness_heads!r},"
            " {bt!r}, {vm_tails!r}, {thickness_tails!r}, {rough_head_tail!r},"
            " {rough_preceding_mono!r}, head_solvent={head_solvent!r},"
            " tail_solvent={tail_solvent!r},"
            " reverse_monolayer={reverse_monolayer}, name={name!r})"
        )
        return s.format(**d)

    def slabs(self, structure=None):
        """
        Slab representation of monolayer, as an array

        Parameters
        ----------
        structure : refnx.reflect.Structure
            The Structure hosting this Component
        """
        layers = np.zeros((2, 5))

        # thicknesses
        layers[0, 0] = float(self.thickness_heads)
        layers[1, 0] = float(self.thickness_tails)

        # real and imag SLD's
        layers[0, 1] = float(self.b_heads_real) / float(self.vm_heads) * 1.0e6
        layers[0, 2] = float(self.b_heads_imag) / float(self.vm_heads) * 1.0e6

        layers[1, 1] = float(self.b_tails_real) / float(self.vm_tails) * 1.0e6
        layers[1, 2] = float(self.b_tails_imag) / float(self.vm_tails) * 1.0e6

        # roughnesses
        layers[0, 3] = float(self.rough_preceding_mono)
        layers[1, 3] = float(self.rough_head_tail)

        # volume fractions
        # head region
        volfrac = self.vm_heads.value / (
            self.apm.value * self.thickness_heads.value
        )
        layers[0, 4] = 1 - volfrac
        if self.head_solvent is not None:
            # we do the solvation here, not in Structure.slabs
            layers[0] = Structure.overall_sld(layers[0], self.head_solvent)
            layers[0, 4] = 0

        # tail region
        volfrac = self.vm_tails.value / (
            self.apm.value * self.thickness_tails.value
        )

        layers[1, 4] = 1 - volfrac
        if self.tail_solvent is not None:
            # we do the solvation here, not in Structure.slabs
            layers[1] = Structure.overall_sld(layers[1], self.tail_solvent)
            layers[1, 4] = 0

        if self.reverse_monolayer:
            layers = np.flipud(layers)
            layers[:, 3] = layers[::-1, 3]

        return layers

    @property
    def parameters(self):
        p = Parameters(name=self.name)
        p.extend(
            [
                self.apm,
                self.b_heads_real,
                self.b_heads_imag,
                self.vm_heads,
                self.thickness_heads,
                self.b_tails_real,
                self.b_tails_imag,
                self.vm_tails,
                self.thickness_tails,
                self.rough_head_tail,
                self.rough_preceding_mono,
            ]
        )
        if self.head_solvent is not None:
            p.append(self.head_solvent.parameters)
        if self.tail_solvent is not None:
            p.append(self.tail_solvent.parameters)

        return p

    def logp(self):
        # penalise unphysical volume fractions.
        volfrac_h = self.vm_heads.value / (
            self.apm.value * self.thickness_heads.value
        )

        # tail region
        volfrac_t = self.vm_tails.value / (
            self.apm.value * self.thickness_tails.value
        )

        if volfrac_h > 1 or volfrac_t > 1:
            return -np.inf

        return 0
