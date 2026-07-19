"""
LipidLeaflet._jax_slabs
=======================

A worked implementation of the ``_jax_slabs(compiler)`` extension hook for
``LipidLeaflet``.

How to use
----------
Monkey-patch this onto ``LipidLeaflet`` before calling ``compile_objective``,
or subclass it:

    from refnx.reflect import LipidLeaflet
    from lipid_leaflet_jax_slabs import lipid_leaflet_jax_slabs

    LipidLeaflet._jax_slabs = lipid_leaflet_jax_slabs

Background: what LipidLeaflet.slabs() does
-------------------------------------------
The component models a monolayer with a head region and a tail region.
It returns a (2, 5) array.  The two rows are ordered so that the
region closest to the *preceding* component (in the fronting direction)
comes first:

  reverse_monolayer=False  →  row 0 = heads, row 1 = tails
  reverse_monolayer=True   →  row 0 = tails, row 1 = heads

For each region the key quantities are:

  phi   = vm / (apm * thickness)          # lipid volume fraction
  vfsolv = 1 - phi                        # solvent volume fraction

  SLD_lipid_real = b_real / (vm * 1e-6)  # pure-lipid SLD (×10⁻⁶ Å⁻²)
  SLD_lipid_imag = b_imag / (vm * 1e-6)

These go into the slab as the *material* SLD and vfsolv.
``Structure.overall_sld`` (or the JAX equivalent in _make_params_to_slabs)
then does the final mixing:

    SLD_mixed = SLD_lipid * (1 - vfsolv) + SLD_solvent * vfsolv
              = SLD_lipid * phi           (+ solvent contribution)

There is a special case: when ``head_solvent`` (or ``tail_solvent``) is not
None the component performs its *own* mixing for that region and sets
``vfsolv = 0`` in the returned slab, signalling to Structure that no further
mixing is needed.  We replicate this below.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List
from refnx.reflect.extra._jax_util import (
    _sub,
    _mul,
    _div,
    _add,
    _const,
    _SlabSpec,
    _ConstraintCompiler,
)

# ---------------------------------------------------------------------------
# Main implementation
# ---------------------------------------------------------------------------


def _lipid_leaflet_jax_slabs(
    self, compiler: _ConstraintCompiler
) -> List[_SlabSpec]:
    """
    ``_jax_slabs`` hook for ``LipidLeaflet``.

    Returns a list of exactly two ``_SlabSpec`` objects, matching the row
    order and formulae of ``LipidLeaflet.slabs()``.  All arithmetic is
    expressed as IR nodes so that JAX can differentiate through every
    quantity that depends on a free parameter.

    Parameters
    ----------
    compiler : _ConstraintCompiler
        The compiler instance created by ``compile_objective``.  Used to
        turn each ``Parameter`` attribute of the leaflet into an IR node.

    Returns
    -------
    list of _SlabSpec, length 2
        [preceding-side region spec, far-side region spec]
    """
    # ------------------------------------------------------------------
    # 1.  Compile every Parameter on self into an IR node
    # ------------------------------------------------------------------
    apm_node = compiler.compile_parameter(self.apm)

    # b_heads and b_tails are stored split into real/imag Parameters
    b_heads_re = compiler.compile_parameter(self.b_heads_real)
    b_heads_im = compiler.compile_parameter(self.b_heads_imag)
    b_tails_re = compiler.compile_parameter(self.b_tails_real)
    b_tails_im = compiler.compile_parameter(self.b_tails_imag)

    vm_heads_node = compiler.compile_parameter(self.vm_heads)
    vm_tails_node = compiler.compile_parameter(self.vm_tails)

    thick_heads_node = compiler.compile_parameter(self.thickness_heads)
    thick_tails_node = compiler.compile_parameter(self.thickness_tails)

    rough_ht_node = compiler.compile_parameter(self.rough_head_tail)
    rough_pre_node = compiler.compile_parameter(self.rough_preceding_mono)

    # ------------------------------------------------------------------
    # 2.  Derive phi (lipid volume fraction) for each region
    #
    #     phi = vm / (apm * thickness)
    # ------------------------------------------------------------------
    phi_heads_node = _div(vm_heads_node, _mul(apm_node, thick_heads_node))
    phi_tails_node = _div(vm_tails_node, _mul(apm_node, thick_tails_node))

    # vfsolv = 1 - phi   (fraction left for solvent, before Structure mixes)
    one = _const(1.0)
    vfsolv_heads_node = _sub(one, phi_heads_node)
    vfsolv_tails_node = _sub(one, phi_tails_node)

    # ------------------------------------------------------------------
    # 3.  Derive pure-lipid SLD for each region
    #
    #     SLD = b / (vm * 1e-6)    [gives ×10⁻⁶ Å⁻²]
    #
    #     The 1e-6 converts b (Å) and vm (Å³) into the conventional
    #     SLD units used throughout refnx.
    # ------------------------------------------------------------------
    scale = _const(1e-6)

    sld_heads_re_node = _div(b_heads_re, _mul(vm_heads_node, scale))
    sld_heads_im_node = _div(b_heads_im, _mul(vm_heads_node, scale))
    sld_tails_re_node = _div(b_tails_re, _mul(vm_tails_node, scale))
    sld_tails_im_node = _div(b_tails_im, _mul(vm_tails_node, scale))

    # ------------------------------------------------------------------
    # 4.  Handle head_solvent / tail_solvent
    #
    #     When head_solvent is not None the component mixes the head
    #     region itself and sets vfsolv = 0 so Structure skips it.
    #     We must replicate this here.  The solvent SLD is baked in as a
    #     constant (same treatment as the stateful code — the solvent
    #     contrast is fixed per experiment).
    #
    #     Mixed SLD = SLD_lipid * phi + SLD_solvent * (1 - phi)
    #               = SLD_lipid * phi + SLD_solvent * vfsolv
    # ------------------------------------------------------------------
    zero = _const(0.0)

    if self.head_solvent is not None:
        solv_h = complex(self.head_solvent)
        solv_h_re = _const(solv_h.real)
        solv_h_im = _const(solv_h.imag)
        # mixed = lipid * phi + solvent * vfsolv
        sld_heads_re_node = _add(
            _mul(sld_heads_re_node, phi_heads_node),
            _mul(solv_h_re, vfsolv_heads_node),
        )
        sld_heads_im_node = _add(
            _mul(sld_heads_im_node, phi_heads_node),
            _mul(solv_h_im, vfsolv_heads_node),
        )
        vfsolv_heads_node = zero  # tell Structure: already mixed

    if self.tail_solvent is not None:
        solv_t = complex(self.tail_solvent)
        solv_t_re = _const(solv_t.real)
        solv_t_im = _const(solv_t.imag)
        sld_tails_re_node = _add(
            _mul(sld_tails_re_node, phi_tails_node),
            _mul(solv_t_re, vfsolv_tails_node),
        )
        sld_tails_im_node = _add(
            _mul(sld_tails_im_node, phi_tails_node),
            _mul(solv_t_im, vfsolv_tails_node),
        )
        vfsolv_tails_node = zero

    # ------------------------------------------------------------------
    # 5.  Assemble _SlabSpec objects
    #
    #     Row ordering mirrors LipidLeaflet.slabs():
    #
    #     reverse_monolayer=False:
    #       row 0  heads  rough = rough_preceding_mono  (heads face fronting)
    #       row 1  tails  rough = rough_head_tail
    #
    #     reverse_monolayer=True:
    #       row 0  tails  rough = rough_preceding_mono  (tails face fronting)
    #       row 1  heads  rough = rough_head_tail
    # ------------------------------------------------------------------
    heads_spec = _SlabSpec(
        thick=thick_heads_node,
        real=sld_heads_re_node,
        imag=sld_heads_im_node,
        rough=rough_ht_node,  # inner roughness (head/tail interface)
        vfsolv=vfsolv_heads_node,
    )
    tails_spec = _SlabSpec(
        thick=thick_tails_node,
        real=sld_tails_re_node,
        imag=sld_tails_im_node,
        rough=rough_ht_node,  # same roughness used for tail side
        vfsolv=vfsolv_tails_node,
    )

    # The roughness of the interface between the *preceding* component and
    # the first region of this leaflet is rough_preceding_mono.
    # In the slab convention, roughness lives on the *top* of a layer
    # (i.e. the interface between layer N and layer N-1 above it in the
    # fronting direction), so we apply it to whichever region faces the
    # preceding component.
    if not self.reverse_monolayer:
        # normal orientation: heads closest to preceding component
        heads_spec = _SlabSpec(
            thick=heads_spec.thick,
            real=heads_spec.real,
            imag=heads_spec.imag,
            rough=rough_pre_node,  # override: preceding/head interface
            vfsolv=heads_spec.vfsolv,
        )
        return [heads_spec, tails_spec]
    else:
        # reversed: tails closest to preceding component
        tails_spec = _SlabSpec(
            thick=tails_spec.thick,
            real=tails_spec.real,
            imag=tails_spec.imag,
            rough=rough_pre_node,  # override: preceding/tail interface
            vfsolv=tails_spec.vfsolv,
        )
        return [tails_spec, heads_spec]
