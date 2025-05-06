"""
*Calculates the specular (Neutron or X-ray) reflectivity from a stratified
series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2025 A. R. J. Nelson, ANSTO

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

"""

from importlib import resources
import numpy as np
import numpy.typing as npt
from typing import Optional

from refnx.analysis import possibly_create_parameter, Parameter, Parameters
from refnx.reflect.reflect_model import (
    ReflectModel,
    SpinChannel,
    kernel,
    reflectivity,
)

try:
    from refnx.reflect._creflect import gepore
except ImportError:

    def gepore(*args, **kwds):
        raise RuntimeError("gepore is not available")


class _MemoisingReflectivityKernel:
    def __init__(self, Aguide=270):
        self._memoised_x = None
        self._memoised_slabs = None
        self._memoised_reflectivity = None
        self.updated = False
        self.c = {
            SpinChannel.UP_UP: 0,
            SpinChannel.UP_DOWN: 1,
            SpinChannel.DOWN_UP: 2,
            SpinChannel.DOWN_DOWN: 3,
        }
        self.spin = SpinChannel.UP_UP
        self.Aguide = Aguide

    def __call__(self, x, slabs, *args, **kwds):
        shp = x.shape

        # calculate all four spin channels for a given set of x.
        if (
            self.updated
            and np.array_equal(self._memoised_x, x)
            and np.array_equal(self._memoised_slabs, slabs)
        ):
            return self._memoised_reflectivity[self.c[self.spin]].reshape(shp)

        # if x is being called by a smearing kernel it may be presented in a
        # N-dimensional fashion, e.g. with shape (N, M).
        # gepore will always return an array of shape (N*M, 4). We therefore
        # store the memoised reflectivity in a flattened fashion, but reshape
        # the correct spin state to (N, M) on return.
        xflat = np.ravel(x)
        kwds["Aguide"] = self.Aguide
        R = gepore(xflat, slabs, *args, **kwds)
        self._memoised_reflectivity = R
        self._memoised_slabs = slabs
        self._memoised_x = x
        self.updated = True
        return R[self.c[self.spin]].reshape(shp)


class PolarisedReflectModel(ReflectModel):
    """
    Extension of ReflectModel for polarised neutron reflectometry.
    See `refnx.reflect.ReflectModel` for documentation of arguments.

    Parameters
    ----------
    structure : refnx.reflect.Structure
        The interfacial structure.
    scales : {float, refnx.analysis.Parameter, sequence, refnx.analysis.Parameters}, optional
        Scale factor for each spin channel. All model values are multiplied by
        this(/these) value(s) before the background is added. They are turned into a
        Parameter during the construction of this object. If an iterable is
        provided it should contain four values (one for each channel), in the
        order:

            - SpinChannel.UP_UP
            - SpinChannel.UP_DOWN
            - SpinChannel.DOWN_UP
            - SpinChannel.DOWN_DOWN

    bkgs : {float, refnx.analysis.Parameter, sequence, refnx.analysis.Parameters}, optional
        Background for each spin channel. Q-independent constant background
        added to all model values. They are turned into a Parameter during
        the construction of this object. If an iterable is
        provided it should contain four values (one for each channel).
    name : str, optional
        Name of the Model
    q_offsets: {float, refnx.analysis.Parameter, sequence, refnx.analysis.Parameters}, optional
        Compensates for uncertainties in the angle at which the measurement is
        performed. A positive/negative `q_offset` corresponds to a situation
        where the measured q values (incident angle) may have been under/over
        estimated, and has the effect of shifting the calculated model to
        lower/higher effective q values. They are turned into a Parameter during
        the construction of this object. If an iterable is
        provided it should contain four values (one for each channel).
    Aguide: float
        Angle of applied field. This value should be 270 or 90 degrees for
        the applied field to lie in the plane of the sample, perpendicular to
        the beam propagation direction. For a magnetic moment to be parallel
        or anti-parallel to the applied field `thetaM` should be 90 or -90 deg
        respectively.
    """

    def __init__(
        self,
        structure,
        scales=1.0,
        bkgs=1e-7,
        name="",
        dq=5.0,
        threads=-1,
        quad_order=17,
        dq_type="constant",
        q_offsets=0,
        Aguide=270,
    ):
        super().__init__(
            structure,
            name=name,
            threads=threads,
            quad_order=quad_order,
            dq=dq,
            dq_type=dq_type,
        )
        # overwrite the scales/bkgs/q_offsets
        if hasattr(scales, "__iter__"):
            _s = [possibly_create_parameter(v) for v in scales]
            if len(_s) != 4:
                raise RuntimeError(
                    "Four scale values need to be provided, one for each spin channel"
                )
            self._scale = Parameters(name="scales", data=_s)
        else:
            self._scale = Parameter(scales, name="scale")

        if hasattr(bkgs, "__iter__"):
            _s = [possibly_create_parameter(v) for v in bkgs]
            if len(_s) != 4:
                raise RuntimeError(
                    "Four bkg values need to be provided, one for each spin channel"
                )
            self._bkg = Parameters(name="bkgs", data=_s)
        else:
            self._bkg = Parameter(bkgs, name="bkg")

        if hasattr(q_offsets, "__iter__"):
            _s = [possibly_create_parameter(v) for v in q_offsets]
            if len(_s) != 4:
                raise RuntimeError(
                    "Four q_offset values need to be provided, one for each spin channel"
                )
            self._q_offset = Parameters(name="q_offsets", data=_s)
        else:
            self._q_offset = Parameter(q_offsets, name="q_offset")

        # update internal parameter view
        self.structure = structure
        self.Aguide = Aguide
        self._memoising_kernel = _MemoisingReflectivityKernel()

    def __repr__(self):
        return (
            f"PolarisedReflectModel({self._structure!r}, name={self.name!r},"
            f" scales={self.scale!r}, bkgs={self.bkg!r},"
            f" dq={self.dq!r}, threads={self.threads},"
            f" quad_order={self.quad_order!r}, dq_type={self.dq_type!r},"
            f" q_offsets={self.q_offset!r}, Aguide={self.Aguide!r})"
        )

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, val):
        raise RuntimeError("Setter disabled in subclass")

    @property
    def bkg(self):
        return self._bkg

    @bkg.setter
    def bkg(self, val):
        raise RuntimeError("Setter disabled in subclass")

    @property
    def q_offset(self):
        return self._q_offset

    @q_offset.setter
    def q_offset(self, val):
        raise RuntimeError("Setter disabled in subclass")

    def model(self, x, p=None, x_err=None, spin="all"):
        r"""
        Calculate the reflectivity of this model

        Parameters
        ----------
        x : float or np.ndarray
            q values for the calculation. This should be a 2-D array with
            4 columns.
            In order the 4 columns represent:

                - SpinChannel.UP_UP
                - SpinChannel.UP_DOWN
                - SpinChannel.DOWN_UP
                - SpinChannel.DOWN_DOWN

            In a given row only one column should contain a finite value, the
            others should be set to np.nan.
            If a single spin channel is requested then `x` can be 1-D.
            Units = Angstrom**-1
        p : refnx.analysis.Parameters, optional
            parameters required to calculate the model
        x_err : {np.ndarray, float} optional
            Specifies how the instrumental resolution smearing is carried out
            for each of the points in `x`. If an array it should have the same
            shape as `x`.
            See :func:`refnx.reflect.reflectivity` for further details.
        spin : {"all", SpinChannel}, optional
            Specifies which spin to return from the calculation. If all then
            `x` needs to be 2-D. If a single spin channel is requested then
            `x` can be 1-D.

        Returns
        -------
        reflectivity : np.ndarray
            Calculated reflectivity

        Notes
        -----
        If `x_err` is not provided then the calculation will fall back to
        the constant dq/q smearing specified by the `dq` attribute of this
        object.
        """
        if p is not None:
            self.parameters.pvals = np.array(p)
        if x_err is None or self.dq_type == "constant":
            # fallback to what this object was constructed with
            x_err = float(self.dq)

        if spin != "all":
            # only want to calculate one spin channel
            pass
        elif (
            x.ndim != 2
            or x.shape[1] != 4
            or np.any(np.count_nonzero(np.isfinite(x), axis=1) > 1)
        ):
            raise RuntimeError(
                "For PolarisedReflectModel.model the x array should have shape"
                " (N, 4). Each of the columns represents a SpinChannel. Only"
                " one of the entries in a row should be finite, the rest"
                " should be np.nan"
            )

        slabs = self.structure.slabs()
        if slabs.shape[1] == 5:
            # unpolarised for some reason
            raise ValueError(
                "Please use ReflectModel, not PolarisedReflectModel,"
                " for non-magnetic systems."
            )
            # slabs = slabs[..., :4]
            # return reflectivity(
            #     _x_union,
            #     slabs,
            #     scale=self.scale[0].value,
            #     bkg=self.bkg[0].value,
            #     dq=x_err,
            #     threads=self.threads,
            #     quad_order=self.quad_order,
            #     q_offset=self.q_offset[0].value,
            # )
        if slabs.shape[1] == 7:
            # polarised
            slabs = np.take_along_axis(
                slabs, np.array([0, 1, 2, 3, 5, 6])[None, :], axis=1
            )

        if isinstance(self.scale, Parameters):
            scale = np.array(self.scale)
        else:
            scale = [self.scale.value] * 4

        if isinstance(self.bkg, Parameters):
            bkg = np.array(self.bkg)
        else:
            bkg = [self.bkg.value] * 4

        if spin != "all":
            # calculate single spin channel
            self._memoising_kernel.spin = spin
            idx = self._memoising_kernel.c[spin]
            scale = scale[idx]
            bkg = bkg[idx]
            _R = reflectivity(
                x,
                slabs,
                scale=scale,
                bkg=bkg,
                dq=x_err,
                threads=self.threads,
                quad_order=self.quad_order,
                fkernel=self._memoising_kernel,
            )
            return _R

        # calculate all spin channels at once
        _x_union = self._calculate_x_union(x)
        _x_err_union = self._calculate_x_err_union(x, x_err)
        active_spins = self._active_spins(x)

        # if we're only calculating NSF, there's no need to use gepore
        # shortcut to use unpolarised calculator.
        def is_there_spin_flip(slabs):
            return any(
                np.logical_and(
                    slabs[:, 4] != 0.0,
                    np.abs(np.sin(np.radians(slabs[:, 5]))) < 1,
                )
            )

        if (
            not any(active_spins[1:3])  # UU, UD, DU, DD
            and self.Aguide == 270
            and not is_there_spin_flip(slabs)
        ):
            return self._calculate_nsf_only(
                active_spins, slabs, x, x_err, scale, bkg
            )

        # calculate smeared reflectivity on the entire _x_union
        R_union = np.zeros(len(x))

        for idx, spin in enumerate(SpinChannel):
            if not active_spins[idx]:
                continue

            msk = np.isfinite(x[:, idx])
            self._memoising_kernel.spin = spin

            _R = reflectivity(
                _x_union,
                slabs,
                dq=_x_err_union,
                threads=self.threads,
                quad_order=self.quad_order,
                fkernel=self._memoising_kernel,
            )
            R_union[msk] = _R[msk] * scale[idx] + bkg[idx]

        return R_union

    def _calculate_x_union(self, x):
        if isinstance(self.q_offset, Parameters):
            q_offsets = np.array(self.q_offset)
        else:
            q_offsets = [self.q_offset.value] * 4

        _x = x.copy()
        for idx in range(4):
            _x[:, idx] += q_offsets[idx]

        return np.nanmax(_x, axis=1)

    def _calculate_x_err_union(self, x, x_err):
        if isinstance(x_err, np.ndarray):
            if x.shape != x_err.shape:
                raise RuntimeError(
                    f"x_err.shape and x.shape should be equal"
                    f" {x_err.shape=}, {x.shape=}"
                )
            x_err_union = np.zeros(len(x))
            for idx in range(4):
                msk = np.isfinite(x[:, idx])
                x_err_union[msk] = x_err[msk, idx]

            return x_err_union
        else:
            # single number represents constant dq/q
            return x_err

    def _active_spins(self, x):
        return np.any(np.isfinite(x), axis=0)

    def _calculate_nsf_only(self, active_spins, slabs, x, x_err, scale, bkg):
        """
        Calculates NSF reflectivities only.

        Parameters
        ----------
        active_spins : np.ndarray
            Length-4 boolean array specifying which spin channels are being
            requested. UU, UD, DU, DD
        slabs : np.ndarray
            Slab representation of the interfacial model.
            See `refnx.reflect.reflectivity` for further details
        x : np.ndarray
            q values for the calculation. This should be a 2-D array with
            4 columns.
            In order the 4 columns represent:

                - SpinChannel.UP_UP
                - SpinChannel.UP_DOWN
                - SpinChannel.DOWN_UP
                - SpinChannel.DOWN_DOWN

            In a given row only one column should contain a finite value, the
            others should be set to np.nan. For this NSF function there should
            be no finite values in columns 1, 2.
            Units = Angstrom**-1
        x_err : {np.ndarray, float} optional
            Specifies how the instrumental resolution smearing is carried out
            for each of the points in `x`.
            See :func:`refnx.reflect.reflectivity` for further details.
        scale : sequence
            Length-4 sequence containing the scale factors for each spin channel.
        bkg : sequence
            Length-4 sequence containing the backgrounds for each spin channel.
        """
        R_union = np.zeros(len(x))

        # work out overall SLD of slabs
        sld_u = slabs[:, 1] + slabs[:, 4] * np.sin(np.radians(slabs[:, 5]))
        sld_d = slabs[:, 1] - slabs[:, 4] * np.sin(np.radians(slabs[:, 5]))
        slabs_u = slabs.copy()[:, :4]
        slabs_d = slabs.copy()[:, :4]
        slabs_u[:, 1] = sld_u
        slabs_d[:, 1] = sld_d

        if active_spins[0]:
            # UU
            msk = np.isfinite(x[:, 0])
            xuu = x[msk, 0]
            if isinstance(x_err, np.ndarray):
                xuu_err = x_err[msk, 0]
            else:
                xuu_err = x_err

            _R = reflectivity(
                xuu,
                slabs_u,
                dq=xuu_err,
                threads=self.threads,
                quad_order=self.quad_order,
            )
            R_union[msk] = _R * scale[0] + bkg[0]

        if active_spins[3]:
            # DD
            msk = np.isfinite(x[:, 3])
            xdd = x[msk, 3]
            if isinstance(x_err, np.ndarray):
                xdd_err = x_err[msk, 3]
            else:
                xdd_err = x_err

            _R = reflectivity(
                xdd,
                slabs_d,
                dq=xdd_err,
                threads=self.threads,
                quad_order=self.quad_order,
            )
            R_union[msk] = _R * scale[3] + bkg[3]

        return R_union


def pnr_data_and_generative(objective):
    """
    The Data1D and generative model for a Polarised Reflectometry system

    Parameters
    ----------
    objective : Objective

    Returns
    -------
    pdg : tuple
        Tuple of (Data1D, generative)
    """
    pdg = []
    combined = objective.data
    generative = objective.generative()
    pos = 0
    for spin in combined.spins.keys():
        data = getattr(combined, spin)
        if data is not None:
            npnts = len(data)
            g = generative[pos : pos + npnts]
            pdg.append((data, g))
            pos += npnts
    return pdg
