.. _faq_chapter:

====================================
Frequently Asked Questions
====================================

.. _mailing list: https://groups.google.com/group/refnx
.. _github issues: https://github.com/refnx/refnx/issues
.. _van Well et al: https://doi.org/10.1016/j.physb.2004.11.058
.. _Nelson et al: https://doi.org/10.1107/S1600576714009595
.. _ORSO: https://www.reflectometry.org/

A list of common questions.

What's the best way to ask for help or submit a bug report?
-----------------------------------------------------------

If you have questions on the use of refnx please use the `mailing list`_.
If you find a bug in the code or documentation, use `GitHub Issues`_.

How should I cite refnx?
------------------------

The full reference for the refnx paper is:

    "Nelson, A.R.J. & Prescott, S.W. (2019). J. Appl. Cryst. 52, https://doi.org/10.1107/S1600576718017296."

How is instrumental resolution smearing handled?
------------------------------------------------

There are a variety of ways that you can account for instrumental resolution
smearing in refnx. The easiest is if the fractional instrumental resolution,
:math:`\frac{dQ}{Q}`, is constant. When setting up
:class:`refnx.reflect.ReflectModel` the fractional resolution can be specified,
and the reflectivity that it calculates is automatically smeared. For a given
:math:`Q` value the :math:`dQ` (found by multiplying the fractional resolution
by :math:`Q`) value refers to the Full Width at Half Maximum (FWHM) of a
Gaussian approximation to the instrumental resolution. This Gaussian
distribution is convolved with the unsmeared model to compare with the data.
The second way of using the resolution function is for the :math:`dQ` values
for each data point to be read in via from a data file (e.g. the 4th column of
a text file). In this way point-by-point resolution smearing is achieved.
The last way of specifying instrumental resolution is for a full resolution
kernel to be provided for each data point. A resolution kernel is a probability
distribution that describes the distribution of possible :math:`Q` vectors for
each data point.
The first two options are typically used, only more advanced users will ever
need to apply the last option. For further details on instrumental resolution
functions it's a good idea to read the papers by `van Well et al`_, and
`Nelson et al`_.

What is data weighting and how is it handled in refnx?
------------------------------------------------------

Experimental data frequently comes with an uncertainty/error bar that expresses
how certain we are of the data's value. When modelling/curve-fitting, we'd
intuitively expect the fitting process to make a bigger effort for the
theoretical model to be close to data points with smaller uncertainty, and a
lesser effort to be close to data points with larger uncertainty. The
experimental uncertainties are stored in the dataset as
:code:`Data1D.y_err`. The uncertainties are :ref:`loadable from the data file<io-formats>`.
See also the documentation for :class:`refnx.dataset.Data1D`.

If you fit with :code:`CurveFitter.fit(method='least_squares')`, then you are
minimising the sum of the squared-(possibly normalised) residuals.

  .. math::

    \chi^2 = \Sigma_i \left(\frac{y_{obs, i} - y_{data, i}}{y_{err, i}}\right)^2

If you fit with a different ``method``, then you either minimizing the
negative log-posterior (:code:`CurveFitter.fit(target='nlpost')`), or
negative log-likelihood (:code:`CurveFitter.fit(target='nll')`).

  .. math::

    nll = \frac{1}{2}\Sigma \left[\left(\frac{y_{obs, i} - y_{calc, i}}{y_{err, i}}\right)^2
                                 + \log \left(2\pi y_{err, i}^2\right)\right]

The negative log posterior is similar, except that it also includes all the
log-prior terms. The log-priors are the probability distributions for what we
expect the model parameter values to be.
You can switch the weighting on/off during the modelling process by setting
:code:`Objective.use_weights` to be :code:`True` or :code:`False`. This has
the effect of setting :math:`y_{err, i}=1` in the above equations.

Minimising the negative log-likelihood/posterior is the same as maximising the
log-likelihood/posterior. These equations assume the experimental uncertainties
are Gaussian in nature.
It is also possible to transform the data and theoretical model before the above
sums are carried out, see :class:`refnx.analysis.Transform`, and
:class:`refnx.analysis.Objective` for more details. For example the ``logY``
transform converts :math:`Y \rightarrow logY`, with the uncertainties being
correctly transformed as well. These transforms can be used to bias the fit in
various ways. For example, while analysing X-ray reflectometry data it's
common to use a log transform with data-weighting switched off,
:code:`Objective(..., use_weighting=False, transform=Transform("logY"))`.
The default settings are usually fine for most neutron reflectometry data.
It's normally a bad idea to use no weighting without a transform to ``logY``.

What are the units of scattering length density?
------------------------------------------------

If the scattering length density of a material is
:math:`(124.88 + 12.85j)\times 10^{-6} A^{-2}` (the X-ray SLD for Au), then you
would use 124.88 as the real part and 12.85 as the imaginary part.

What are the 'fronting' and 'backing' media?
--------------------------------------------

The 'fronting' and 'backing' media are infinite. The 'fronting' medium carries
the incident beam of radiation, whilst the 'backing' medium will carry the
transmitted beam away from the interface.

How do I open the standalone app on macOS Catalina?
----------------------------------------------------

macOS Catalina expects all apps to be code-signed and notarised for them to be
able to run via 'double-clicking' in the finder. The project is working towards
fulfilling those conditions, but in the meantime you can still open the
standalone motofit.app by right-clicking and selecting 'open'.

Can I save models/objectives to file?
-----------------------------------------
I'm assuming that you have a :class:`refnx.reflect.ReflectModel` or
:class:`refnx.analysis.Objective` that you'd like to
save to file. The easiest way to do this is via serialisation to a Python
pickle::

    import pickle
    # save
    with open('my_objective.pkl', 'wb+') as f:
        pickle.dump(objective, f)

    # load
    with open('my_objective.pkl', 'rb') as f:
        restored_objective = pickle.load(f)

The saved pickle files are in a binary format, and are not human readable.
It may also be useful to save the representation, :code:`repr(objective)`.

How do I install pyqt6?
-----------------------
PyQt6 and qtpy is needed for the refnx GUI. The `pyqt6` and `qtpy` packages
are currently available from PyPI and can be installed as
:code:`pip install pyqt6 qtpy`. However, pyqt6 is not currently available via
conda-forge. You can use conda to install most of the refnx dependencies, but
you will need to use `pip` to install pyqt6.

.. _io-formats:

What input/output file formats are supported?
---------------------------------------------
`refnx` can read a range of file types. The most common is 2/3/4 column
ASCII data. The first two columns are Q (:math:`A^{-1}`) and R. If present the
third column will be the standard deviation of the reflectivity. If present
the fourth column will be the full width half maximum of the instrument
resolution smearing function.
`refnx` can also read/write the ORT and ORB files of the Open Reflectometry
Standards Organisation (`ORSO`_).

Can `refnx` analyse Polarised Neutron Reflectometry data?
---------------------------------------------------------
Yes, the ability to simultaneously analyse non-spin flip and spin flip data was
added in version 0.1.51. See :class:`refnx.reflect.PolarisedReflectModel` and
:class:`refnx.reflect.MagneticSlab` for further details.
