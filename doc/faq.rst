.. _faq_chapter:

====================================
Frequently Asked Questions
====================================

.. _mailing list: https://groups.google.com/group/refnx
.. _github issues: https://github.com/refnx/refnx/issues
.. _van Well et al: https://doi.org/10.1016/j.physb.2004.11.058
.. _Nelson et al: https://doi.org/10.1107/S1600576714009595

A list of common questions.

What's the best way to ask for help or submit a bug report?
================================================================

If you have questions on the use of refnx please use the `mailing list`_.
If you find a bug in the code or documentation, use `GitHub Issues`_.

How should I cite refnx?
===========================

The full reference for the refnx paper is:

    "Nelson, A.R.J. & Prescott, S.W. (2019). J. Appl. Cryst. 52, https://doi.org/10.1107/S1600576718017296."

How is instrumental resolution smearing handled?
=================================================

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