.. refnx documentation master file, created by
   sphinx-quickstart on Fri Oct 23 10:21:57 2015.

refnx - Neutron and X-ray reflectometry analysis in Python
==========================================================

.. _refnx github repository:   http://github.com/refnx/refnx
.. _github:     https://github.com/refnx/refnx/releases/latest
.. _scipy.optimize:      http://docs.scipy.org/doc/scipy/reference/optimize.html
.. _emcee:      http://emcee.readthedocs.io/en/stable/
.. _refnx YouTube channel:     https://www.youtube.com/channel/UCvhOxwZsdFMGqSzasE0ZSOw

*refnx* is a flexible, powerful, Python package for generalised curvefitting
analysis, specifically neutron and X-ray reflectometry data.

It uses several `scipy.optimize`_ algorithms for fitting data, and estimating
parameter uncertainties. As well as the scipy algorithms *refnx* uses the
`emcee`_ Affine Invariant Markov chain Monte Carlo (MCMC) Ensemble sampler for
Bayesian parameter estimation.

Reflectometry analysis uses a modular and object oriented approach to model
parameterisation. Models are made up by sequences of components, frequently
slabs of uniform scattering length density, but other components are available,
including splines for freeform modelling of a scattering length density profile.
These components allow the parameterisation of a model in terms of physically
relevant parameters. The Bayesian nature of the package allows the
specification of prior probabilities for the model, so parameter bounds can be
described in terms of probability distribution functions. These priors not only
applicable to any parameter, but can apply to any other user-definable knowledge
about the system (such as adsorbed amount).
Co-refinement of multiple contrast datasets is straightforward, with sharing of
joint parameters across each model.

Various tutorials are available from the `refnx YouTube channel`_, and there are
GUI programs available on `github`_ as well.

The refnx package is free software, using a BSD licence. If you are interested
in participating in this project please use the `refnx github repository`_, all
contributions are welcomed.

.. toctree::
    :maxdepth: 2

    installation
    getting_started.ipynb
    gui.rst
    reflectometry_global.ipynb
    inequality_constraints.ipynb
    lipid.ipynb
    faq
    testimonials
    modules

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
