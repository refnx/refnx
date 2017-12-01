.. _gettingstarted_chapter:

===================================
Getting started
===================================

.. _Github:   https://github.com/refnx/refnx/blob/master/examples/reflectometry_analysis.ipynb


Fitting a neutron reflectometry dataset
=======================================


This example analyses a neutron reflectometry dataset, you can find this
example on `Github`_. Start off with relevant imports:

>>> import numpy as np
>>> from refnx.analysis import Objective, CurveFitter, Parameter, Transform
>>> from refnx.analysis import process_chain
>>> from refnx.reflect import Slab, SLD, ReflectModel
>>> from refnx.dataset import ReflectDataset

Load some data

>>> data = ReflectDataset('c_PLP0011859_q.txt')

Set up the SLD's for the model

>>> si = SLD(2.07, name='Si')
>>> sio2 = SLD(3.47, name='SiO2')
>>> film = SLD(2.0, name='film')
>>> d2o = SLD(6.36, name='d2o')

Create some slabs. The sio2 oxide layer is 30 Å thick with a 3 Å roughness.

>>> sio2_layer = sio2(30, 3)
>>> sio2_layer.thick.setp(bounds=(15, 50), vary=True)
>>> sio2_layer.rough.setp(bounds=(1, 15), vary=True)

>>> film_layer = film(250, 3)
>>> film_layer.thick.setp(bounds=(200, 300), vary=True)
>>> film_layer.sld.real.setp(bounds=(0.1, 3), vary=True)
>>> film_layer.rough.setp(bounds=(1, 15), vary=True)

>>> d2o_layer = d2o(0, 3)
>>> d2o_layer.rough.setp(vary=True, bounds=(1, 15))

Make a Structure from Slab components

>>> structure = si | sio2_layer | film_layer | d2o_layer

Setup a reflectivity model from the structure

>>> model = ReflectModel(structure, bkg=3e-6)
>>> model.scale.setp(bounds=(0.6, 1.2), vary=True)
>>> model.bkg.setp(bounds=(1e-9, 9e-6), vary=True)

An Objective requires a Model and a Data1D object. The transform kwd says that
we want to fit as logY vs X. Lets also plot it (requires matplotlib be
installed).

>>> objective = Objective(model, data, transform=Transform('logY'))
>>> fig, ax = objective.plot()

.. image:: _images/c_PLP0011859_q.png

Once we have an Objective we can create a CurveFitter. Do a quick fit with the
'differential_evolution' solver.

>>> fitter = CurveFitter(objective)
>>> fitter.fit('differential_evolution')

Now lets do some MCMC sampling with the CurveFitter object.

>>> fitter.sample(1000)

Before we can use the results we have to burn and thin to reduce correlation.

>>> process_chain(objective, fitter.chain, nburn=400, nthin=50)

Look at the parameters and plot again.

>>> print(objective)
>>> fig, ax = objective.plot()

.. image:: _images/fitted_c_PLP0011859_q.png

Visualise the covariance with a corner plot (requires the corner package be
installed)

>>> import corner
>>> corner.corner(fitter.sampler.flatchain)

.. image:: _images/corner_c_PLP0011859_q.png

