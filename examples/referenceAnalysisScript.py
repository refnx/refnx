import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from refnx.dataset import ReflectDataset
from refnx.analysis import CurveFitter, Objective, Transform
from refnx.reflect import ReflectModel, SLD

matplotlib.pyplot.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.pyplot.rcParams['figure.dpi'] = 600


DATASET_NAME = 'c_PLP0011859_q.txt'

# load the data
data = ReflectDataset(DATASET_NAME)

si = SLD(2.07, name='Si')
sio2 = SLD(3.47, name='SiO2')
film = SLD(2, name='film')
d2o = SLD(6.36, name='d2o')

structure = si | sio2(30, 3) | film(250, 3) | d2o(0, 3)
structure[1].thick.setp(vary=True, bounds=(15., 50.))
structure[1].rough.setp(vary=True, bounds=(1., 6.))
structure[2].thick.setp(vary=True, bounds=(200, 300))
structure[2].sld.real.setp(vary=True, bounds=(0.1, 3))
structure[2].rough.setp(vary=True, bounds=(1, 6))

model = ReflectModel(structure, bkg=9e-6, scale=1.)
model.bkg.setp(vary=True, bounds=(1e-8, 1e-5))
model.scale.setp(vary=True, bounds=(0.9, 1.1))

# fit on a logR scale, but use weighting
objective = Objective(model, data, transform=Transform('logY'),
                      use_weights=True)

# create the fit instance
fitter = CurveFitter(objective)

# do the fit
res = fitter.fit(method='differential_evolution')

# see the fit results
print(objective)

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.scatter(data.x, data.y, label=DATASET_NAME)
ax.semilogy()
ax.plot(data.x, model.model(data.x, x_err=data.x_err), label='fit')
plt.xlabel('Q')
plt.ylabel('logR')
plt.legend()
ax2 = fig.add_subplot(2, 1, 2)
z, rho_z = structure.sld_profile()
ax2.plot(z, rho_z)
