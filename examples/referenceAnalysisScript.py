from __future__ import print_function, division
from refnx.dataset.reflectdataset import ReflectDataset
import refnx.analysis.reflect as reflect
import refnx.analysis.curvefitter as curvefitter
from lmfit.printfuncs import fit_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.pyplot.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.pyplot.rcParams['figure.dpi'] = 600


DATASET_NAME = 'c_PLP0011859_q.txt'
USE_DIFFERENTIAL_EVOLUTION = True

#load the data
data = ReflectDataset(DATASET_NAME)

# create an array to hold the parameters, also create limits
layers = np.array([[0,   2.07, 0, 0],     # fronting medium
                   [30,  3.47, 0, 3],     # 1st layer
                   [250, 2.00, 0, 3],     # 2nd layer
                   [0,   6.36, 0, 3]])     # backing medium

lowlim = np.array([[0,   2.07, 0, 0],     # fronting medium
                   [15,  3.47, 0, 1],     # 1st layer
                   [200, 0.10, 0, 1],     # 2nd layer
                   [0,   6.36, 0, 1]])     # backing medium

hilim = np.array([[0,   2.07, 0, 0],     # fronting medium
                  [50,  3.47, 0, 6],     # 1st layer
                  [300, 3.00, 0, 6],     # 2nd layer
                  [0,   6.36, 0, 6]])     # backing medium

# create a linear array of the parameters
# coefs[1] is the scale factor
# coefs[6] is the background
# these will both be 1 and 0 respectively to start off with
coefs = reflect.layer_to_coefs(layers, scale=1.0, bkg=3e-6)
lowlim = reflect.layer_to_coefs(lowlim, scale=0.9, bkg=0)
hilim = reflect.layer_to_coefs(hilim, scale=11, bkg=9e-6)

bounds = zip(lowlim, hilim)

# create a parameter instance
parameters = curvefitter.to_parameters(coefs, bounds=bounds, varies=[False] * 16)

# which parameters do you want to allow to vary
fitted_parameters = np.array([1, 6, 7, 8, 11, 12, 13, 15])
for fit in fitted_parameters:
    parameters['p%d' % fit].vary = True

# use resolution smearing and fit on a logR scale (transform the data as well)
t = reflect.Transform('logY').transform
ydata, dydata = t(xdata, ydata, dydata)
kwds = {'dqvals': dxdata, 'transform': t}

# create the fit instance
fitter = reflect.ReflectivityFitFunction(parameters, xdata, ydata, edata=dydata,
                                    kwds=kwds)

#do the fit
if USE_DIFFERENTIAL_EVOLUTION:
    fitter.fit(method='differential_evolution')

fitter.fit()

print('-------------------------------------------------------------------')
print(DATASET_NAME)
print(fit_report(fitter))

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.scatter(xdata, ydata, label=DATASET_NAME)
ax.plot(xdata, fitter.model(parameters), label='fit')
plt.ylim(min(np.min(ydata), np.min(fitter.model(parameters))),
         max(np.max(ydata), np.max(fitter.model(parameters))))
plt.xlim(np.min(xdata), np.max(xdata))
plt.xlabel('Q')
plt.ylabel('logR')
plt.legend()
ax2 = fig.add_subplot(2, 1, 2)
z, rho_z = fitter.sld_profile(parameters)
ax2.plot(z, rho_z)