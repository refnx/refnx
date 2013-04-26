import pyplatypus.dataset.DataObject as data
import pyplatypus.analysis.reflect as reflect
import numpy as np


#load the data
dataset = data.DataObject(fname = '/Users/anz/Documents/Andy/programming/pyplatypus/pyplatypus/analysis/test/c_PLP0011859_q.txt')

#create an array to hold the parameters
#total number of parameters is 4N+8, where N is number of layers
coefs = np.zeros((16))

#fill out what the coefficients for the fit are
coefs[0] = 2
coefs[1] = 1.
coefs[2] = 2.07
coefs[3] = 0
coefs[4] = 6.36
coefs[5] = 0
coefs[6] = 2e-6
coefs[7] = 3
coefs[8] = 30
coefs[9] = 3.47
coefs[10] = 0
coefs[11] = 4
coefs[12] = 250
coefs[13] = 2
coefs[14] = 0
coefs[15] = 4

#which parameters do you want to allow to vary
fitted_parameters = np.array([6, 7,8,11,12,13,15])

#set up lower and upper limits for the fit
#you don't necessarily have to use these limits, default ones are calculated in the fit
#the defaults are 0 and 2 * initial parameter value
limits = np.zeros((2, 16))
limits[:, 6] = 0, 1e-5
limits[:, 7] = 1, 5
limits[:, 8] = 20, 40
limits[:, 11] = 1, 6
limits[:, 12] = 200, 300
limits[:, 13] = 1, 3
limits[:, 15] = 1, 6

fitobject = reflect.ReflectivityFitObject(dataset.W_q,
                                    dataset.W_ref,
                                     dataset.W_refSD,
                                        coefs,
                                         fitted_parameters = fitted_parameters,
                                          dqvals = dataset.W_qSD,
                                           limits = limits)
        
fittedvals, uncertainties, chi2 = fitobject.fit()

print fittedvals, uncertainties, chi2

