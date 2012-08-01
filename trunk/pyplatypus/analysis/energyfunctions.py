from __future__ import division
import numpy as np
import math
import DEsolver

def energy_for_fitting(params, *args):
	''' 
		energy function for curve fitting.
		This energy function should work with DEsolver as well as the scipy.optimize modules.
		
		params are the parameters you are fitting.
		
		We have to pass in data, etc, through args. This args is passed through the optimize modules directly to 
		this energy function.
		
		The first argument in args should be a FitObject instance. This will have a method energy.

	'''
	
	return args[0].energy(params)
	

class FitObject(object):
	'''
		
		An object used to perform a curvefitting analysis.
				
		
	'''
	
	def __init__(self, xdata, ydata, edata, fitfunction, parameters, *args, **kwds):
		'''
		
		Construction of the object initialises the data for the curve fit, but doesn't actually start it.
		
		xdata[numpoints] - np.ndarray that contains the independent variables for the fit. Should have [numpoints] rows.
							Add extra columns for multi dimensional fitting (the fit function you want to use
							should be designed to cope with this).
							
		ydata[numpoints] - np.ndarray that contains the observations corresponding to each measurement point.
		
		edata[numpoints] - np.ndarray that contains the uncertainty (s.d.) for each of the observed y data points.
		
		fitfunction - callable function  of the form f(xdata, parameters, *args, **kwds). The args and kwds supplied
						in the construction of this object are passed directly to the fitfunction and should be used to pass
						auxillary information to it. You do can use None for fitfunction _IF_ you subclass this 
						FitObject and provide your own energy method.
						
		parameters - np.ndarray that contains _all_ the parameters to be supplied to the fitfunction, not just those being fitted
		
		You may set the following optional parameters in kwds:
		
		holdvector - an np.ndarray that is the same length as the number of fitting parameters. If an element in the array
					evaluates to True that parameter should be held.

		limits - an np.ndarray that contains the lower and upper limits for all the parameters.
				should have shape (2, numparams).

		
		'''
		self.xdata = np.copy(xdata)
		self.ydata = np.copy(ydata.flatten())
		self.edata = np.copy(edata.flatten())
		self.numpoints = np.size(ydata, 0)
		
		self.fitfunction = fitfunction
		self.parameters = np.copy(parameters)
		self.numparams = np.size(parameters, 0)
		self.args = args
		self.kwds = kwds
		
		if 'holdvector' in kwds and kwds['holdvector'] is not None:
			self.holdvector = np.copy(kwds['holdvector'])
			self.fitted_parameters = np.where(np.where(self.holdvector, False, True))[0]
		else:
			self.holdvector = np.zeros(self.numparams, 'int')
			self.fitted_parameters = np.arange(self.numparams)
			
		if 'limits' in kwds and kwds['limits'] is not None:
			self.limits = kwds['limits']
		else:
			self.limits = np.zeros((2, self.numparams))
			self.limits[0, :] = 0
			self.limits[1, :] = 2 * self.parameters
		
		#limits for those that are varying.
		self.fitted_limits = self.limits[:, self.fitted_parameters]
			
	def energy(self, params = None):
		'''
			
			The default cost function for the fit object is chi2 - the sum of the squared residuals divided by the error bars for each point.
			params - np.ndarray containing the parameters that are being fitted, i.e. this array is np.size(self.fitted_parameters) long.
					If this is omitted the energy function uses the defaults that we supplied when the object was constructed.
			
			If you require a different cost function provide a subclass that overloads the method.
			Returns chi2.
		
		'''
		test_parameters = np.copy(self.parameters)
	
		if params is not None:
			test_parameters[self.fitted_parameters] = params
		
		if self.fitfunction is not None:
			model = self.calc_model(test_parameters)
		else:
			raise Exception("You used the default FitObject.energy method, but did not specify a fitfunction")
			
		return  np.sum(np.power((self.ydata - model) / self.edata, 2))


	def calc_model(self, parameters = None):
		'''
			
			calculate the theoretical model using the fitfunction.
			
			parameters - the full np.ndarray containing the parameters that are required for the fitfunction
			
			returns the theoretical model for the xdata, i.e. self.fitfunction(self.xdata, test_parameters, *args, **kwds)
			
		'''	
				
		if parameters is not None:
			test_parameters = parameters
		else:
			test_parameters = self.parameters
			
		return self.fitfunction(self.xdata, test_parameters, *self.args, **self.kwds)

	def fit(self):
		de = DEsolver.DEsolver(self.fitted_limits, energy_for_fitting, self)
		thefit, chi2 = de.solve()
		self.parameters[self.fitted_parameters] = thefit
		return self.parameters, chi2
		