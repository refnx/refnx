from __future__ import division
import numpy as np
import math
from . import fitting
from . import reflect


class LinkageException(Exception):

    '''
        an exception that gets raised if the LinkageArray is not correct
    '''

    def __init__(self):
        pass


class GlobalFitObject(fitting.FitObject):

    '''

        Performs a global curvefitting analysis, subclassing the main fitting
        fitting.FitObject class. Global curvefitting analyses several datasets
        at the same time. It is not necessary to analyse each dataset with the
        same theoretical model, they can be different.  Usage of a linkageArray
        allows one to have common parameters between datasets.  This reduces
        the dimensionality of the problem.

        Here is an example linkageArray for a two dataset analysis (you can
        analyse a single dataset if you want to). The model for the first
        dataset has 8 parameters, the model for the second dataset has 10
        parameters.

        >>>linkageArray.shape
        (2,10)
        >>>linkageArray
        array([[ 0,  1,  2,  3,  4,  5,  2,  6, -1, -1],
               [ 0,  7,  8,  9,  2, 10, 11, 12, 13, 14]])

        The linkages in this array are:
           dataset0:parameter0 = dataset1:parameter0
           dataset0:parameter2 = dataset1:parameter4
           dataset0:parameter2 = dataset0:parameter6

        The -1 entries at the end of the first row are used as padding values.
        This is because the maximum number of model parameters is 10, so there
        are a total of 10 columns in the array.  Since the model for the first
        dataset only has 8 parameters the last two columns are padded with -1.
        It is important to note that the unique parameters are consecutively
        numbered. In this situation there are 15 unique parameters.

        The class is used as follows:
        1) setup fitting.FitObjects for each of the individual datasets, as if
        you were fitting them separately.  The individual FitObject is
        responsible for calculating the theoretical model data for that dataset.
        2) create the linkageArray.
        3) Initialise the GlobalFitObject
        4) Call GlobalFitObject.fit() (a super class method)

        The overall costmetric (default is Chi2) that is minimised is calculated
        by the fitter.FitObject.energy() method.  This in turn calls the
        GlobalFitObject.model() method to request evaluation of the theoretical
        model.  The GlobalFitObject class does the work of combining the
        theoretical model from each of the individual datasets to calculate an
        overall Chi2.
        The individual FitObjects are responsible for calculating the
        theoretical model for their own dataset. Individual de-interlaced
        parameter arrays are supplied to each FitObject.model() method by the
        GlobalFitObject.model() method, with the de-interlacing controlled by
        the linkageArray.  The individual FitObject can either be a subclass of
        FitObject that overrides the model() method, or it can be the FitObject
        class itself, with a fitfunction specified.

        If you wish to use a costmetric other that Chi2 you have a few options.
        You can supply a callable costfunction with the signature
        costfunction(model, ydata, edata, parameters) in the kwds.
        e.g.
        cost = lambda model, ydata, edata, parameters: (np.sum(((model -
                                                 ydata)/edata)**2))
        kwds['costfunction'] = cost
        -OR-
        subclass this GlobalFitObject and override the energy method.

        The initial parameters are drawn from the original FitObjects.
        The lower and upper limits for each parameter can be specified by a
        'limits' array in the kwds supplied to __init__. This limits array must
        have the shape (2, N), where the first row contains the lower limits,
        the second row the upper limits and N is the number of unique
        parameters.  If 'limits' is not in kwds, then individual limits are
        drawn from the original FitObjects.
        If there is a 'fitted_parameters' array in kwds supplied to __init__,
        then the fitting will vary the corresponding unique parameter during the
        fit. The maximum value in the fitted_parameter array must be N - 1,
        where N is the number of unique parameters. If the fitted_parameters
        array is not specified in kwds, then each individual FitObject is
        consulted to see if the parameter should be allowed to vary. Only unique
        parameters in the combined dataset are allowed to vary.
    '''

    def __init__(self, fitObjectTuple, linkageArray, args=(), **kwds):
        '''
        FitObjectTuple is a tuple of fitting.FitObject objects.
        The linkageArray specifies which parameters are common between datasets.
        '''

        self.linkageArray = np.atleast_2d(linkageArray)
        self.linkageArray = self.linkageArray.astype('int32')
        self.fitObjectTuple = fitObjectTuple

        self.is_linkage_array_corrupted()

        totalydata = np.concatenate(
            [fitObject.ydata for fitObject in fitObjectTuple])
        totaledata = np.concatenate(
            [fitObject.edata for fitObject in fitObjectTuple])
        totalparams = np.concatenate(
            [fitObject.parameters for fitObject in fitObjectTuple])
        self.FitObjectTuple = fitObjectTuple

        self.unique_pars, self.unique_pars_idx, self.unique_pars_inv = np.unique(
            self.linkageArray.astype('int32'),
            return_index=True,
            return_inverse=True)

        self.unique_pars_vector = totalparams[
            self.unique_pars_idx[self.unique_pars >= 0]]
        uniquelocs = self.unique_pars_idx[self.unique_pars >= 0]

        '''
            sort out which parameters are to be fitted.
            If you supply an np.array, fitted_parameters in kwds, then the code
            will use that. But, it will have to make sense compared to
            unique_pars_vector (i.e. no fitted parameter number >
            unique_pars_vector.size - 1.
            Alternatively it will fit the parameters listed in the individual
            fitObject.fitted_parameters arrays IFF they are unique parameters.
            Note that when you set up the individual fitObject if you don't
            supply the fitted_parameters keyword, then the default is to fit
            them all.
        '''
        if ('fitted_parameters' in kwds
             and kwds['fitted_parameters'] is not None):
            # if it's in kwds, then it'll get passed to the superclass
            # constructor
            pass
        else:
            # initiate fitted_parameters from the individual fitObjects
            fitted_parameters = np.array([], dtype='int32')

            for idx, pos in enumerate(uniquelocs):
                row = int(pos // np.size(self.linkageArray, 1))
                col = pos % (np.size(self.linkageArray, 1))
                if col in fitObjectTuple[row].fitted_parameters:
                    fitted_parameters = np.append(fitted_parameters, idx)

            kwds['fitted_parameters'] = fitted_parameters

        '''
        If you supply the limits array in kwds, then the code will use that. But
        it has to make sense with respect to the size of
        self.unique_pars_vector:
        The shape of limits should be limits.shape = (2, N)
        The shape of unique_pars_vector should be unique_pars_vector.shape = N
        In other words, each parameter has an upper and lower value.
        If the limits array is not supplied, then each parameter in
        unique_pars_vector will use the limits from the individual fitObject
        that it came from. When you setup the individual fitObject if you don't
        supply the limits keyword, then the default is 0 and 2 times the initial
        parametervalue.
        '''
        if 'limits' in kwds and kwds['limits'] is not None and np.size(kwds['limits'], 1) == self.unique_pars_vector.size:
            # self.limits gets setup in the superclass constructor
            pass
        else:
            # setup limits from individual fitObject
            limits = np.zeros((2, self.unique_pars_vector.size))

            for idx, pos in enumerate(uniquelocs):
                row = int(pos // np.size(self.linkageArray, 1))
                col = pos % (np.size(self.linkageArray, 1))
                limits[0, idx] = fitObjectTuple[row].limits[0, col]
                limits[1, idx] = fitObjectTuple[row].limits[1, col]

            kwds['limits'] = limits

        # initialise the FitObject superclass
        super(GlobalFitObject, self).__init__(None,
                                              totalydata,
                                              totaledata,
                                              None,
                                              self.unique_pars_vector,
                                              args=args,
                                              **kwds)

    def model(self, parameters=None, *args):
        '''
        calculate the model function for the global fit function.
        params is a np.array that has the same size as self.parameters
        '''

        if parameters is not None:
            test_parameters = parameters
        else:
            test_parameters = self.parameters

        substituted_pars = test_parameters[
            self.unique_pars[self.unique_pars_inv]]

        off = lambda idx: idx * np.size(self.linkageArray, 1)

        evaluateddata = [x.model(
            parameters=substituted_pars[off(i): off(i) + x.numparams],
            *args) for i,
            x in enumerate(self.fitObjectTuple)]

        return np.r_[evaluateddata].flatten()

    def fit(self, method=None):
        pars, uncertainty, chi2 = super(GlobalFitObject, self).fit(method)
        substituted_pars = pars[self.unique_pars[self.unique_pars_inv]]
        substituted_uncertainty = uncertainty[
            self.unique_pars[self.unique_pars_inv]]
        return substituted_pars, substituted_uncertainty, chi2

    def is_linkage_array_corrupted(self):
        '''
        Is the linkageArray corrupted?
        Although this is against the spirit of python, some testing here
        seems like a very good idea because the fitting process may accept
        garbage linkages.
        '''
        uniqueparam = -1
        for ii in xrange(np.size(self.linkageArray, 0)):
            for jj in xrange(np.size(self.linkageArray, 1)):
                val = self.linkageArray[ii, jj]
                if val < 0 and jj <= self.fitObjectTuple[ii].numparams - 1:
                    raise LinkageException
                if val > uniqueparam + 1:
                    raise LinkageException
                if self.linkageArray[ii, jj] < -1:
                    raise LinkageException
                if val == uniqueparam + 1:
                    uniqueparam += 1
