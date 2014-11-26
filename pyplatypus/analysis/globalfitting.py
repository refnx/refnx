from __future__ import division
import numpy as np
import math
from . import curvefitter


class LinkageException(Exception):

    '''
        an exception that gets raised if the linkage_array is not correct
    '''

    def __init__(self):
        pass


class GlobalFitter(curvefitter.CurveFitter):

    '''

        Performs a global curvefitting analysis, subclassing the main fitting
        fitting.FitObject class. Global curvefitting analyses several datasets
        at the same time. It is not necessary to analyse each dataset with the
        same theoretical model, they can be different.  Usage of a linkage_array
        allows one to have common parameters between datasets.  This reduces
        the dimensionality of the problem.

        Here is an example linkage_array for a two dataset analysis (you can
        analyse a single dataset if you want to). The model for the first
        dataset has 8 parameters, the model for the second dataset has 10
        parameters.

        >>>linkage_array.shape
        (2,10)
        >>>linkage_array
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
        2) create the linkage_array.
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
        the linkage_array.  The individual FitObject can either be a subclass of
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

    def __init__(self, fitter_objects, linkage_array, args=(), **kwds):
        '''
        fitter_objects is a tuple of CurveFitter objects.
        The linkage_array specifies which parameters are common between datasets.
        '''

        self.linkage_array = np.atleast_2d(linkage_array)
        self.linkage_array = self.linkage_array.astype('int32')
        self.fitter_objects = fitter_objects

        self.is_linkage_array_corrupted()

        ydata_total = np.concatenate(
            [fitter_object.ydata.flatten() for fitter_object in fitter_objects])
        edata_total = np.concatenate(
            [fitter_object.edata.flatten() for fitter_object in fitter_objects])
        p_total = np.concatenate(
            [fitter_object.p for fitter_object in fitter_objects])

        self.fitter_objects = fitter_objects

        temp = np.unique(self.linkage_array.astype('int32'), return_index=True,
                         return_inverse=True)

        self.p_unique, self.p_unique_idx, self.p_unique_inv = temp

        uniquelocs = self.p_unique_idx[self.p_unique >= 0]

        self.p_unique_vals = p_total[uniquelocs]

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
        #which parameters are going to be held, figure out from the individual
        #fit_objects
        fitted_parameters = np.array([], dtype='int32')

        for idx, pos in enumerate(uniquelocs):
            row = int(pos // np.size(self.linkage_array, 1))
            col = pos % (np.size(self.linkage_array, 1))
            if col in fitter_objects[row].fitted_parameters:
                fitted_parameters = np.append(fitted_parameters, idx)

        p_held = np.setdiff1d(np.arange(uniquelocs.size), fitted_parameters)

        #setup bounds from individual fit_object
        #default limits are 0 and twice the parameter
        limits = np.zeros((2, self.p_unique_vals.size))
        limits[1, :] = 2 * self.p_unique_vals

        for idx, pos in enumerate(uniquelocs):
            row = int(pos // np.size(self.linkage_array, 1))
            if fitter_objects[row].bounds is not None:
                col = pos % (np.size(self.linkage_array, 1))

                limits[0, idx] = fitter_objects[row].bounds[col][0]
                limits[1, idx] = fitter_objects[row].bounds[col][1]

        bounds = [(low, high) for low, high in zip(limits[0], limits[1])]

        # initialise the FitObject superclass
        super(GlobalFitter, self).__init__(None,
                                           ydata_total,
                                           None,
                                           self.p_unique_vals,
                                           edata=edata_total,
                                           p_held=p_held,
                                           args=args,
                                           bounds=bounds,
                                           **kwds)

    def model(self, p, *args, **kwds):
        '''
        calculate the model function for the global fit function.
        params is a np.array that has the same size as self.parameters
        '''

        p_individuals = self._p_substituted(p)

        model_data = [x.model(p_individuals[i], *x.args, **x.kwds).flatten()
                      for i, x in enumerate(self.fitter_objects)]

        return np.r_[model_data]

    def _p_substituted(self, p):
        '''
        transform collective parameter set into individual parameter sets
        '''
        p_substituted = p[self.p_unique[self.p_unique_inv]]
        off = lambda idx: idx * np.size(self.linkage_array, 1)

        individuals = [p_substituted[off(i): off(i) + fo.nparams]
                       for i, fo in enumerate(self.fitter_objects)]
        return individuals

    def fit(self, method='leastsq', minimizer_kwds=None):
        fit_result = super(GlobalFitter, self).fit(method=method,
                                         minimizer_kwds=minimizer_kwds)
        p_individuals = self._p_substituted(fit_result.p)

        return p_individuals

    def is_linkage_array_corrupted(self):
        '''
        Is the linkage_array corrupted?
        Although this is against the spirit of python, some testing here
        seems like a very good idea because the fitting process may accept
        garbage linkages.
        '''
        uniqueparam = -1
        for i in range(np.size(self.linkage_array, 0)):
            for j in range(np.size(self.linkage_array, 1)):
                val = self.linkage_array[i, j]
                if val < 0 and j <= self.fitter_objects[i].numparams - 1:
                    raise LinkageException
                if val > uniqueparam + 1:
                    raise LinkageException
                if self.linkage_array[i, j] < -1:
                    raise LinkageException
                if val == uniqueparam + 1:
                    uniqueparam += 1
