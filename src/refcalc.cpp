/*
    refcalc.h

    *Calculates the specular (Neutron or X-ray) reflectivity from a stratified
    series of layers.
    @Copyright, Andrew Nelson 2014.
 */

#include "refcalc.h"
#include <math.h>
#include "MyComplex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <assert.h>

#ifdef _WIN32

#else
#include <unistd.h>
#include <pthread.h>
#define HAVE_PTHREAD_H
#endif

#ifdef _OPENMP
   #include <omp.h>
#endif

#define NUM_CPUS 4
#define PI 3.141592653589793

using namespace std;
using namespace MyComplexNumber;

#ifdef __cplusplus
extern "C" {
#endif

void AbelesCalc_ImagAll(int numcoefs,
                        const double *coefP,
                        int npoints,
                        double *yP,
                        const double *xP){
		int j;
		double scale, bkg;
		double num = 0, den = 0, answer = 0;

		MyComplex super;
		MyComplex sub;
		MyComplex oneC = MyComplex(1,0);
		MyComplex MRtotal[2][2];
		MyComplex MI[2][2];
		MyComplex temp2[2][2];
		MyComplex qq2;
		MyComplex *kn = NULL;
		MyComplex *SLD = NULL;
		double *thickness = NULL;
		double *roughness = NULL;

		int nlayers = (int) coefP[0];

		try{
			kn = new MyComplex[nlayers + 2];
			SLD = new MyComplex[nlayers + 2];
			thickness = new double[nlayers];
			roughness = new double[nlayers + 1];
		} catch(...) {
			goto done;
        }

		scale = coefP[1];
		bkg = coefP[6];
		sub = MyComplex(coefP[4] * 1.e-6, coefP[5] * 1.e-6);
		super = MyComplex(coefP[2] * 1e-6, coefP[3] * 1.e-6);

		// fillout all the SLD's for all the layers
		for(int ii = 1; ii < nlayers + 1; ii += 1){
			SLD[ii] = 4 * PI * (MyComplex(coefP[4 * ii + 5] * 1.e-6,
			                              coefP[4 * ii + 6] * 1.e-6) - super);
			thickness[ii - 1] = fabs(coefP[4 * ii + 4]);
			roughness[ii - 1] = fabs(coefP[4 * ii + 7]);
        }

		SLD[0] = MyComplex(0, 0);
		SLD[nlayers + 1] = 4 * PI * (sub - super);
        roughness[nlayers] = fabs(coefP[7]);

// if you have omp.h, then can do the calculation in parallel.
#ifdef _OPENMP
#pragma omp parallel for shared(kn) private(j)
#endif

		for (j = 0; j < npoints; j++) {
			MyComplex beta, rj;

			qq2 = MyComplex(xP[j] * xP[j] / 4, 0);

			// work out the wavevector in each of the layers
			for(int ii = 0; ii < nlayers + 2 ; ii++)
				kn[ii] = compsqrt(qq2 - SLD[ii]);

			//now calculate reflectivities
			for(int ii = 0 ; ii < nlayers + 1 ; ii++){
			    rj = ((kn[ii] - kn[ii + 1])/(kn[ii] + kn[ii + 1]))
			          * compexp(kn[ii] * kn[ii + 1] * -2.
			          * roughness[ii] * roughness[ii]) ;

				// work out the beta for the layer
				beta = (ii == 0)? oneC
				                  :
				                  compexp(kn[ii]
				                          * MyComplex(0, thickness[ii - 1]));

				// this is the characteristic matrix of a layer
				MI[0][0] = beta;
				MI[0][1] = rj * beta;
				MI[1][1] = oneC / beta;
				MI[1][0] = rj * MI[1][1];

                if(!ii){
                    memcpy(MRtotal, MI, sizeof(MRtotal));
                } else {
                    memcpy(temp2, MRtotal, sizeof(MRtotal));

                    // multiply MRtotal, MI to get the updated total matrix.
                    matmul(temp2, MI, MRtotal);
                }
			}

			den = compnorm(MRtotal[0][0]);
			num = compnorm(MRtotal[1][0]);
			answer = (num / den);
			answer = (answer * scale) + fabs(bkg);

			yP[j] = answer;
		}

#ifdef _OPENMP
#pragma omp join
#endif

	done:
		if(kn)
			delete [] kn;
		if(SLD)
			delete[] SLD;
        if(thickness)
			delete[] thickness;
        if(roughness)
			delete[] roughness;
	}

/* pthread version*/
#ifdef HAVE_PTHREAD_H

	typedef struct{
		// a double array containing the model coefficients
		const double *coefP;
		// number of coefficients
		int numcoefs;
		// number of Q points we have to calculate
		int npoints;
		// the Reflectivity values to return
		double *yP;
		// the Q values to do the calculation for.
		const double *xP;
	}  pointCalcParm;

	void *ThreadWorker(void *arg){
	    int err = NULL;
		pointCalcParm *p = (pointCalcParm *) arg;
        AbelesCalc_ImagAll(p->numcoefs,
                           p->coefP,
                           p->npoints,
                           p->yP,
                           p->xP);
		pthread_exit((void*)err);
		return NULL;
	}

	void AbelesCalc_Imag(int numcoefs,
	                      const double *coefP,
	                       int npoints,
	                        double *yP,
	                         const double *xP){
		int err = 0;

		pthread_t *threads = NULL;
		pointCalcParm *arg = NULL;

		int threadsToCreate = NUM_CPUS - 1;
		int pointsEachThread, pointsRemaining, pointsConsumed;

		// create threads for the calculation
		threads = (pthread_t *) malloc((threadsToCreate) * sizeof(pthread_t));
		if(!threads && NUM_CPUS > 1){
			err = 1;
			goto done;
		}

		//create arguments to be supplied to each of the threads
		arg = (pointCalcParm *) malloc(sizeof(pointCalcParm)
		                               * (threadsToCreate));
		if(!arg && NUM_CPUS > 1){
			err = 1;
			goto done;
		}

		//need to calculated how many points are given to each thread.
		if(threadsToCreate > 0){
			pointsEachThread = floorl(npoints / (threadsToCreate + 1));
		} else {
			pointsEachThread = npoints;
		}

		pointsRemaining = npoints;
		pointsConsumed = 0;

		//if you have two CPU's, only create one extra thread because the main
		//thread does half the work
		for (int ii = 0; ii < threadsToCreate ; ii++){
			arg[ii].coefP = coefP;
			arg[ii].numcoefs = numcoefs;

			arg[ii].npoints = pointsEachThread;

			//the following two lines specify where the Q values and R values
			//i.e. an offset of the original array.
			arg[ii].xP = xP + pointsConsumed;
			arg[ii].yP = yP + pointsConsumed;

			pthread_create(&threads[ii], NULL, ThreadWorker,
			               (void *)(arg + ii));
			pointsRemaining -= pointsEachThread;
			pointsConsumed += pointsEachThread;
		}
		//do the last points in the main thread.
		AbelesCalc_ImagAll(numcoefs, coefP, pointsRemaining, yP + pointsConsumed, xP + pointsConsumed);

		for (int ii = 0; ii < threadsToCreate ; ii++)
			pthread_join(threads[ii], NULL);


	done:
		if(threads)
			free(threads);
		if(arg)
			free(arg);
	}
#endif

/*
Parallelised version
*/
void reflectMT(int numcoefs,
            const double *coefP,
            int npoints,
            double *yP,
            const double *xP){
/*
choose between the mode of calculation, depending on whether pthreads or omp.h
is present for parallelisation.
*/
#ifdef HAVE_PTHREAD_H
    AbelesCalc_Imag(numcoefs, coefP, npoints, yP, xP);
#else
    AbelesCalc_ImagAll(numcoefs, coefP, npoints, yP, xP);
#endif
}

/*
Non parallelised version
*/
void reflect(int numcoefs,
            const double *coefP,
            int npoints,
            double *yP,
            const double *xP){
	AbelesCalc_ImagAll(numcoefs, coefP, npoints, yP, xP);
}

#ifdef __cplusplus
	}
#endif
