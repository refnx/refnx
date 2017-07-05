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
#include <cstdio>

#ifdef _WIN32
    #include <windows.h>
    #include <process.h>
#endif
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    #include <unistd.h>
    #include <pthread.h>
    #define HAVE_PTHREAD_H
#endif

#ifdef _OPENMP
   #include <omp.h>
#endif

#define NUM_CPUS 4
#define PI 3.141592653589793

using namespace MyComplexNumber;

#ifdef __cplusplus
extern "C" {
#endif


void* malloc2d(int ii, int jj, int sz){
    void** p;
    size_t sz_ptr_array;
    size_t sz_elt_array;
    size_t sz_allocation;
    long i = 0;
    char *c = NULL;

    sz_ptr_array = ii * sizeof(void*);
    sz_elt_array = jj * sz;
    sz_allocation = sz_ptr_array + ii * sz_elt_array;

    p = (void**) malloc(sz_allocation);
    if (p == NULL)
        return p;
    memset(p, 0, sz_allocation);

    c = ((char*) p) + sz_ptr_array;
    for (i = 0; i < ii; ++i)
    {
        //*(p+i) = (void*) ((long)p + sz_ptr_array + i * sz_elt_array);
        p[i] = (void*) (c + i * sz_elt_array);
    }
    return p;
}


void AbelesCalc_ImagAll(int numcoefs,
                        const double *coefP,
                        int npoints,
                        double *yP,
                        const double *xP,
                        int workers){
        int j;
        double scale, bkg;
        double num = 0, den = 0, answer = 0;

        MyComplex super;
        MyComplex sub;
        MyComplex oneC = MyComplex(1, 0);
        MyComplex MRtotal[2][2];
        MyComplex MI[2][2];
        MyComplex temp2[2][2];
        MyComplex qq2;
        MyComplex *SLD = NULL;
        double *thickness = NULL;
        double *rough_sqr = NULL;

        int nlayers = (int) coefP[0];

        try{
//		    // 2D array to hold wavevectors for each point, kn[npoints][nlayers + 2]
//		    kn_all = (MyComplex **) malloc2d(npoints, nlayers + 2, sizeof(MyComplex));
//		    if(kn_all == NULL)
//		        goto done;

            SLD = new MyComplex[nlayers + 2];
            thickness = new double[nlayers];
            rough_sqr = new double[nlayers + 1];
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
            rough_sqr[ii - 1] = -2 * coefP[4 * ii + 7] * coefP[4 * ii + 7];
        }

        SLD[0] = MyComplex(0, 0);
        SLD[nlayers + 1] = 4 * PI * (sub - super);
        rough_sqr[nlayers] = -2 * coefP[7] * coefP[7];

// if you have omp.h, then can do the calculation in parallel.
#ifdef _OPENMP
        omp_set_num_threads(workers);
        #pragma omp parallel for shared(kn_all) private(j, num, den, answer, qq2, MRtotal, MI, temp2)
#endif

        for (j = 0; j < npoints; j++) {
            MyComplex beta, rj;
            MyComplex kn, kn_next;

            qq2 = MyComplex(xP[j] * xP[j] / 4, 0);

            // now calculate reflectivities and wavevectors
            kn = compsqrt(qq2 - SLD[0]);
            for(int ii = 0 ; ii < nlayers + 1 ; ii++){
                // wavevector in the layer
                kn_next = compsqrt(qq2 - SLD[ii + 1]);

                // reflectance of the interface
                rj = (kn - kn_next)/(kn + kn_next)
                      * compexp(kn * kn_next * rough_sqr[ii]) ;

                if (!ii){
                    // characteristic matrix for first interface
                    MRtotal[0][0] = oneC;
                    MRtotal[0][1] = rj;
                    MRtotal[1][1] = oneC;
                    MRtotal[1][0] = rj;
                } else {
                    // work out the beta for the layer
                    beta = compexp(kn * MyComplex(0, thickness[ii - 1]));
                    // this is the characteristic matrix of a layer
                    MI[0][0] = beta;
                    MI[0][1] = rj * beta;
                    MI[1][1] = oneC / beta;
                    MI[1][0] = rj * MI[1][1];

                    // multiply MRtotal, MI to get the updated total matrix.
                    memcpy(temp2, MRtotal, sizeof(MRtotal));
                    matmul(temp2, MI, MRtotal);
                }
                    kn = kn_next;
            }

            num = compnorm(MRtotal[1][0]);
            den = compnorm(MRtotal[0][0]);
            answer = (num / den);
            answer = (answer * scale) + fabs(bkg);

            yP[j] = answer;
        }

    done:
        if(SLD)
            delete[] SLD;
        if(thickness)
            delete[] thickness;
        if(rough_sqr)
            delete[] rough_sqr;
    }

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

/* pthread version */
#ifdef HAVE_PTHREAD_H

    void *ThreadWorker(void *arg){
        pointCalcParm *p = (pointCalcParm *) arg;
        AbelesCalc_ImagAll(p->numcoefs,
                           p->coefP,
                           p->npoints,
                           p->yP,
                           p->xP,
                           0);
        pthread_exit((void*)0);
        return NULL;
    }

    void AbelesCalc_Imag(int numcoefs,
                          const double *coefP,
                           int npoints,
                            double *yP,
                             const double *xP,
                              int workers){

        pthread_t *threads = NULL;
        pointCalcParm *arg = NULL;

        int threadsToCreate = workers - 1;
        int pointsEachThread, pointsRemaining, pointsConsumed;

        // create threads for the calculation
        threads = (pthread_t *) malloc((threadsToCreate) * sizeof(pthread_t));
        if(!threads && workers > 1)
            goto done;

        // create arguments to be supplied to each of the threads
        arg = (pointCalcParm *) malloc(sizeof(pointCalcParm)
                                       * (threadsToCreate));
        if(!arg && workers > 1)
            goto done;

        // need to calculated how many points are given to each thread.
        if(threadsToCreate > 0){
            pointsEachThread = floorl(npoints / (threadsToCreate + 1));
        } else {
            pointsEachThread = npoints;
        }

        pointsRemaining = npoints;
        pointsConsumed = 0;

        // if you have two CPU's, only create one extra thread because the main
        // thread does half the work
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
        // do the last points in the main thread.
        AbelesCalc_ImagAll(numcoefs, coefP, pointsRemaining, yP + pointsConsumed, xP + pointsConsumed, 0);

        for (int ii = 0; ii < threadsToCreate ; ii++)
            pthread_join(threads[ii], NULL);

    done:
        if(threads)
            free(threads);
        if(arg)
            free(arg);
    }
#endif

#ifdef _WIN32
    unsigned int __stdcall ThreadWorker(void *arg){
            pointCalcParm *p = (pointCalcParm *) arg;
            AbelesCalc_ImagAll(p->numcoefs,
                               p->coefP,
                               p->npoints,
                               p->yP,
                               p->xP,
                               0);
            return 0;
    }

    void AbelesCalc_Imag(int numcoefs,
                          const double *coefP,
                           int npoints,
                            double *yP,
                             const double *xP,
                              int workers){

        uintptr_t *threads = NULL;
        pointCalcParm *arg = NULL;

        int threadsToCreate = workers - 1;
        int pointsEachThread, pointsRemaining, pointsConsumed;

        // create threads for the calculation
        threads = (uintptr_t *) malloc((threadsToCreate) * sizeof(uintptr_t));
        if(!threads && workers > 1)
            goto done;

        // create arguments to be supplied to each of the threads
        arg = (pointCalcParm *) malloc(sizeof(pointCalcParm)
                                           * (threadsToCreate));
        if(!arg && workers > 1)
            goto done;

        // need to calculated how many points are given to each thread.
        if(threadsToCreate > 0){
            pointsEachThread = (int) floor((double) npoints / ((double) threadsToCreate + 1));
        } else {
            pointsEachThread = npoints;
        }

        pointsRemaining = npoints;
        pointsConsumed = 0;

        for( int ii=0; ii<threadsToCreate; ii++ )
        {
            // if you have two CPU's, only create one extra thread because the main
            // thread does half the work
            arg[ii].coefP = coefP;
            arg[ii].numcoefs = numcoefs;

            arg[ii].npoints = pointsEachThread;

            //the following two lines specify where the Q values and R values
            //i.e. an offset of the original array.
            arg[ii].xP = xP + pointsConsumed;
            arg[ii].yP = yP + pointsConsumed;

            threads[ii] = _beginthreadex(
                                         NULL,                   // default security attributes
                                         0,                      // use default stack size
                                         ThreadWorker,           // thread function name
                                         (void*) &arg[ii],        // argument to thread function
                                         0,                      // use default creation flags
                                         NULL);                  // returns the thread identifier

            if(threads[ii] == 0)
                goto done;

            pointsRemaining -= pointsEachThread;
            pointsConsumed += pointsEachThread;
        }
        // do the last points in the main thread.
        AbelesCalc_ImagAll(numcoefs, coefP, pointsRemaining, yP + pointsConsumed, xP + pointsConsumed, 0);

        // Wait until all threads have terminated.
        WaitForMultipleObjects(threadsToCreate, (HANDLE *)threads, TRUE, INFINITE);

        done:
            if(threads)
                // Close all thread handles and free memory allocations.
                for(int ii=0; ii<threadsToCreate; ii++){
                    if(threads[ii])
                        CloseHandle((HANDLE) threads[ii]);
                }

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
               const double *xP,
               int threads){
/*
choose between the mode of calculation, depending on whether pthreads or omp.h
is present for parallelisation.
*/
#if defined HAVE_PTHREAD_H
    AbelesCalc_Imag(numcoefs, coefP, npoints, yP, xP, threads);
#elif defined _OPENMP
    AbelesCalc_ImagAll(numcoefs, coefP, npoints, yP, xP, threads);
#elif defined _WIN32
    AbelesCalc_Imag(numcoefs, coefP, npoints, yP, xP, threads);
#else
    AbelesCalc_ImagAll(numcoefs, coefP, npoints, yP, xP, 0);
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
    AbelesCalc_ImagAll(numcoefs, coefP, npoints, yP, xP, 0);
}

#ifdef __cplusplus
    }
#endif
