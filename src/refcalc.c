/*
    refcalc.c

    *Calculates the specular (Neutron or X-ray) reflectivity from a stratified
    series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, Australian Nuclear Science and Technology Organisation

Permission to use and redistribute the source code or binary forms of this
software and its documentation, with or without modification is hereby
granted provided that the above notice of copyright, these terms of use,
and the disclaimer of warranty below appear in the source code and
documentation, and that none of the names of above institutions or
authors appear in advertising or endorsement of works derived from this
software without specific prior written permission from all parties.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THIS SOFTWARE.

*/

/*
Efforts to speed this code up have included:

1) Use <complex> using valarrays for vectorisation.
2) Use Strassen approach to matrix multiplication.
https://www.johndcook.com/blog/2018/08/31/how-fast-can-you-multiply-matrices/
3) Experiments have found that c++ <complex> is about 10-20% slower than C99
   double complex. C99 complex doesn't work on Windows.

However, the following remains the fastest calculation  so far.
*/


#include "stdlib.h"
#include "complex.h"
#include "tgmath.h"
#include "string.h"

#define NUM_CPUS 4
#define PI 3.14159265358979323846
// TINY is required to make sure a complex sqrt takes the correct branch
// if you choose too small a number for tiny then the complex square root
// takes a lot longer.
#define TINY 1e-30

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CMPLX
#define CMPLX(x, y)  ((double complex)((double)(x) + \
                            _Complex_I * (double)(y)))
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

void
matmul(_Complex double a[2][2], _Complex double b[2][2], _Complex double c[2][2]){
    c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0];
    c[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1];
    c[1][0]	= a[1][0]*b[0][0] + a[1][1]*b[1][0];
    c[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1];
}


void abeles(int numcoefs,
            const double* restrict coefP,
            int npoints,
            double* restrict yP,
            const double* restrict xP){
    int j;
    double scale, bkg;
    double num = 0, den = 0, answer = 0;

    double complex super;
    double complex sub;
    double complex _t;
    double complex oneC = CMPLX(1., 0.);
    _Complex double MRtotal[2][2];
    _Complex double MI[2][2];
    _Complex double temp2[2][2];
    _Complex double *SLD = NULL;
    _Complex double *thickness = NULL;
    double complex qq2;
    double *rough_sqr = NULL;

    int nlayers = (int) coefP[0];

    SLD = (_Complex double *) malloc ((nlayers + 2) * sizeof(_Complex double));
    if (!SLD)
        goto done;

    thickness = (_Complex double *) malloc ((nlayers) * sizeof(_Complex double));
    if (!thickness)
        goto done;

    rough_sqr = (double *) malloc ((nlayers + 1) * sizeof(double));
    if (!rough_sqr)
        goto done;

    scale = coefP[1];
    bkg = coefP[6];
    sub = CMPLX(coefP[4], fabs(coefP[5]) + TINY);
    super = CMPLX(coefP[2], 0);

    // fill out all the SLD's for all the layers
    for(int ii = 1; ii < nlayers + 1; ii += 1){
        _t = CMPLX(coefP[4 * ii + 5], fabs(coefP[4 * ii + 6]) + TINY);
        SLD[ii] = 4e-6 * PI * (_t - super);

        thickness[ii - 1] = CMPLX(0, fabs(coefP[4 * ii + 4]));
        rough_sqr[ii - 1] = -2 * coefP[4 * ii + 7] * coefP[4 * ii + 7];
    }

    SLD[0] = CMPLX(0, 0);
    SLD[nlayers + 1] = 4e-6 * PI * (sub - super);
    rough_sqr[nlayers] = -2 * coefP[7] * coefP[7];

    for (j = 0; j < npoints; j++) {
        double complex beta, rj;
        double complex kn, kn_next;

        qq2 = CMPLX(xP[j] * xP[j] / 4, 0);

        // now calculate reflectivities and wavevectors
        kn = xP[j] / 2.;
        for(int ii = 0 ; ii < nlayers + 1 ; ii++){
            // wavevector in the layer
            kn_next = csqrt(qq2 - SLD[ii + 1]);

            // reflectance of the interface
            rj = (kn - kn_next)/(kn + kn_next)
                  * cexp(kn * kn_next * rough_sqr[ii]) ;

            if (!ii){
                // characteristic matrix for first interface
                MRtotal[0][0] = oneC;
                MRtotal[0][1] = rj;
                MRtotal[1][1] = oneC;
                MRtotal[1][0] = rj;
            } else {
                // work out the beta for the layer
                beta = cexp(kn * thickness[ii - 1]);
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

        num = cabs(MRtotal[1][0]);
        num *= num;
        den = cabs(MRtotal[0][0]);
        den *= den;
        answer = (num / den);
        answer = (answer * scale) + bkg;

        yP[j] = answer;
    }

    done:
        if(SLD)
            free(SLD);
        if(thickness)
            free(thickness);
        if(rough_sqr)
            free(rough_sqr);
    }


void parratt(int numcoefs,
             const double* restrict coefP,
             int npoints,
             double* restrict yP,
             const double* restrict xP){
    int j;
    double scale, bkg;

    double complex super;
    double complex sub;
    double complex _t;
    _Complex double *SLD = NULL;
    _Complex double *thickness = NULL;
    double complex qq2;
    double *rough_sqr = NULL;

    int nlayers = (int) coefP[0];

    SLD = (_Complex double *) malloc ((nlayers + 2) * sizeof(_Complex double));
    if (!SLD)
        goto done;

    thickness = (_Complex double *) malloc ((nlayers) * sizeof(_Complex double));
    if (!thickness)
        goto done;

    rough_sqr = (double *) malloc ((nlayers + 1) * sizeof(double));
    if (!rough_sqr)
        goto done;

    scale = coefP[1];
    bkg = coefP[6];
    sub = CMPLX(coefP[4], fabs(coefP[5]) + TINY);
    super = CMPLX(coefP[2], 0);

    // fill out all the SLD's for all the layers
    for(int ii = 1; ii < nlayers + 1; ii += 1){
        _t = CMPLX(coefP[4 * ii + 5], fabs(coefP[4 * ii + 6]) + TINY);
        SLD[ii] = 4e-6 * PI * (_t - super);

        thickness[ii - 1] = CMPLX(0, -2.0 * fabs(coefP[4 * ii + 4]));
        rough_sqr[ii - 1] = -2 * coefP[4 * ii + 7] * coefP[4 * ii + 7];
    }

    SLD[0] = CMPLX(0, 0);
    SLD[nlayers + 1] = 4e-6 * PI * (sub - super);
    rough_sqr[nlayers] = -2 * coefP[7] * coefP[7];

    for (j = 0; j < npoints; j++) {
        double complex beta, rj;
        double complex kn, kn_next, RRJ_1, RRJ;
        RRJ = CMPLX(0, 0);
        RRJ_1 = CMPLX(0, 0);

        qq2 = CMPLX(xP[j] * xP[j] / 4, 0);

        // now calculate reflectivities and wavevectors

        // start from subphase
		kn_next = csqrt(qq2 - SLD[nlayers + 1]);

        for(int ii = nlayers ; ii > -1 ; ii--){
            // wavevector in the layer
            kn = csqrt(qq2 - SLD[ii]);

            // reflectance of the interface
            // factor of 2 is already incorporated in rough_sqr
            rj = (kn - kn_next)/(kn + kn_next)
                  * cexp(kn * kn_next * rough_sqr[ii]);

            if (ii == nlayers){
                // characteristic matrix for first interface
                RRJ = rj;
            } else {
                // factor of 2i is alread incorporated in thickness
                beta = cexp(kn_next * thickness[ii]);
                RRJ = (rj + RRJ_1 * beta) / (1 + RRJ_1 * beta * rj);
            }
            kn_next = kn;
            RRJ_1 = RRJ;
        }
        yP[j] = (creal(RRJ * conj(RRJ)) * scale) + bkg;
    }

    done:
        if(SLD)
            free(SLD);
        if(thickness)
            free(thickness);
        if(rough_sqr)
            free(rough_sqr);
    }


#ifdef __cplusplus
    }
#endif
