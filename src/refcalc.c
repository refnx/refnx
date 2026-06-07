/*
    refcalc.c

    *Calculates the specular (Neutron or X-ray) reflectivity from a stratified
    series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2015 A. R. J. Nelson, Australian Nuclear Science and Technology
Organisation

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

/*
 * Feature test macros must come before ALL includes.
 *
 * _DEFAULT_SOURCE: enables sincos() on glibc >= 2.19 (Linux/gcc)
 * _GNU_SOURCE:     fallback for older glibc; implies _DEFAULT_SOURCE
 * Neither is needed on macOS or MSVC, where we use alternatives anyway.
 */
#if defined(__linux__) || defined(__unix__)
#define _GNU_SOURCE
#endif

#include "assert.h"
#include "complex.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "tgmath.h"

/*
 * Portable sincos wrapper.
 * All paths result in a single combined sin/cos operation on
 * platforms that support it; MSVC fuses separate calls under /O2.
 */
static inline void sincos_portable(double angle, double *s, double *c) {
#if defined(_MSC_VER)
  *s = sin(angle);
  *c = cos(angle);
#elif defined(__APPLE__)
  __sincos(angle, s, c);
#elif defined(__GLIBC__) || defined(_DEFAULT_SOURCE) || defined(_GNU_SOURCE)
  sincos(angle, s, c);
#else
  /* Fallback: derive both from cexp — standard C99, no extensions needed */
  double complex e = cexp(CMPLX(0.0, angle));
  *c = creal(e);
  *s = cimag(e);
#endif
}

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
#define CMPLX(x, y) ((double complex)((double)(x) + _Complex_I * (double)(y)))
#endif

void *malloc2d(int ii, int jj, int sz) {
  void **p;
  size_t sz_ptr_array;
  size_t sz_elt_array;
  size_t sz_allocation;
  long i = 0;
  char *c = NULL;

  sz_ptr_array = ii * sizeof(void *);
  sz_elt_array = jj * sz;
  sz_allocation = sz_ptr_array + ii * sz_elt_array;

  p = (void **)malloc(sz_allocation);
  if (p == NULL)
    return p;
  memset(p, 0, sz_allocation);

  c = ((char *)p) + sz_ptr_array;
  for (i = 0; i < ii; ++i) {
    //*(p+i) = (void*) ((long)p + sz_ptr_array + i * sz_elt_array);
    p[i] = (void *)(c + i * sz_elt_array);
  }
  return p;
}

void matmul(_Complex double a[2][2], _Complex double b[2][2],
            _Complex double c[2][2]) {
  c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
  c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];
  c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
  c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
}

#define BATCH 4 /* process 4 Q-values at once; matches AVX2 double width */

void abeles(int numcoefs, const double *restrict coefP, int npoints,
            double *restrict yP, const double *restrict xP) {

  double scale, bkg;
  double complex super, sub, _t;

  _Complex double *SLD = NULL;
  _Complex double *thickness = NULL;
  double *rough_sqr = NULL;

  int nlayers = (int)coefP[0];

  SLD = (_Complex double *)malloc((nlayers + 2) * sizeof(_Complex double));
  if (!SLD)
    goto done;

  thickness = (_Complex double *)malloc(nlayers * sizeof(_Complex double));
  if (!thickness)
    goto done;

  rough_sqr = (double *)malloc((nlayers + 1) * sizeof(double));
  if (!rough_sqr)
    goto done;

  scale = coefP[1];
  bkg = coefP[6];
  sub = CMPLX(coefP[4], fabs(coefP[5]) + TINY);
  super = CMPLX(coefP[2], 0);

  for (int ii = 1; ii < nlayers + 1; ii++) {
    _t = CMPLX(coefP[4 * ii + 5], fabs(coefP[4 * ii + 6]) + TINY);
    SLD[ii] = 4e-6 * PI * (_t - super);
    thickness[ii - 1] = CMPLX(0, fabs(coefP[4 * ii + 4]));
    rough_sqr[ii - 1] = -2 * coefP[4 * ii + 7] * coefP[4 * ii + 7];
  }
  SLD[0] = CMPLX(0, 0);
  SLD[nlayers + 1] = 4e-6 * PI * (sub - super);
  rough_sqr[nlayers] = -2 * coefP[7] * coefP[7];

  /* ------------------------------------------------------------------ *
   * Batched Q-loop: process BATCH points simultaneously so the compiler
   * can vectorise the independent arithmetic across the batch dimension.
   *
   * Data layout (SoA within the batch):
   *   kn_re[b], kn_im[b]          -- real/imag parts of kn for lane b
   *   MRtotal_re[elem][b]          -- real parts of the 4 matrix elements
   *   MRtotal_im[elem][b]          -- imag parts of the 4 matrix elements
   *
   * Keeping real and imaginary parts in separate arrays (rather than
   * interleaved _Complex double) is what enables auto-vectorisation:
   * the compiler sees BATCH independent doubles in a contiguous array
   * and can map them onto a single SIMD register.
   * ------------------------------------------------------------------ */

  int j = 0;

  for (; j <= npoints - BATCH; j += BATCH) {

    double kn_re[BATCH], kn_im[BATCH];
    double qq2_re[BATCH]; /* add this — constant per Q-point */

    double M00_re[BATCH], M00_im[BATCH];
    double M01_re[BATCH], M01_im[BATCH];
    double M10_re[BATCH], M10_im[BATCH];
    double M11_re[BATCH], M11_im[BATCH];

    for (int b = 0; b < BATCH; b++) {
      kn_re[b] = xP[j + b] / 2.0;
      kn_im[b] = 0.0;
      qq2_re[b] = xP[j + b] * xP[j + b] / 4.0; /* store once */
    }

    for (int ii = 0; ii < nlayers + 1; ii++) {

      double kn_next_re[BATCH], kn_next_im[BATCH];

      /* Use qq2_re, not kn^2, to match the scalar version exactly */
      for (int b = 0; b < BATCH; b++) {
        double complex kn_next = csqrt(CMPLX(qq2_re[b], 0.0) - SLD[ii + 1]);
        kn_next_re[b] = creal(kn_next);
        kn_next_im[b] = cimag(kn_next);
      }

      double rj_re[BATCH], rj_im[BATCH];

      for (int b = 0; b < BATCH; b++) {
        double num_re = kn_re[b] - kn_next_re[b];
        double num_im = kn_im[b] - kn_next_im[b];
        double den_re = kn_re[b] + kn_next_re[b];
        double den_im = kn_im[b] + kn_next_im[b];

        double den2 = den_re * den_re + den_im * den_im;
        double t_re = (num_re * den_re + num_im * den_im) / den2;
        double t_im = (num_im * den_re - num_re * den_im) / den2;

        double rs = rough_sqr[ii];
        double p_re =
            (kn_re[b] * kn_next_re[b] - kn_im[b] * kn_next_im[b]) * rs;
        double p_im =
            (kn_re[b] * kn_next_im[b] + kn_im[b] * kn_next_re[b]) * rs;

        double dw = exp(p_re);
        double dw_re = dw * cos(p_im);
        double dw_im = dw * sin(p_im);

        rj_re[b] = t_re * dw_re - t_im * dw_im;
        rj_im[b] = t_re * dw_im + t_im * dw_re;
      }

      if (!ii) {
        for (int b = 0; b < BATCH; b++) {
          M00_re[b] = 1.0;
          M00_im[b] = 0.0;
          M01_re[b] = rj_re[b];
          M01_im[b] = rj_im[b];
          M10_re[b] = rj_re[b];
          M10_im[b] = rj_im[b];
          M11_re[b] = 1.0;
          M11_im[b] = 0.0;
        }
      } else {
        double t = cimag(thickness[ii - 1]);

        double beta_re[BATCH], beta_im[BATCH];
        double inv_beta_re[BATCH], inv_beta_im[BATCH];

        for (int b = 0; b < BATCH; b++) {
          double arg_re = -kn_im[b] * t;
          double arg_im = kn_re[b] * t;
          double env = exp(arg_re);
          beta_re[b] = env * cos(arg_im);
          beta_im[b] = env * sin(arg_im);

          double mag2 = beta_re[b] * beta_re[b] + beta_im[b] * beta_im[b];
          inv_beta_re[b] = beta_re[b] / mag2;
          inv_beta_im[b] = -beta_im[b] / mag2;
        }

        for (int b = 0; b < BATCH; b++) {
          double p0_re = M00_re[b] * beta_re[b] - M00_im[b] * beta_im[b];
          double p0_im = M00_re[b] * beta_im[b] + M00_im[b] * beta_re[b];
          double q0_re =
              M01_re[b] * inv_beta_re[b] - M01_im[b] * inv_beta_im[b];
          double q0_im =
              M01_re[b] * inv_beta_im[b] + M01_im[b] * inv_beta_re[b];

          M00_re[b] = p0_re + rj_re[b] * q0_re - rj_im[b] * q0_im;
          M00_im[b] = p0_im + rj_re[b] * q0_im + rj_im[b] * q0_re;
          M01_re[b] = rj_re[b] * p0_re - rj_im[b] * p0_im + q0_re;
          M01_im[b] = rj_re[b] * p0_im + rj_im[b] * p0_re + q0_im;

          double p1_re = M10_re[b] * beta_re[b] - M10_im[b] * beta_im[b];
          double p1_im = M10_re[b] * beta_im[b] + M10_im[b] * beta_re[b];
          double q1_re =
              M11_re[b] * inv_beta_re[b] - M11_im[b] * inv_beta_im[b];
          double q1_im =
              M11_re[b] * inv_beta_im[b] + M11_im[b] * inv_beta_re[b];

          M10_re[b] = p1_re + rj_re[b] * q1_re - rj_im[b] * q1_im;
          M10_im[b] = p1_im + rj_re[b] * q1_im + rj_im[b] * q1_re;
          M11_re[b] = rj_re[b] * p1_re - rj_im[b] * p1_im + q1_re;
          M11_im[b] = rj_re[b] * p1_im + rj_im[b] * p1_re + q1_im;
        }
      }

      for (int b = 0; b < BATCH; b++) {
        kn_re[b] = kn_next_re[b];
        kn_im[b] = kn_next_im[b];
      }
    }

    for (int b = 0; b < BATCH; b++) {
      double num = M10_re[b] * M10_re[b] + M10_im[b] * M10_im[b];
      double den = M00_re[b] * M00_re[b] + M00_im[b] * M00_im[b];
      yP[j + b] = (num / den) * scale + bkg;
    }
  }

  /* ------------------------------------------------------------------ *
   * Scalar tail: handle remaining points when npoints % BATCH != 0.
   * Reuse the original scalar abeles logic for simplicity.
   * ------------------------------------------------------------------ */
  for (; j < npoints; j++) {
    double complex kn, kn_next, rj, beta;
    _Complex double MRtotal[2][2];
    double complex oneC = CMPLX(1., 0.);

    double complex qq2 = CMPLX(xP[j] * xP[j] / 4, 0);
    kn = xP[j] / 2.;

    for (int ii = 0; ii < nlayers + 1; ii++) {
      kn_next = csqrt(qq2 - SLD[ii + 1]);
      rj = (kn - kn_next) / (kn + kn_next) * cexp(kn * kn_next * rough_sqr[ii]);

      if (!ii) {
        MRtotal[0][0] = oneC;
        MRtotal[0][1] = rj;
        MRtotal[1][0] = rj;
        MRtotal[1][1] = oneC;
      } else {
        beta = cexp(kn * thickness[ii - 1]);
        _Complex double inv_beta = oneC / beta;
        _Complex double p0 = MRtotal[0][0] * beta;
        _Complex double q0 = MRtotal[0][1] * inv_beta;
        _Complex double p1 = MRtotal[1][0] * beta;
        _Complex double q1 = MRtotal[1][1] * inv_beta;
        MRtotal[0][0] = p0 + rj * q0;
        MRtotal[0][1] = rj * p0 + q0;
        MRtotal[1][0] = p1 + rj * q1;
        MRtotal[1][1] = rj * p1 + q1;
      }
      kn = kn_next;
    }

    double num = cabs(MRtotal[1][0]);
    num *= num;
    double den = cabs(MRtotal[0][0]);
    den *= den;
    yP[j] = (num / den) * scale + bkg;
  }

done:
  if (SLD)
    free(SLD);
  if (thickness)
    free(thickness);
  if (rough_sqr)
    free(rough_sqr);
}

void parratt(int numcoefs, const double *restrict coefP, int npoints,
             double *restrict yP, const double *restrict xP) {
  int j;
  double scale, bkg;

  double complex super;
  double complex sub;
  double complex _t;
  _Complex double *SLD = NULL;
  _Complex double *thickness = NULL;
  double complex qq2;
  double *rough_sqr = NULL;

  int nlayers = (int)coefP[0];

  SLD = (_Complex double *)malloc((nlayers + 2) * sizeof(_Complex double));
  if (!SLD)
    goto done;

  thickness = (_Complex double *)malloc((nlayers) * sizeof(_Complex double));
  if (!thickness)
    goto done;

  rough_sqr = (double *)malloc((nlayers + 1) * sizeof(double));
  if (!rough_sqr)
    goto done;

  scale = coefP[1];
  bkg = coefP[6];
  sub = CMPLX(coefP[4], fabs(coefP[5]) + TINY);
  super = CMPLX(coefP[2], 0);

  // fill out all the SLD's for all the layers
  for (int ii = 1; ii < nlayers + 1; ii += 1) {
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

    for (int ii = nlayers; ii > -1; ii--) {
      // wavevector in the layer
      kn = csqrt(qq2 - SLD[ii]);

      // reflectance of the interface
      // factor of 2 is already incorporated in rough_sqr
      rj = (kn - kn_next) / (kn + kn_next) * cexp(kn * kn_next * rough_sqr[ii]);

      if (ii == nlayers) {
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
  if (SLD)
    free(SLD);
  if (thickness)
    free(thickness);
  if (rough_sqr)
    free(rough_sqr);
}

#ifdef __cplusplus
}
#endif
