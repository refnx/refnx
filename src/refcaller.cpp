/*
    refcaller.cpp

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

extern "C" {
#include "refcalc.h"
}

#include <math.h>
#include <stdlib.h>

#include "pnr/reflcalc.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#define NUM_CPUS 4
using namespace std;

// function pointer for a reflectometry calculator
typedef void (*ref_calculator)(int, const double *, int, double *,
                               const double *xP);

/*
batch worker
*/
void batch_worker(ref_calculator fn, unsigned int num_batches, int numcoefs,
                  const double *coefP, int npoints, double *yP,
                  const double *xP) {
  unsigned int ii;
  for (ii = 0; ii < num_batches; ii++) {
    fn(numcoefs, coefP + (ii * numcoefs), npoints, yP + (ii * npoints), xP);
  }
}

void MT_wrapper(ref_calculator fn, unsigned int batch, int numcoefs,
                const double *coefP, int npoints, double *yP, const double *xP,
                int workers) {

  std::vector<std::thread> threads;

  if (batch < 2) {
    int pointsEachThread, pointsRemaining, pointsConsumed;

    // need to calculate how many points are given to each thread.
    if (workers > 0) {
      pointsEachThread = floorl(npoints / workers);
    } else {
      pointsEachThread = npoints;
    }

    pointsRemaining = npoints;
    pointsConsumed = 0;

    for (int ii = 0; ii < workers; ii++) {
      if (ii < workers - 1) {
        threads.emplace_back(std::thread(fn, numcoefs, coefP, pointsEachThread,
                                         yP + pointsConsumed,
                                         xP + pointsConsumed));
        pointsRemaining -= pointsEachThread;
        pointsConsumed += pointsEachThread;
      } else {
        threads.emplace_back(std::thread(fn, numcoefs, coefP, pointsRemaining,
                                         yP + pointsConsumed,
                                         xP + pointsConsumed));
        pointsRemaining -= pointsRemaining;
        pointsConsumed += pointsRemaining;
      }
    }
  } else {
    unsigned int batchesEachThread, batchesRemaining, batchesConsumed;
    if (workers > 0) {
      batchesEachThread = floorl(batch / workers);
    } else {
      batchesEachThread = batch;
    }

    batchesRemaining = batch;
    batchesConsumed = 0;

    for (int ii = 0; ii < workers; ii++) {
      if (ii < workers - 1) {
        threads.emplace_back(
            std::thread(batch_worker, fn, batchesEachThread, numcoefs,
                        coefP + (batchesConsumed * numcoefs), npoints,
                        yP + (batchesConsumed * npoints), xP));
        batchesRemaining -= batchesEachThread;
        batchesConsumed += batchesEachThread;
      } else {
        threads.emplace_back(
            std::thread(batch_worker, fn, batchesRemaining, numcoefs,
                        coefP + (batchesConsumed * numcoefs), npoints,
                        yP + (batchesConsumed * npoints), xP));
        batchesRemaining -= batchesRemaining;
        batchesConsumed += batchesRemaining;
      }
    }
  }

  // synchronise threads
  for (auto &th : threads)
    th.join();
}

/*
Parallelised version
*/
void abeles_wrapper_MT(unsigned int batch, int numcoefs, const double *coefP,
                       int npoints, double *yP, const double *xP, int threads) {
  MT_wrapper(abeles, batch, numcoefs, coefP, npoints, yP, xP, threads);
}

void parratt_wrapper_MT(unsigned int batch, int numcoefs, const double *coefP,
                        int npoints, double *yP, const double *xP,
                        int threads) {
  MT_wrapper(parratt, batch, numcoefs, coefP, npoints, yP, xP, threads);
}

/*
Non parallelised version
*/
void abeles_wrapper(int numcoefs, const double *coefP, int npoints, double *yP,
                    const double *xP) {
  abeles(numcoefs, coefP, npoints, yP, xP);
}

void parratt_wrapper(int numcoefs, const double *coefP, int npoints, double *yP,
                     const double *xP) {
  parratt(numcoefs, coefP, npoints, yP, xP);
}

void pnr(int layers, const double *d, const double *sigma, const double *rho,
         const double *irho, const double *rhoM, const double *thetaM, double H,
         double Aguide, int points, const double *xP, double *Ra, double *Rb,
         double *Rc, double *Rd) {
  std::vector<double> kz(points);
  std::vector<Cplx> u1(layers), u3(layers);
  std::vector<Cplx> Rac(points), Rbc(points), Rcc(points), Rdc(points);
  std::vector<double> rhoM_mod(layers);
  int idx;

  memcpy(rhoM_mod.data(), rhoM, sizeof(double) * layers);

  for (idx = 0; idx < layers; idx++) {
    calculate_U1_U3(H, *(rhoM_mod.data() + idx), thetaM[idx] * M_PI / 180.,
                    Aguide, *(u1.data() + idx), *(u3.data() + idx));
  }
  for (idx = 0; idx < points; idx++) {
    kz[idx] = xP[idx] / 2.0;
  }

  magnetic_amplitude(layers, d, sigma, rho, irho, rhoM_mod.data(), u1.data(),
                     u3.data(), points, kz.data(), NULL, Rac.data(), Rbc.data(),
                     Rcc.data(), Rdc.data());
  for (idx = 0; idx < points; idx++) {
    Ra[idx] = std::norm(Rac[idx]);
    Rb[idx] = std::norm(Rbc[idx]);
    Rc[idx] = std::norm(Rcc[idx]);
    Rd[idx] = std::norm(Rdc[idx]);
  }
}

void pnr_MT(int workers, int layers, const double *d, const double *sigma,
            const double *rho, const double *irho, const double *rhoM,
            const double *thetaM, double H, double Aguide, int points,
            const double *xP, double *Ra, double *Rb, double *Rc, double *Rd) {
  std::vector<std::thread> threads;
  int pointsEachThread, pointsRemaining, pointsConsumed;

  // need to calculate how many points are given to each thread.
  if (workers > 0) {
    pointsEachThread = floorl(points / workers);
  } else {
    pointsEachThread = points;
  }

  pointsRemaining = points;
  pointsConsumed = 0;

  for (int ii = 0; ii < workers; ii++) {
    if (ii < workers - 1) {
      threads.emplace_back(std::thread(
          pnr, layers, d, sigma, rho, irho, rhoM, thetaM, H, Aguide,
          pointsEachThread, xP + pointsConsumed, Ra + pointsConsumed,
          Rb + pointsConsumed, Rc + pointsConsumed, Rd + pointsConsumed));
      pointsRemaining -= pointsEachThread;
      pointsConsumed += pointsEachThread;
    } else {
      threads.emplace_back(std::thread(
          pnr, layers, d, sigma, rho, irho, rhoM, thetaM, H, Aguide,
          pointsRemaining, xP + pointsConsumed, Ra + pointsConsumed,
          Rb + pointsConsumed, Rc + pointsConsumed, Rd + pointsConsumed));
      pointsRemaining -= pointsRemaining;
      pointsConsumed += pointsRemaining;
    }
  }

  // synchronise threads
  for (auto &th : threads)
    th.join();
}
