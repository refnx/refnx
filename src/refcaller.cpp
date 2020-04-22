/*
    refcaller.cpp

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


extern "C" {
    #include "refcalc.h"
}

#include <math.h>
#include <cmath>
#include <stdlib.h>
#include <thread>
#include <vector>


#define NUM_CPUS 4
using namespace std;


void AbelesCalc_Imag(int numcoefs,
                  const double *coefP,
                   int npoints,
                    double *yP,
                     const double *xP,
                      int workers){

    std::vector<std::thread> threads;

    int pointsEachThread, pointsRemaining, pointsConsumed;

    // need to calculated how many points are given to each thread.
    if(workers > 0){
        pointsEachThread = floorl(npoints / workers);
    } else {
        pointsEachThread = npoints;
    }

    pointsRemaining = npoints;
    pointsConsumed = 0;

    for (int ii = 0; ii < workers; ii++){
        if(ii < workers - 1){
            threads.emplace_back(std::thread(AbelesCalc_ImagAll,
                                             numcoefs,
                                             coefP,
                                             pointsEachThread,
                                             yP + pointsConsumed,
                                             xP + pointsConsumed));
            pointsRemaining -= pointsEachThread;
            pointsConsumed += pointsEachThread;
        } else {
            threads.emplace_back(std::thread(AbelesCalc_ImagAll,
                                             numcoefs,
                                             coefP,
                                             pointsRemaining,
                                             yP + pointsConsumed,
                                             xP + pointsConsumed));
            pointsRemaining -= pointsRemaining;
            pointsConsumed += pointsRemaining;
        }
    }

    // synchronise threads
    for (auto& th : threads) th.join();
}


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
    AbelesCalc_Imag(numcoefs, coefP, npoints, yP, xP, threads);
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
