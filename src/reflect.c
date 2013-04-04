#include "refcalc.h"
#include "Python.h"
#ifdef _POSIX_THREADS
#include <unistd.h>
#include <pthread.h>
#endif
void abeles(double *yP, int oo, double *xP, int pp, double *coefP, int n) {
	
	int err = 0;
	
	if(oo != pp){
		PyErr_Format(PyExc_Exception,
				 "arg1 needs to be same size as number of values in arg3");
		err = 2;
	}
	if(n>0){
		if (4 * (int)coefP[0] + 8 != n) {
			PyErr_Format(PyExc_Exception,
				"arg2 needs to be length 4 * arg2[0] + 8");
			err = 2;
		}
	}
#ifdef _POSIX_THREADS
		if(!err)
		AbelesCalc_Imag(coefP, n, yP, xP, pp, 0, 0, 0);
#else
		if(!err)
		AbelesCalc_ImagAll(coefP, n, yP, xP, pp, 0, 0, 0);
#endif
}