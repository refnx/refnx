%module _creflect

%{
#define SWIG_FILE_WITH_INIT
#include "reflect.h"
	%}

%include "numpy.i"

%init %{
    import_array();
	%}

%apply (double* ARGOUT_ARRAY1, int DIM1) {(double *yP, int oo)}
%apply (double* IN_ARRAY1, int DIM1) {(double *xP, int pp)}
%apply (double* IN_ARRAY1, int DIM1) {(double *coefP, int n)}

%include "reflect.h"