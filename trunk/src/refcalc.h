/*
 *  myfitfunctions.h
 *  motoMC
 *
 *  Created by andrew on 29/05/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#define PI 3.14159265358979323846

#ifdef __cplusplus
extern "C" {
#endif
	
//fitfunctions
void AbelesCalc_ImagAll(double *coefP, int numcoefs, double *yP, double *xP,int npoints, int Vmullayers, int Vmulappend, int Vmulrep);
		
#ifdef __cplusplus
}
#endif
