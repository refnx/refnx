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
void abelescalcall(double *coefP, int numcoefs, double *yP, double *xP,  int len3);
	
#ifdef __cplusplus
}
#endif
