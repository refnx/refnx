/*
 *  myfitfunctions.cpp
 *  motoMC
 *
 *  Created by andrew on 29/05/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "refcalc.h"
#include <math.h>
#include "MyComplex.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_CPUS 1


using namespace std;
using namespace MyComplexNumber;

#ifdef __cplusplus
extern "C" {
#endif
	
	MyComplex fres(MyComplex a,MyComplex b,double rough){
		return (compexp(-2*rough*rough*a*b))*(a-b)/(a+b);
	}
	
	void abelescalc_imagall(double *coefP, int numcoefs, double *yP, double *xP, int npoints){
			int err = 0;
			int j;
			int Vmullayers = 0;
			int Vmulappend = 0;
			int Vmulrep = 0;
			
			int ii=0,jj=0,kk=0;
			
			double scale,bkg,subrough, qq;
			double num=0,den=0, answer=0;
			double anum, anum2;
			
			MyComplex super;
			MyComplex sub;
			MyComplex temp,SLD,beta,rj;
			MyComplex oneC = MyComplex(1,0);
			int offset=0;
			MyComplex MRtotal[2][2];
			MyComplex subtotal[2][2];
			MyComplex MI[2][2];
			MyComplex temp2[2][2];
			MyComplex qq2;
			MyComplex *pj_mul = NULL;
			MyComplex *pj = NULL;
			MyComplex *SLDmatrix = NULL;
			MyComplex *SLDmatrixREP = NULL;
			
			int nlayers = (int)coefP[0];
			
			try{
				pj = new MyComplex[nlayers+2];
				SLDmatrix = new MyComplex[nlayers + 2];
			} catch(...){
				err = 1;
				goto done;
			}
			
			memset(pj, 0, sizeof(pj));
			memset(SLDmatrix, 0, sizeof(SLDmatrix));
			
			scale = coefP[1];
			bkg = coefP[6];
			subrough = coefP[7];
			sub= MyComplex(coefP[4]*1e-6, coefP[5]);
			super = MyComplex(coefP[2]*1e-6, coefP[3]);
			
			//offset tells us where the multilayers start.
			offset = 4 * nlayers + 8;
			
			//fillout all the SLD's for all the layers
			for(ii=1; ii<nlayers+1;ii+=1)
				*(SLDmatrix + ii) = 4 * PI * (MyComplex(coefP[4 * ii + 5] * 1e-6, coefP[4 * ii + 6]) - super);
			
			*(SLDmatrix) = MyComplex(0,0);
			*(SLDmatrix + nlayers + 1) = 4 * PI * (sub - super);
			
			if(Vmullayers > 0 && Vmulrep > 0 && Vmulappend >= 0){
				//set up an array for wavevectors
				try{
					SLDmatrixREP = new MyComplex[Vmullayers];
					pj_mul = new MyComplex[Vmullayers];
				} catch(...){
					err = 1;
					goto done;
				}
				memset(pj_mul, 0, sizeof(pj_mul));
				memset(SLDmatrixREP,0,sizeof(SLDmatrixREP));
				for(ii=0; ii<Vmullayers;ii+=1)
					*(SLDmatrixREP + ii) = 4 * PI * (MyComplex(coefP[(4 * ii) + offset + 1] * 1e-6, coefP[(4 * ii) + offset + 2])  - super);
			}
			
			
			for (j = 0; j < npoints; j++) {
				//intialise the matrices
				memset(MRtotal, 0, sizeof(MRtotal));
				MRtotal[0][0].re = 1.;
				MRtotal[0][0].im = 0.;
				MRtotal[1][1].re = 1.;
				MRtotal[1][1].im = 0.;
				
				qq = xP[j] * xP[j] / 4;
				qq2.re = qq;
				qq2.im = 0;
								
				for(ii=0; ii<nlayers+2 ; ii++){			//work out the wavevector in each of the layers
					pj[ii] = compsqrt(qq2-*(SLDmatrix+ii));
				}

				
				//workout the wavevector in the toplayer of the multilayer, if it exists.
				if(Vmullayers > 0 && Vmulrep > 0 && Vmulappend >=0){
					memset(subtotal,0,sizeof(subtotal));
					subtotal[0][0]=MyComplex(1,0);subtotal[1][1]=MyComplex(1,0);
					pj_mul[0] = compsqrt(qq2-*SLDmatrixREP);
				}
				
				//now calculate reflectivities
				for(ii = 0 ; ii < nlayers+1 ; ii++){
					//work out the fresnel coefficient
					if(Vmullayers > 0 && ii == Vmulappend && Vmulrep > 0 )
						rj = fres(pj[ii], pj_mul[0], coefP[offset+3]);
					else {
						if((pj[ii]).im == 0 && (pj[ii + 1]).im == 0){
							anum = (pj[ii]).re;
							anum2 = (pj[ii + 1]).re;
							rj.re = (ii == nlayers) ? 
							((anum - anum2) / (anum + anum2)) * exp(anum * anum2 * -2 * subrough * subrough)
							:
							((anum - anum2) / (anum + anum2)) * exp(anum * anum2 * -2 * coefP[4 * (ii + 1) + 7] * coefP[4 * (ii + 1) + 7]);
							rj.im = 0.;
						} else {
							rj = (ii == nlayers) ?
							((pj[ii] - pj[ii + 1])/(pj[ii] + pj[ii + 1])) * compexp(pj[ii] * pj[ii + 1] * -2 * subrough * subrough)
							:
							((pj[ii] - pj[ii + 1])/(pj[ii] + pj[ii + 1])) * compexp(pj[ii] * pj[ii + 1] * -2 * coefP[4 * (ii + 1) + 7] * coefP[4 * (ii + 1) + 7]);	
						};
					}
					

					//work out the beta for the (non-multi)layer
					temp.re = 0;
					temp.im = fabs(coefP[4 * ii + 2]);
					beta = (ii == 0)? oneC : compexp(pj[ii] * temp);
					
					//this is the characteristic matrix of a layer
					MI[0][0] = beta;
					MI[0][1] = rj * beta;
					MI[1][1] = oneC / beta;
					MI[1][0] = rj * MI[1][1];
					
					memcpy(temp2, MRtotal, sizeof(MRtotal));
					
					//multiply MR,MI to get the updated total matrix.			
					matmul(temp2, MI, MRtotal);
					
					
					if(Vmullayers > 0 && ii == Vmulappend && Vmulrep > 0){
						//workout the wavevectors in each of the layers
						for(jj=1 ; jj < Vmullayers; jj++){
							pj_mul[jj] = compsqrt(qq2-*(SLDmatrixREP+jj));
						}
						
						//work out the fresnel coefficients
						for(jj = 0 ; jj < Vmullayers; jj++){
							rj = (jj == Vmullayers-1) ?
							//if you're in the last layer then the roughness is the roughness of the top
							((pj_mul[jj]-pj_mul[0])/(pj_mul[jj]+pj_mul[0]))* compexp((pj_mul[jj]*pj_mul[0])*-2*coefP[offset+3]*coefP[offset+3])
							:
							//otherwise it's the roughness of the layer below
							((pj_mul[jj]-pj_mul[jj+1])/(pj_mul[jj]+pj_mul[jj+1]))*compexp((pj_mul[jj]*pj_mul[jj+1])*-2*coefP[4*(jj+1)+offset+3]*coefP[4*(jj+1)+offset+3]);
							
							
							//Beta's
							beta = compexp(MyComplex(0,fabs(coefP[4*jj+offset]))*pj_mul[jj]);
							
							MI[0][0]=beta;
							MI[0][1]=rj*beta;
							MI[1][1]=oneC/beta;
							MI[1][0]=rj*MI[1][1];
							
							memcpy(temp2, subtotal, sizeof(subtotal));

							
							matmul(temp2,MI,subtotal);
						};
						
						for(kk = 0; kk < Vmulrep; kk++){		//if you are in the last multilayer
							if(kk==Vmulrep-1){					//if you are in the last layer of the multilayer
								for(jj=0;jj<Vmullayers;jj++){
									beta = compexp((MyComplex(0,fabs(coefP[4*jj+offset]))*pj_mul[jj]));
									
									if(jj==Vmullayers-1){
										if(Vmulappend==nlayers){
											rj = ((pj_mul[Vmullayers-1]-pj[nlayers+1])/(pj_mul[Vmullayers-1]+pj[nlayers+1]))*compexp((pj_mul[Vmullayers-1]*pj[nlayers+1])*(-2*subrough*subrough));
										} else {
											rj = ((pj_mul[Vmullayers-1]-pj[Vmulappend+1])/(pj_mul[Vmullayers-1]+pj[Vmulappend+1]))* compexp((pj_mul[Vmullayers-1]*pj[Vmulappend+1])*(-2*coefP[4*(Vmulappend+1)+7]*coefP[4*(Vmulappend+1)+7]));
										};
									} else {
										rj = ((pj_mul[jj]-pj_mul[jj+1])/(pj_mul[jj]+pj_mul[jj+1]))*compexp((pj_mul[jj]*pj_mul[jj+1])*-2*coefP[4*(jj+1)+offset+3]*coefP[4*(jj+1)+offset+3]);
									}
									
									MI[0][0]=beta;
									MI[0][1]=rj*beta;
									MI[1][1]=MyComplex(1,0)/MI[0][0];
									MI[1][0]=rj*MI[1][1];
									
									memcpy(temp2, MRtotal, sizeof(MRtotal));
									
									matmul(temp2,MI,MRtotal);
								}
							} else {
								memcpy(temp2, MRtotal, sizeof(MRtotal));

								
								matmul(temp2,subtotal,MRtotal);
							};
						};
					};
					
				}
				
				den= compnorm(MRtotal[0][0]);
				num=compnorm(MRtotal[1][0]);
				answer=(num/den);//(num*num)/(den*den);
				answer=(answer*scale)+fabs(bkg);
				
				*yP++ = answer;
			}
			
		done:
			if(pj != NULL)
				delete [] pj;
			if(pj_mul !=NULL)
				delete[] pj_mul;
			if(SLDmatrix != NULL)
				delete[] SLDmatrix;
			if(SLDmatrixREP != NULL)
				delete[] SLDmatrixREP;
			
		}
		
		
		
#ifdef __cplusplus
	}
#endif
	
