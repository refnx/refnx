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
#include <vector>
#include <assert.h>
#include <unistd.h>
#ifdef _POSIX_THREADS
#include <pthread.h>
#endif


#define NUM_CPUS 2


using namespace std;
using namespace MyComplexNumber;

#ifdef __cplusplus
extern "C" {
#endif
	
	MyComplex fres(MyComplex a,MyComplex b,double rough){
		return (compexp(-2*rough*rough*a*b))*(a-b)/(a+b);
	}
	
	void AbelesCalc_ImagAll(double *coefP, int numcoefs, double *yP, double *xP,int npoints, int Vmullayers, int Vmulappend, int Vmulrep){
		int err = 0;
		int j;
		
		int ii=0,jj=0,kk=0;
		
		double scale,bkg,subrough;
		double num=0,den=0, answer=0;
		double anum, anum2;
		
		MyComplex super;
		MyComplex sub;
		MyComplex temp,SLD,beta,rj,arg;
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
			memset(MRtotal,0,sizeof(MRtotal));
			MRtotal[0][0]=oneC;MRtotal[1][1]=oneC;
			
			qq2=MyComplex(xP[j]*xP[j]/4,0);
			
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
				beta = (ii==0)? oneC : compexp(pj[ii] * MyComplex(0,fabs(coefP[4*ii+4])));
				
				//this is the characteristic matrix of a layer
				MI[0][0]=beta;
				MI[0][1]=rj*beta;
				MI[1][1]=oneC/beta;
				MI[1][0]=rj*MI[1][1];
				
				memcpy(temp2, MRtotal, sizeof(MRtotal));
				
				//multiply MR,MI to get the updated total matrix.			
				matmul(temp2,MI,MRtotal);
				
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
								//						temp2[0][0] = MRtotal[0][0];
								//						temp2[0][1] = MRtotal[0][1];
								//						temp2[1][0] = MRtotal[1][0];
								//						temp2[1][1] = MRtotal[1][1];
								
								matmul(temp2,MI,MRtotal);
							}
						} else {
							memcpy(temp2, MRtotal, sizeof(MRtotal));
							//					temp2[0][0] = MRtotal[0][0];
							//					temp2[0][1] = MRtotal[0][1];
							//					temp2[1][0] = MRtotal[1][0];
							//					temp2[1][1] = MRtotal[1][1];
							
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
	
	/* openMP version	
	void AbelesCalc_Imag(double *coefP, int numcoefs, double *yP, double *xP,int npoints, int Vmullayers, int Vmulappend, int Vmulrep){
		int err = 0;
		int j;
		
		int jj=0,kk=0;
		
		double scale,bkg,subrough;		
		MyComplex super;
		MyComplex sub;
		MyComplex oneC = MyComplex(1,0);
		int offset=0;

		MyComplex *pj_mul = NULL;
		MyComplex **pj = NULL;
		MyComplex *SLDmatrix = NULL;
		MyComplex *SLDmatrixREP = NULL;
		
		int nlayers = (int)coefP[0];
						
		try{
			pj = new MyComplex*[npoints];
			for(j = 0 ; j < npoints ; j++)
				pj[j] = new MyComplex[nlayers + 2];
	
			SLDmatrix = new MyComplex[nlayers + 2];
		} catch(...){
			err = 1;
			goto done;
		}
		
		memset(SLDmatrix, 0, sizeof(SLDmatrix));
		
		scale = coefP[1];
		bkg = coefP[6];
		subrough = coefP[7];
		sub= MyComplex(coefP[4]*1e-6, coefP[5]);
		super = MyComplex(coefP[2]*1e-6, coefP[3]);
		
		//offset tells us where the multilayers start.
		offset = 4 * nlayers + 8;
		
		//fillout all the SLD's for all the layers
		for(kk=1; kk<nlayers+1;kk+=1)
			*(SLDmatrix + kk) = 4 * PI * (MyComplex(coefP[4 * kk + 5] * 1e-6, coefP[4 * kk + 6]) - super);
		
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
			for(kk=0; kk<Vmullayers;kk+=1)
				*(SLDmatrixREP + kk) = 4 * PI * (MyComplex(coefP[(4 * kk) + offset + 1] * 1e-6, coefP[(4 * kk) + offset + 2])  - super);
		}

		#pragma omp parallel for shared(pj) private(j)
		for (j = 0; j < npoints; j++) {
			MyComplex temp,SLD,beta,rj,arg;
			int ii = 0;
			double num=0,den=0, answer=0;
			double anum, anum2;

			MyComplex MRtotal[2][2];
			MyComplex subtotal[2][2];
			MyComplex MI[2][2];
			MyComplex temp2[2][2];
			MyComplex qq2;
			
			//intialise the matrices
			memset(MRtotal,0,sizeof(MRtotal));
			MRtotal[0][0].re = 1;
			MRtotal[1][1].re = 1;

			qq2=MyComplex(xP[j]*xP[j]/4,0);
			
			for(ii=0; ii<nlayers+2 ; ii++){			//work out the wavevector in each of the layers
				pj[j][ii] = compsqrt(qq2-*(SLDmatrix+ii));
			}
			
			//workout the wavevector in the toplayer of the multilayer, if it exists.
			if(Vmullayers > 0 && Vmulrep > 0 && Vmulappend >=0){
				memset(subtotal,0,sizeof(subtotal));
				subtotal[0][0]=MyComplex(1,0);subtotal[1][1]=MyComplex(1,0);
				pj_mul[0] = compsqrt(qq2-*SLDmatrixREP);
			}
			
			//now calculate reflectivities
			for(ii = 0 ; ii < nlayers+1 ; ii++){
				if(Vmullayers > 0 && ii == Vmulappend && Vmulrep > 0 )
					rj = fres(pj[j][ii], pj_mul[0], coefP[offset+3]);
				else {
					if((pj[j][ii]).im == 0 && (pj[j][ii + 1]).im == 0){
						anum = (pj[j][ii]).re;
						anum2 = (pj[j][ii + 1]).re;
						rj.re = (ii == nlayers) ? 
						((anum - anum2) / (anum + anum2)) * exp(anum * anum2 * -2 * subrough * subrough)
						:
						((anum - anum2) / (anum + anum2)) * exp(anum * anum2 * -2 * coefP[4 * (ii + 1) + 7] * coefP[4 * (ii + 1) + 7]);
						rj.im = 0.;
					} else {
						rj = (ii == nlayers) ?
						((pj[j][ii] - pj[j][ii + 1])/(pj[j][ii] + pj[j][ii + 1])) * compexp(pj[j][ii] * pj[j][ii + 1] * -2 * subrough * subrough)
						:
						((pj[j][ii] - pj[j][ii + 1])/(pj[j][ii] + pj[j][ii + 1])) * compexp(pj[j][ii] * pj[j][ii + 1] * -2 * coefP[4 * (ii + 1) + 7] * coefP[4 * (ii + 1) + 7]);	
					};
				}
				
				
				//work out the beta for the (non-multi)layer
				beta = (ii==0)? oneC : compexp(pj[j][ii] * MyComplex(0,fabs(coefP[4*ii+4])));
				
				//this is the characteristic matrix of a layer
				MI[0][0]=beta;
				MI[0][1]=rj*beta;
				MI[1][1]=oneC/beta;
				MI[1][0]=rj*MI[1][1];
				
				memcpy(temp2, MRtotal, sizeof(MRtotal));
				
				//multiply MR,MI to get the updated total matrix.			
				matmul(temp2,MI,MRtotal);
				
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
										rj = ((pj_mul[Vmullayers-1]-pj[j][nlayers+1])/(pj_mul[Vmullayers-1]+pj[j][nlayers+1]))*compexp((pj_mul[Vmullayers-1]*pj[j][nlayers+1])*(-2*subrough*subrough));
									} else {
										rj = ((pj_mul[Vmullayers-1]-pj[j][Vmulappend+1])/(pj_mul[Vmullayers-1]+pj[j][Vmulappend+1]))* compexp((pj_mul[Vmullayers-1]*pj[j][Vmulappend+1])*(-2*coefP[4*(Vmulappend+1)+7]*coefP[4*(Vmulappend+1)+7]));
									};
								} else {
									rj = ((pj_mul[jj]-pj_mul[jj+1])/(pj_mul[jj]+pj_mul[jj+1]))*compexp((pj_mul[jj]*pj_mul[jj+1])*-2*coefP[4*(jj+1)+offset+3]*coefP[4*(jj+1)+offset+3]);
								}
								
								MI[0][0]=beta;
								MI[0][1]=rj*beta;
								MI[1][1]=MyComplex(1,0)/MI[0][0];
								MI[1][0]=rj*MI[1][1];
								
								memcpy(temp2, MRtotal, sizeof(MRtotal));
								//						temp2[0][0] = MRtotal[0][0];
								//						temp2[0][1] = MRtotal[0][1];
								//						temp2[1][0] = MRtotal[1][0];
								//						temp2[1][1] = MRtotal[1][1];
								
								matmul(temp2,MI,MRtotal);
							}
						} else {
							memcpy(temp2, MRtotal, sizeof(MRtotal));
							//					temp2[0][0] = MRtotal[0][0];
							//					temp2[0][1] = MRtotal[0][1];
							//					temp2[1][0] = MRtotal[1][0];
							//					temp2[1][1] = MRtotal[1][1];
							
							matmul(temp2,subtotal,MRtotal);
						};
					};
				};
				
			}
			
			den= compnorm(MRtotal[0][0]);
			num=compnorm(MRtotal[1][0]);
			answer=(num/den);//(num*num)/(den*den);
			answer=(answer*scale)+fabs(bkg);
			
			yP[j] = answer;
		}
#pragma omp join
	done:
		if(pj != NULL){
			for(j = 0 ; j < npoints ; j++)
				delete [] pj[j];
			delete [] pj;
		}

		if(pj_mul !=NULL)
			delete[] pj_mul;
		if(SLDmatrix != NULL)
			delete[] SLDmatrix;
		if(SLDmatrixREP != NULL)
			delete[] SLDmatrixREP;
		
	}
	*/
	
	/* pthread version*/
	
#ifdef	_POSIX_THREADS
	
	typedef struct{
		//number of Q points we have to calculate
		long npoints;
		//how many layers in the multilayer (optional, but if not specified should be 0)
		int Vmullayers;
		//where the multilayer is appended to the basic model.
		int Vappendlayer;
		//how many repeats in the multilayer
		int Vmulrep;
		//a double array containing the model coefficients (assumed to be correct)
		const double *coefP;
		//the Reflectivity values to return
		double *yP;
		//the Q values to do the calculation for.
		const double *xP;
		MyComplex *pj;
		MyComplex *SLDmatrix;
		MyComplex *SLDmatrixREP;
	}  pointCalcParm;

	
	void *ThreadWorker(void *arg){
		int err = NULL;
		pointCalcParm *p = (pointCalcParm *) arg;

		MyComplex temp,SLD,beta,rj;
		int ii = 0, jj = 0, kk=0;
		double num=0,den=0, answer=0;
		double anum, anum2;
		
		MyComplex MRtotal[2][2];
		MyComplex subtotal[2][2];
		MyComplex MI[2][2];
		MyComplex temp2[2][2];
		MyComplex qq2;
		MyComplex *SLDmatrix = p->SLDmatrix;
		MyComplex *SLDmatrixREP = p->SLDmatrixREP;
		MyComplex *pj = p->pj;
		MyComplex *pj_mul = NULL;
		MyComplex oneC = MyComplex(1,0);
		const double *xP = p->xP;
		double *yP = p->yP;
		int Vmullayers = p->Vmullayers;
		int Vmulrep = p->Vmulrep;
		int Vmulappend = p->Vappendlayer;
		int j = 0;
		int npoints = p->npoints;
		const double *coefP = p->coefP;
		int nlayers = (int)coefP[0];
		int offset = 4 * nlayers + 8;
		double subrough = coefP[7];
		double scale = coefP[1];
		double bkg = coefP[6];

		for (j = 0; j < p->npoints; j++) {
			
			//intialise the matrices
			memset(MRtotal,0,sizeof(MRtotal));
			MRtotal[0][0].re = 1;
			MRtotal[1][1].re = 1;
			
			qq2=MyComplex(xP[j]*xP[j]/4,0);


			for(ii=0; ii<nlayers+2 ; ii++){			//work out the wavevector in each of the layers
				pj[ii] = compsqrt(qq2 - SLDmatrix[ii]);
			}
			
			//workout the wavevector in the toplayer of the multilayer, if it exists.
			if(Vmullayers > 0 && Vmulrep > 0 && Vmulappend >=0){
				memset(subtotal,0,sizeof(subtotal));
				subtotal[0][0]=MyComplex(1,0);subtotal[1][1]=MyComplex(1,0);
				pj_mul[0] = compsqrt(qq2-*SLDmatrixREP);
			}
			//now calculate reflectivities
			for(ii = 0 ; ii < nlayers+1 ; ii++){
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
				beta = (ii==0)? oneC : compexp(pj[ii] * MyComplex(0,fabs(coefP[4*ii+4])));
				
				//this is the characteristic matrix of a layer
				MI[0][0]=beta;
				MI[0][1]=rj*beta;
				MI[1][1]=oneC/beta;
				MI[1][0]=rj*MI[1][1];
				
				memcpy(temp2, MRtotal, sizeof(MRtotal));
				
				//multiply MR,MI to get the updated total matrix.			
				matmul(temp2,MI,MRtotal);
				
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
			
			yP[j] = answer;
		}
		pthread_exit((void*)err);
		return NULL;
	}
	
		
	void AbelesCalc_Imag(double *coefP, int numcoefs, double *yP, double *xP,int npoints, int Vmullayers, int Vmulappend, int Vmulrep){
		int err = 0;
		int j;
		
		int jj=0,kk=0;

		pthread_t *threads = NULL;
		pointCalcParm *arg = NULL;

		double scale,bkg,subrough;		
		MyComplex super;
		MyComplex sub;
		MyComplex oneC = MyComplex(1,0);
		int offset=0;
		int ii;
		int threadsToCreate = NUM_CPUS;
		int pointsEachThread, pointsRemaining, pointsConsumed;

		MyComplex *pj_mul = NULL;
		MyComplex **pj = NULL;
		MyComplex *SLDmatrix = NULL;
		MyComplex *SLDmatrixREP = NULL;
		
		int nlayers = (int)coefP[0];
		
		try{
			pj = new MyComplex*[threadsToCreate];
			for(j = 0 ; j < threadsToCreate ; j++)
				pj[j] = new MyComplex[nlayers + 2];
			
			SLDmatrix = new MyComplex[nlayers + 2];
		} catch(...){
			err = 1;
			goto done;
		}
		
		memset(SLDmatrix, 0, sizeof(SLDmatrix));
		
		scale = coefP[1];
		bkg = coefP[6];
		subrough = coefP[7];
		sub= MyComplex(coefP[4]*1e-6, coefP[5]);
		super = MyComplex(coefP[2]*1e-6, coefP[3]);
		
		//offset tells us where the multilayers start.
		offset = 4 * nlayers + 8;
		
		//fillout all the SLD's for all the layers
		for(kk=1; kk<nlayers+1;kk+=1)
			*(SLDmatrix + kk) = 4 * PI * (MyComplex(coefP[4 * kk + 5] * 1e-6, coefP[4 * kk + 6]) - super);
		
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
			for(kk=0; kk<Vmullayers;kk+=1)
				*(SLDmatrixREP + kk) = 4 * PI * (MyComplex(coefP[(4 * kk) + offset + 1] * 1e-6, coefP[(4 * kk) + offset + 2])  - super);
		}
		
		//create threads for the calculation
		threads = (pthread_t *) malloc((threadsToCreate) * sizeof(pthread_t));
		if(!threads && NUM_CPUS > 1){
			err = 1;
			goto done;
		}
		//create arguments to be supplied to each of the threads
		arg = (pointCalcParm *) malloc (sizeof(pointCalcParm)*(threadsToCreate));
		if(!arg && NUM_CPUS > 1){
			err = 1;
			goto done;
		}
		
		//need to calculated how many points are given to each thread.
		pointsEachThread = floorl(npoints / threadsToCreate);
		pointsRemaining = npoints;
		pointsConsumed = 0;
		
		//if you have two CPU's, only create one extra thread because the main thread does half the work
		for (ii = 0; ii < threadsToCreate; ii++){
			arg[ii].coefP = coefP;
			if(ii == threadsToCreate - 1)
				pointsEachThread = pointsRemaining;
				
			arg[ii].npoints = pointsEachThread;
			arg[ii].Vmullayers = Vmullayers;
			arg[ii].Vappendlayer = Vmulappend;
			arg[ii].Vmulrep = Vmulrep;
			arg[ii].pj = pj[ii];
			arg[ii].SLDmatrix = SLDmatrix;
			arg[ii].SLDmatrixREP = SLDmatrixREP;
			//the following two lines specify where the Q values and R values will be sourced/written.
			//i.e. an offset of the original array.
			arg[ii].xP = xP+pointsConsumed;
			arg[ii].yP = yP+pointsConsumed;
			
			pthread_create(&threads[ii], NULL, ThreadWorker, (void *)(arg+ii));
			pointsRemaining -= pointsEachThread;
			pointsConsumed += pointsEachThread;
		}
		
		for (ii = 0; ii < threadsToCreate ; ii++)
			pthread_join(threads[ii], NULL);
		
	done:
		if(pj != NULL){
			for(j = 0 ; j < threadsToCreate ; j++)
				delete [] pj[j];
			delete [] pj;
		}
		if(threads)
			free(threads);
		if(arg)
			free(arg);
		
		if(pj_mul !=NULL)
			delete[] pj_mul;
		if(SLDmatrix != NULL)
			delete[] SLDmatrix;
		if(SLDmatrixREP != NULL)
			delete[] SLDmatrixREP;
		
	}
#endif

#ifdef __cplusplus
	}
#endif
	
