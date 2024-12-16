/* This program is public domain. */

#ifndef _REFLCALC_H
#define _REFLCALC_H

/*


Reflectivity calculations on rectangular slabs.

   Parameters:
   - M:      Number of slabs
   - d[M]:   depth of each slab; first and last ignored (semi-infinite)
   - rho[M]: scattering length density in number density units
   - mu[M]:  absorption; incident layer ignored
   - lambda: wavelength (required if absorption != 0)
   - P[M]:   magnetic scattering length density
   - exptheta[2*M]: cos(theta),sin(theta) pairs for magnetic scattering angle
   - Aguide  polarization angle relative to the guide (usually -90)
   - N:      Number of Q points
   - Q[N]:   Q points; negative for back reflectivity, which is implemented
             by reversing the order of the layers
   - R[N]:   returned reflectivity = |r|^2
   - r[N]:   returned complex reflectivity amplitude
   - real_r[N]: returned real portion of complex reflectivity amplitude

   Functions:
   reflectivity(M,d,rho,mu,lambda,N,Q,R)
   reflectivity_amplitude(M,d,rho,mu,lambda,N,Q,r)
   reflectivity_real(M,d,rho,mu,lambda,N,Q,real_r)
   reflectivity_imag(M,d,rho,mu,lambda,N,Q,real_r)
   magnetic_reflectivity(M,d,rho,mu,lambda,P,exptheta,Aguide,N,Q,R)
   magnetic_amplitude(M,d,rho,mu,lambda,P,exptheta,Aguide,N,Q,r)


Fresnel reflectivity from a single interface.

   Parameters:
   - Vrho:   incident medium rho
   - Srho:   substrate rho
   - Vmu:    incident medium absorption
   - Smu:    substrate absorption
   - lambda: wavelength (required if absorption != 0)
   - N:      number of points
   - f[N]:   Fresnel complex reflectivity amplitude
   - F[N]:   Fresnel reflectivity = |f|^2

   Functions:
   fresnel_reflectivity(Vrho, Srho, Vmu, Smu, N, Q, F, lambda)
   fresnel_amplitude(Vrho, Srho, Vmu, Smu, N, Q, f, lambda)

Resolution convolution

   Parameters:
   - M          number of points in model
   - Qin[M]     Q values of points in model
   - Rin[M]     R values of points in model
   - N          number of points in data
   - Q[N]       Q values of points in data
   - dQ[N]      dQ uncertainty in Q values
   - R[N]       returned estimate of reflectivity

   Functions:
   resolution(M, Qin, Rin, N, Q, dQ, R)

Resolution estimate to determine the width of the gaussian dQ to associate
with each point Q

   Parameters:
   - s1         slit 1 opening
   - s2         slit 2 opening
   - d          distance between slits
   - A3         angle at which s1,s2 were measured
   - dT         angular divergence for fixed slits
   - dToT       angular divergence for opening slits
   - L          wavelength
   - dLoL       wavelength divergence
   - N          number of points in data
   - Q[N]       Q values of points in data
   - dQ[N]      returned estimate of uncertainty


   Functions:
   dT = resolution_dT(s1,s2,d)
   dToT = resolution_dToT(s1,s2,d,A3)
   resolution_fixed(L,dLoL,dT,N,Q,dQ)
   resolution_varying(L,dLoL,dToT,N,Q,dQ)

Resolution padding to determine the number of additional steps beyond
Q of a given step size in order to reach 0.001 on a gaussian of width dQ

   Parameters:
   - w          number of additional steps needed to compute reflectivity
   - step       step size

   Functions:
   w = resolution_padding(dQ, step)
*/

// For C++ use the standard complex double type
// For C just use an array of doubles
// Note: C99 could use complex double.
#ifdef __cplusplus
#include <cmath>
#include <complex>
typedef std::complex<double> Cplx;
#else
#include <math.h>
typedef double Cplx;
#endif

// Inclusion from C
#ifdef __cplusplus
extern "C" {
#endif

void reflectivity_amplitude(const int layers, const double d[],
                            const double sigma[], const double rho[],
                            const double irho[], const int points,
                            const double kz[], const int rho_offset[],
                            Cplx r[]);

void magnetic_amplitude(const int layers, const double d[],
                        const double sigma[], const double rho[],
                        const double irho[], const double rhoM[],
                        const Cplx u1[], const Cplx u3[], const int points,
                        const double kz[], const int rho_offset[], Cplx Ra[],
                        Cplx Rb[], Cplx Rc[], Cplx Rd[]);

void calculate_U1_U3(const double H, double &rhoM, const double thetaM,
                     const double Aguide, Cplx &U1, Cplx &U3);

int align_magnetic(int nlayers, double d[], double sigma[], double rho[],
                   double irho[], int nlayersM, double dM[], double sigmaM[],
                   double rhoM[], double thetaM[], int noutput,
                   double output[]);

int contract_by_step(int n, double d[], double sigma[], double rho[],
                     double irho[], double dh);

int contract_by_area(int n, double d[], double sigma[], double rho[],
                     double irho[], double dA);

int contract_mag(int n, double d[], double sigma[], double rho[], double irho[],
                 double rhoM[], double thetaM[], double dA);

void convolve_gaussian(size_t Nin, const double xin[], const double yin[],
                       size_t N, const double x[], const double dx[],
                       double y[]);

void convolve_sampled(size_t Nin, const double xin[], const double yin[],
                      size_t Np, const double xp[], const double yp[], size_t N,
                      const double x[], const double dx[], double y[]);

void build_profile(size_t NZ, size_t NP, size_t NI,
                   const double z[],             /* length NZ */
                   const double offset[],        /* length NI */
                   const double roughness[],     /* length NI */
                   const double contrast[],      /* length NP * NI */
                   const double initial_value[], /* length NP */
                   double profile[] /* length NP * NZ (num profiles * num z) */
);

#ifdef __cplusplus
}
#endif

#endif /* _REFLCALC_H */
