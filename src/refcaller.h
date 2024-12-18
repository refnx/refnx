/*
    refcalc.h

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
    reflect uses the Abeles matrix method to calculate specular reflectivity.

    parratt uses the Parratt recursion formula to calculate specular
    reflectivity.

    The system is assumed to consist of a series of uniformly dense layers
    (slabs). The radiation is incident through the fronting medium, leaving
    through the backing medium.

    Parameters
    ----------
    batch - MT versions only. The batch size of the calculation. coefP should
    have size `batch * (4*N + 8)`.

    numcoefs - the number of parameters held in the coefP array. This should be
    `4*N + 8` values long, were N is the number of layers.

    coefP - an array holding the parameters.  See below for a more detailed
    description

    npoints - the number of points in the yP and xP arrays

    yP - this user supplied array is filled by the reflect function.  It must
    be npoints long

    xP - array containing the Q (momentum transfer) points. It has units Å**-1.
    The array is npoints long

    threads - specifies the number of parallel threads to use.


    Detailed description of the entries in coefP
    --------------------------------------------
    coefP[0] - the number of layers. Should be an integer.
               4 * coefP[0] + 8 == numcoefs

    coefP[1] - scale factor.  The calculated reflectivity is multiplied by this
    value. i.e. yP = scale * ref + background

    coefP[2] - fronting SLD, real part. (10**-6 Å**-2)

    coefP[3] - fronting SLD, imaginary part. (10**-6 Å**-2)

    coefP[4] - backing SLD, real part. (10**-6 Å**-2)

    coefP[5] - backing SLD, imaginary part. (10**-6 Å**-2)

    coefP[6] - background.  A constant value added to all the reflectivity
    points.

    coefP[7] - roughness between backing medium and bottom most layer (Å).  If
    there are no layers, the roughness between the fronting and backing media.

    coefP[4 * M + 8] - thickness of layer M, with M = 0, ... N - 1. (Å)

    coefP[4 * M + 9] - SLD of layer M, real part (10**-6 Å**-2)

    coefP[4 * M + 10] - SLD of layer M, imaginary part (10**-6 Å**-2)

    coefP[4 * M + 11] - roughness between layer M - 1 / M (Å)
*/

/*
Parallelised
*/
void abeles_wrapper_MT(unsigned int batch, int numcoefs, const double *coefP,
                       int npoints, double *yP, const double *xP, int threads);

void parratt_wrapper_MT(unsigned int batch, int numcoefs, const double *coefP,
                        int npoints, double *yP, const double *xP, int threads);

/*
Non parallelised
*/
void abeles_wrapper(int numcoefs, const double *coefP, int npoints, double *yP,
                    const double *xP);

void parratt_wrapper(int numcoefs, const double *coefP, int npoints, double *yP,
                     const double *xP);

/*
Polarised neutron reflection measurement
*/
void pnr(int layers, const double *d, const double *sigma, const double *rho,
         const double *irho, const double *rhoM, const double *thetaM, double H,
         double Aguide, int points, const double *xP, double *Ra, double *Rb,
         double *Rc, double *Rd);

void pnr_MT(int workers, int layers, const double *d, const double *sigma,
            const double *rho, const double *irho, const double *rhoM,
            const double *thetaM, double H, double Aguide, int points,
            const double *xP, double *Ra, double *Rb, double *Rc, double *Rd);
