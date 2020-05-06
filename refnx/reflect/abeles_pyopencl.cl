/*
Calculates the specular (Neutron or X-ray) reflectivity from a stratified
series of layers.

The refnx code is distributed under the following license:

Copyright (c) 2020 A. R. J. Nelson, ANSTO

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
#define PYOPENCL_DEFINE_CDOUBLE
#include <pyopencl-complex.h>

__kernel void abeles(
    __global const double *q_g, __global const double *coefs_g, __global double *ref_g)
{
    int gid = get_global_id(0);

    cdouble_t q2, kn, k_next, rj, SLD, _t, beta, I;
    cdouble_t mrtot00, mrtot01, mrtot10, mrtot11;
    cdouble_t mi00, mi01, mi10, mi11, p0, p1;

    int ii;
    int nlayers;
    double rough, M_4PI, thick;

    M_4PI = 12.566370614359172;
    nlayers = (int) coefs_g[0];

    q2 = cdouble_fromreal(q_g[gid] * q_g[gid] / 4.);
    kn = cdouble_fromreal(q_g[gid] / 2.);

    mrtot00 = cdouble_fromreal(1.0);
    mrtot11 = cdouble_fromreal(1.0);
    mrtot01 = cdouble_fromreal(0.);
    mrtot10 = cdouble_fromreal(0.);
    I.real = 0.;
    I.imag = 1.;

    for(ii = 0 ; ii < nlayers + 1; ii += 1){
        if(ii == nlayers){
            // the backing medium
            SLD.real = M_4PI * (coefs_g[4] - coefs_g[2]) * 1e-6;
            SLD.imag = M_4PI * (fabs(coefs_g[5]) + 1e-30) * 1e-6;
            rough = -2. * coefs_g[7] * coefs_g[7];
        } else {
            SLD.real = M_4PI * (coefs_g[4*ii + 9] - coefs_g[2]) * 1e-6;
            SLD.imag = M_4PI * (fabs(coefs_g[4*ii + 10]) + 1e-30) * 1e-6;
            rough = -2 * coefs_g[4*ii + 11] * coefs_g[4*ii + 11];
        }

        // wavevector in next layer
        k_next = cdouble_sqrt(cdouble_sub(q2, SLD));

        // reflectance coefficient
        rj = cdouble_divide(cdouble_sub(kn, k_next), cdouble_add(kn, k_next));
        _t = cdouble_exp(cdouble_mulr(cdouble_mul(kn, k_next), rough));
        rj = cdouble_mul(rj, _t);

        if(ii == 0){
            mrtot01 = rj;
            mrtot10 = rj;
        } else {
            thick = coefs_g[4*(ii - 1) + 8];
            _t = cdouble_mul(kn, cdouble_mulr(I, thick));
            beta = cdouble_exp(_t);

            //this is the characteristic matrix of a layer
            mi00 = beta;
            mi11 = cdouble_rdivide(1., beta);
            mi10 = cdouble_mul(rj, mi00);
            mi01 = cdouble_mul(rj, mi11);

            // matrix multiply
            p0 = cdouble_add(cdouble_mul(mrtot00, mi00), cdouble_mul(mrtot10, mi01));
            p1 = cdouble_add(cdouble_mul(mrtot00, mi10), cdouble_mul(mrtot10, mi11));

            mrtot00 = p0;
            mrtot10 = p1;

            p0 = cdouble_add(cdouble_mul(mrtot01, mi00), cdouble_mul(mrtot11, mi01));
            p1 = cdouble_add(cdouble_mul(mrtot01, mi10), cdouble_mul(mrtot11, mi11));
            mrtot01 = p0;
            mrtot11 = p1;

        }
        kn = k_next;

    } // end of layer accumulation

    _t = cdouble_divide(mrtot01, mrtot00);
    _t = cdouble_mul(_t, cdouble_conj(_t));

    ref_g[gid] = _t.real;
    ref_g[gid] *= coefs_g[1];
    ref_g[gid] += coefs_g[6];
}
