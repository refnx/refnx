from __future__ import division, print_function
import numpy as np


"""
import numpy as np
q = np.linspace(0.01, 0.5, 1000)
w = np.array([[0, 2.07, 0, 0],
              [100, 3.47, 0, 3],
              [500, -0.5, 0.00001, 3],
              [0, 6.36, 0, 3]])
"""


def abeles(q, layers, scale=1., bkg=0, threads=0):
    """
    Abeles matrix formalism for calculating reflectivity from a stratified
    medium.

    Parameters
    ----------
    layers: np.ndarray
        coefficients required for the calculation, has shape (2 + N, 4),
        where N is the number of layers
        layers[0, 1] - SLD of fronting (/1e-6 Angstrom**-2)
        layers[0, 2] - iSLD of fronting (/1e-6 Angstrom**-2)
        layers[N, 0] - thickness of layer N
        layers[N, 1] - SLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 2] - iSLD of layer N (/1e-6 Angstrom**-2)
        layers[N, 3] - roughness between layer N-1/N
        layers[-1, 1] - SLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 2] - iSLD of backing (/1e-6 Angstrom**-2)
        layers[-1, 3] - roughness between backing and last layer
    q: array_like
        the q values required for the calculation.
        Q = 4 * Pi / lambda * sin(omega).
        Units = Angstrom**-1
    scale: float
        Multiply all reflectivities by this value.
    bkg: float
        Linear background to be added to all reflectivities
    threads: int, optional
        <THIS OPTION IS CURRENTLY IGNORED>

    Returns
    -------
    Reflectivity: np.ndarray
        Calculated reflectivity values for each q value.
    """
    qvals = np.asfarray(q)
    flatq = qvals.ravel()

    nlayers = layers.shape[0] - 2
    npnts = flatq.size

    kn = np.zeros((npnts, nlayers + 2), np.complex128)

    sld = np.zeros(nlayers + 2, np.complex128)
    sld[:] += ((layers[:, 1] - layers[0, 1]) +
               1j * (layers[:, 2] - layers[0, 2])) * 1.e-6

    # kn is a 2D array. Rows are Q points, columns are kn in a layer.
    # calculate wavevector in each layer, for each Q point.
    kn[:] = np.sqrt(flatq[:, np.newaxis] ** 2. / 4. - 4. * np.pi * sld)

    # initialise matrix total
    mrtot00 = 1
    mrtot11 = 1
    mrtot10 = 0
    mrtot01 = 0
    k = kn[:, 0]

    for idx in range(1, nlayers + 2):
        k_next = kn[:, idx]

        # reflectance of an interface
        rj = (k - k_next) / (k + k_next)
        rj *= np.exp(k * k_next * -2. * layers[idx, 3] ** 2)

        # work out characteristic matrix of layer
        mi00 = np.exp(k * 1j * np.fabs(layers[idx - 1, 0])) if idx - 1 else 1
        # mi11 = (np.exp(k * -1j * np.fabs(layers[idx - 1, 0]))
        #         if idx - 1 else 1)
        mi11 = 1 / mi00 if idx - 1 else 1

        mi10 = rj * mi00
        mi01 = rj * mi11

        # matrix multiply mrtot by characteristic matrix
        p0 = mrtot00 * mi00 + mrtot10 * mi01
        p1 = mrtot00 * mi10 + mrtot10 * mi11
        mrtot00 = p0
        mrtot10 = p1

        p0 = mrtot01 * mi00 + mrtot11 * mi01
        p1 = mrtot01 * mi10 + mrtot11 * mi11

        mrtot01 = p0
        mrtot11 = p1

        k = k_next

    reflectivity = (mrtot01 * np.conj(mrtot01)) / (mrtot00 * np.conj(mrtot00))
    reflectivity *= scale
    reflectivity += bkg
    return np.real(np.reshape(reflectivity, qvals.shape))


"""
PNR calculation
"""
def pmatrix(qu, qd, dspac):
    M_p = np.zeros((4, 4), np.complex128)
    # TODO: this is setting leading diagonal, use np.fill_diagonal
    # TODO: reduce computation, there is symmetry in some of the operations.
    M_p[0, 0] = np.exp(complex(0, -1) * qu * dspac)
    M_p[1, 1] = np.exp(complex(0, 1) * qu * dspac)
    M_p[2, 2] = np.exp(complex(0, -1) * qd * dspac)
    M_p[3, 3] = np.exp(complex(0, 1) * qd * dspac)

    return M_p


def dmatrix(qu, qd):
    M_d = np.zeros((4, 4), np.complex128)

    M_d[0][0] = 1
    M_d[0][1] = 1
    M_d[1][0] = qu
    M_d[1][1] = -qu

    M_d[2][2] = 1
    M_d[2][3] = 1
    M_d[3][2] = qd
    M_d[3][3] = -qd

    return M_d


def qcal(qq, nb):
    # I think this is calculating kn_n
    return np.sqrt(qq**2 - 4 * np.pi * nb)


Function RRcalc(theta)
variable theta
make/o/d/c/n=(4,4) M_RR = 0
theta/=2
M_RR[0][0] = cos(theta)*cmplx(1,0)
M_RR[1][1] = cos(theta)*cmplx(1,0)

M_RR[0][2] = sin(theta)*cmplx(1,0)
M_RR[1][3] = sin(theta)*cmplx(1,0)

M_RR[2][0] = -sin(theta)*cmplx(1,0)
M_RR[3][1] = -sin(theta)*cmplx(1,0)

M_RR[2][2] = cos(theta)*cmplx(1,0)
M_RR[3][3] = cos(theta)*cmplx(1,0)

End

Function pnr(w,xx)
    Wave w,xx
    //ww parameter wave, xx is the number of x points for calculation
    //w[0] = number of layers
    //w[1] = scale
    //	[2] = sldupper
    //	[3] = Bupper
    //	[4] = thetaupper
    //	[5] = sldlower
    //	[6] = Blower

    //	[ 4n+7] = d n
    //	[ 4n+8] = sld n
    //	[ 4n+9] = b n
    //	[ 4n+10] = theta n

    make/o/d/n=(numpnts(xx),4) M_pnr
    make/o/d/n=(4,4)/c M,MM,RR

    variable nb_air=w[2]
    variable nb_sub=w[5]
    variable ii,jj
    for(ii=0;ii<numpnts(xx);iI+=1)
        variable qvac = xx[ii]*0.5
        //   qvac(iq)=(qmin+float(iq)*qstep)*0.5
        variable/c qair_u=qcal(qvac,nb_air+w[3])
        variable/c qair_d=qcal(qvac,nb_air-w[3])

        variable/c qsub_u=qcal(qvac,(nb_sub+w[6]))
        variable/c qsub_d=qcal(qvac,(nb_sub-w[6]))
        MM = 0
        MM[0][0] = cmplx(1,0)
        MM[1][1] = cmplx(1,0)
        MM[2][2] = cmplx(1,0)
        MM[3][3] = cmplx(1,0)

        for( jj=0;jj<w[0];jj+=1)
            variable/c qu = qcal(qvac,w[4*jj+8]+w[4*jj+9])
            variable/c qd = qcal(qvac,w[4*jj+8]-w[4*jj+9])
            variable/c thetai
            if(jj==0)
                thetai = (w[4])*Pi/180
            else
                thetai = (w[4*jj+10])*Pi/180
            endif

            RRcalc(thetai)

            dmatrix(qu,qd)	//create M_d
            pmatrix(qu,qd,w[4*jj+7])

            Wave M_d,M_p,M_RR
            MatrixOp/O MM = MM x M_d x M_p x (inv(M_d) x M_RR)
        endfor

        RRcalc(Pi/180*w[4])

        dmatrix(qair_u,qair_d)
        Wave M_d,M_RR

        MatrixOp/o M = (inv(M_d)) x M_RR x MM
        dmatrix(qsub_u,qsub_d)
        MatrixOp/o M = M x M_d

        M_pnr[ii][0] = magsqr((M[1][0]*M[2][2]-M[1][2]*M[2][0])/(M[0][0]*M[2][2]-M[0][2]*M[2][0]))//uu
        M_pnr[ii][1] = magsqr((M[3][2]*M[0][0]-M[3][0]*M[0][2])/(M[0][0]*M[2][2]-M[0][2]*M[2][0]))//dd
        M_pnr[ii][2] = magsqr((M[3][0]*M[2][2]-M[3][2]*M[2][0])/(M[0][0]*M[2][2]-M[0][2]*M[2][0]))//ud
        M_pnr[ii][3] = magsqr((M[1][2]*M[0][0]-M[1][0]*M[0][2])/(M[0][0]*M[2][2]-M[0][2]*M[2][0]))//du

    endfor

End


if __name__ == '__main__':
    a = np.zeros(12)
    a[0] = 1.
    a[1] = 1.
    a[4] = 2.07
    a[7] = 3
    a[8] = 100
    a[9] = 3.47
    a[11] = 2

    b = np.arange(1000.)
    b /= 2000.
    b += 0.001

    def loop():
        abeles(b, a)

    for i in range(1000):
        loop()
