import os

import numpy as np
from numpy import exp, sin, cos, arctan, array, pi
from numpy.testing import assert_allclose, assert_

from refnx.analysis import CurveFitter, Objective, Parameter, Parameters, Model

thisdir, thisfile = os.path.split(__file__)
NIST_DIR = os.path.join(thisdir, "NIST_STRD")


def ndig(a, b):
    "precision for NIST values"
    return np.round(
        -np.log10((np.abs(np.abs(a) - np.abs(b)) + 1.0e-15) / np.abs(b))
    )


def read_params(params):
    return np.array(params)


def Bennett5(x, b):
    b = read_params(b)
    return b[0] * (b[1] + x) ** (-1 / b[2])


def BoxBOD(x, b):
    b = read_params(b)
    return b[0] * (1 - exp(-b[1] * x))


def Chwirut(x, b):
    b = read_params(b)
    return exp(-b[0] * x) / (b[1] + b[2] * x)


def DanWood(x, b):
    b = read_params(b)
    return b[0] * x ** b[1]


def ENSO(x, b):
    b = read_params(b)
    return b[0] + (
        b[1] * cos(2 * pi * x / 12)
        + b[2] * sin(2 * pi * x / 12)
        + b[4] * cos(2 * pi * x / b[3])
        + b[5] * sin(2 * pi * x / b[3])
        + b[7] * cos(2 * pi * x / b[6])
        + b[8] * sin(2 * pi * x / b[6])
    )


def Eckerle4(x, b):
    b = read_params(b)
    return (b[0] / b[1]) * exp(-0.5 * ((x - b[2]) / b[1]) ** 2)


def Gauss(x, b):
    b = read_params(b)
    return b[0] * exp(-b[1] * x) + (
        b[2] * exp(-((x - b[3]) ** 2) / b[4] ** 2)
        + b[5] * exp(-((x - b[6]) ** 2) / b[7] ** 2)
    )


def Hahn1(x, b):
    b = read_params(b)
    return (b[0] + b[1] * x + b[2] * x**2 + b[3] * x**3) / (
        1 + b[4] * x + b[5] * x**2 + b[6] * x**3
    )


def Kirby(x, b):
    b = read_params(b)
    return (b[0] + b[1] * x + b[2] * x**2) / (1 + b[3] * x + b[4] * x**2)


def Lanczos(x, b):
    b = read_params(b)
    return (
        b[0] * exp(-b[1] * x) + b[2] * exp(-b[3] * x) + b[4] * exp(-b[5] * x)
    )


def MGH09(x, b):
    b = read_params(b)
    return b[0] * (x**2 + x * b[1]) / (x**2 + x * b[2] + b[3])


def MGH10(x, b):
    b = read_params(b)
    return b[0] * exp(b[1] / (x + b[2]))


def MGH17(x, b):
    b = read_params(b)
    return b[0] + b[1] * exp(-x * b[3]) + b[2] * exp(-x * b[4])


def Misra1a(x, b):
    b = read_params(b)
    return b[0] * (1 - exp(-b[1] * x))


def Misra1b(x, b):
    b = read_params(b)
    return b[0] * (1 - (1 + 0.5 * b[1] * x) ** (-2))


def Misra1c(x, b):
    b = read_params(b)
    return b[0] * (1 - (1 + 2 * b[1] * x) ** (-0.5))


def Misra1d(x, b):
    b = read_params(b)
    return b[0] * b[1] * x * ((1 + b[1] * x) ** (-1))


def Nelson(x, b):
    b = read_params(b)
    x1 = x[:, 0]
    x2 = x[:, 1]
    return b[0] - b[1] * x1 * exp(-b[2] * x2)


def Rat42(x, b):
    b = read_params(b)
    return b[0] / (1 + exp(b[1] - b[2] * x))


def Rat43(x, b):
    b = read_params(b)
    return b[0] / ((1 + exp(b[1] - b[2] * x)) ** (1 / b[3]))


def Roszman1(x, b):
    b = read_params(b)
    return b[0] - b[1] * x - arctan(b[2] / (x - b[3])) / pi


def Thurber(x, b):
    b = read_params(b)
    return (b[0] + b[1] * x + b[2] * x**2 + b[3] * x**3) / (
        1 + b[4] * x + b[5] * x**2 + b[6] * x**3
    )


#  Model name        fcn,    #fitting params, dim of x
NIST_Models = {
    "Bennett5": (Bennett5, 3, 1),
    "BoxBOD": (BoxBOD, 2, 1),
    "Chwirut1": (Chwirut, 3, 1),
    "Chwirut2": (Chwirut, 3, 1),
    "DanWood": (DanWood, 2, 1),
    "ENSO": (ENSO, 9, 1),
    "Eckerle4": (Eckerle4, 3, 1),
    "Gauss1": (Gauss, 8, 1),
    "Gauss2": (Gauss, 8, 1),
    "Gauss3": (Gauss, 8, 1),
    "Hahn1": (Hahn1, 7, 1),
    "Kirby2": (Kirby, 5, 1),
    "Lanczos1": (Lanczos, 6, 1),
    "Lanczos2": (Lanczos, 6, 1),
    "Lanczos3": (Lanczos, 6, 1),
    "MGH09": (MGH09, 4, 1),
    "MGH10": (MGH10, 3, 1),
    "MGH17": (MGH17, 5, 1),
    "Misra1a": (Misra1a, 2, 1),
    "Misra1b": (Misra1b, 2, 1),
    "Misra1c": (Misra1c, 2, 1),
    "Misra1d": (Misra1d, 2, 1),
    "Nelson": (Nelson, 3, 2),
    "Rat42": (Rat42, 3, 1),
    "Rat43": (Rat43, 4, 1),
    "Roszman1": (Roszman1, 4, 1),
    "Thurber": (Thurber, 7, 1),
}


def NIST_runner(
    dataset,
    method="least_squares",
    chi_atol=1e-5,
    val_rtol=1e-2,
    err_rtol=6e-3,
):
    NIST_dataset = ReadNistData(dataset)
    x, y = (NIST_dataset["x"], NIST_dataset["y"])

    if dataset == "Nelson":
        y = np.log(y)

    params = NIST_dataset["start"]

    fitfunc = NIST_Models[dataset][0]
    model = Model(params, fitfunc)
    objective = Objective(model, (x, y))
    fitter = CurveFitter(objective)
    result = fitter.fit(method=method)

    assert_allclose(
        objective.chisqr(), NIST_dataset["sum_squares"], atol=chi_atol
    )

    certval = NIST_dataset["cert_values"]
    assert_allclose(result.x, certval, rtol=val_rtol)

    if "stderr" in result:
        certerr = NIST_dataset["cert_stderr"]
        assert_allclose(result.stderr, certerr, rtol=err_rtol)


def ReadNistData(dataset, start="start2"):
    """
    NIST STRD data is in a simple, fixed format with line numbers being
    significant!
    """
    with open(os.path.join(NIST_DIR, "%s.dat" % dataset), "r") as finp:
        lines = [line[:-1] for line in finp.readlines()]

    model_lines = lines[30:39]
    param_lines = lines[40:58]
    data_lines = lines[60:]

    words = model_lines[1].strip().split()
    nparams = int(words[0])

    start1 = np.zeros(nparams)
    start2 = np.zeros(nparams)
    certval = np.zeros(nparams)
    certerr = np.zeros(nparams)

    for i, text in enumerate(param_lines[:nparams]):
        [s1, s2, val, err] = [float(x) for x in text.split("=")[1].split()]
        start1[i] = s1
        start2[i] = s2
        certval[i] = val
        certerr[i] = err

    for t in param_lines[nparams:]:
        t = t.strip()
        if ":" not in t:
            continue
        val = float(t.split(":")[1])
        if t.startswith("Residual Sum of Squares"):
            sum_squares = val
        elif t.startswith("Residual Standard Deviation"):
            std_dev = val
        elif t.startswith("Degrees of Freedom"):
            nfree = int(val)
        elif t.startswith("Number of Observations"):
            ndata = int(val)

    y, x = [], []
    for d in data_lines:
        vals = [float(i) for i in d.strip().split()]
        y.append(vals[0])
        if len(vals) > 2:
            x.append(vals[1:])
        else:
            x.append(vals[1])

    y = array(y)
    x = array(x)

    params = Parameters()
    for i in range(nparams):
        pname = "p%i" % (i + 1)
        if start == "start2":
            pval = start2[i]
        elif start == "start1":
            pval = start1[i]
        p = Parameter(pval, name=pname, vary=True)
        params.append(p)

    out = {
        "y": y,
        "x": x,
        "nparams": nparams,
        "ndata": ndata,
        "nfree": nfree,
        "start": params,
        "sum_squares": sum_squares,
        "std_dev": std_dev,
        "cert_values": certval,
        "cert_stderr": certerr,
    }

    return out
