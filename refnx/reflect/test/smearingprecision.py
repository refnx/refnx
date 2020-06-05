import refnx.reflect.reflect_model as reflect
import numpy as np
from time import time


def smearing_precision_comparison(maxorder=50):
    """
    a quick script to see how the smeared reflectivity changes as a function of
    smearing precision
    """
    # import q values and dqvalues from the smearing test
    theoretical = np.loadtxt("refnx/analysis/test/smeared_theoretical.txt")
    qvals, rvals, dqvals = np.hsplit(theoretical, 3)
    qvals = qvals.flatten()
    dqvals = dqvals.flatten()

    # coefficients for the model
    a = np.zeros((12))
    a[0] = 1.0
    a[1] = 1.0
    a[4] = 2.07
    a[7] = 3
    a[8] = 100
    a[9] = 3.47
    a[11] = 2

    # setup an array for the smeared rvals.
    smeared_rvals = np.zeros((maxorder + 1, qvals.size))

    # now output all the smearing.
    t0 = time()
    smeared_rvals[0, :] = reflect.reflectivity(
        qvals, a, **{"dqvals": dqvals, "quad_order": "ultimate"}
    )
    t1 = time()
    print("ultimate takes %f" % (t1 - t0))

    for idx in range(1, maxorder + 1):
        t0 = time()
        smeared_rvals[idx, :] = reflect.reflectivity(
            qvals, a, **{"dqvals": dqvals, "quad_order": idx}
        )
        t1 = time()
        print(idx, " takes %f" % (t1 - t0))
    np.savetxt("smearing_comp", smeared_rvals.T)


if __name__ == "__main__":
    smearing_precision_comparison()
