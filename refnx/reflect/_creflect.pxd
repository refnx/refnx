cimport numpy as np

cpdef np.ndarray abeles(
    np.ndarray x,
    double[:, :] w,
    double scale=?,
    double bkg=?,
    int threads=?,
)
