import multiprocessing
import numpy
import scipy
import matplotlib
import periodictable


if __name__ == "__main__":
    # needed for windows using multiprocessing!
    multiprocessing.freeze_support()

    from refnx.reflect import main
    main()
