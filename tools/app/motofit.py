import multiprocessing
import warnings
import sys


if __name__ == "__main__":
    # needed for windows using multiprocessing!
    multiprocessing.freeze_support()
    warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*",
                            category=UserWarning)
    from refnx.reflect import main
    main()
