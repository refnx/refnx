from refnx.dataset.data1d import Data1D
from refnx.dataset.reflectdataset import ReflectDataset, OrsoDataset
from refnx._lib._testutils import PytestTester
from refnx._lib import possibly_open_file as _possibly_open_file

test = PytestTester(__name__)
del PytestTester


def load_data(f):
    """
    Loads a dataset

    Parameters
    ----------
    f: {file-like, str}
        f can be a string or file-like object referring to a File to
        load the dataset from.

    Returns
    -------
    data: Data1D-like
        data object
    """
    try:
        data = OrsoDataset(f)
        return data
    except Exception:
        # not an ORSO file
        pass

    try:
        d = ReflectDataset(f)
        return d
    except Exception:
        pass

    d = Data1D(f)
    return d


__all__ = [s for s in dir() if not s.startswith("_")]
