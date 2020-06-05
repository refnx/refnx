# contents of conftest.py
import zipfile
import shutil
import urllib.request
from os.path import join as pjoin

import pytest

import refnx


@pytest.fixture(scope="session")
def data_directory(tmpdir_factory):
    """
    Retrieves the refnx-testdata repository, placing it in a temporary
    directory, for use in pytest fixtures

    Returns
    -------
    data_dir: str or None
        If the retrieval works then a str pointing to the test data is
        returned. If the retrieval fails then None is returned.
    """
    url = "https://github.com/refnx/refnx-testdata/archive/master.zip"
    tmpdir = tmpdir_factory.mktemp("data")

    try:
        # grab the test data
        with urllib.request.urlopen(url, timeout=5) as response, open(
            pjoin(tmpdir, "master.zip"), "wb"
        ) as f:
            shutil.copyfileobj(response, f)

        # master.zip is in tmpdir
        with zipfile.ZipFile(pjoin(tmpdir, "master.zip")) as zf:
            zf.extractall(path=tmpdir)

        data_dir = pjoin(tmpdir, "refnx-testdata-master", "data")
    except urllib.error.URLError:
        data_dir = None

    return data_dir


@pytest.fixture(scope="session")
def no_data_directory(data_directory):
    if data_directory is None:
        pytest.skip("No data directory available")
