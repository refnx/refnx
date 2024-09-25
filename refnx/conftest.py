# contents of conftest.py
import zipfile
import shutil
import urllib.request
from pathlib import Path

import pytest

import refnx


@pytest.fixture(scope="session")
def data_directory(tmp_path_factory):
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
    tmpdir = tmp_path_factory.mktemp("data")

    try:
        # grab the test data
        with (
            urllib.request.urlopen(url, timeout=5) as response,
            open(tmpdir / "master.zip", "wb") as f,
        ):
            shutil.copyfileobj(response, f)

        # master.zip is in tmpdir
        with zipfile.ZipFile(tmpdir / "master.zip") as zf:
            zf.extractall(path=tmpdir)

        data_dir = tmpdir / "refnx-testdata-master" / "data"
    except (urllib.error.URLError, TimeoutError):
        pytest.skip("No data directory available")

    return data_dir
