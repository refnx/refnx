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
    """
    tmpdir = tmpdir_factory.mktemp("data")
    with urllib.request.urlopen(
        "https://github.com/refnx/refnx-testdata/archive/master.zip"
    ) as response, open(pjoin(tmpdir, 'master.zip'), 'wb') as f:
        shutil.copyfileobj(response, f)

    # master.zip is in tmpdir
    with zipfile.ZipFile(pjoin(tmpdir, 'master.zip')) as zf:
        zf.extractall(path=tmpdir)

    return pjoin(tmpdir, 'refnx-testdata-master', 'data')
