import os
import pytest
import refnx.reduce.xray as xray


@pytest.mark.usefixtures("no_data_directory")
def test_reduction_runs(data_directory):
    # just ensure that the reduction occurs without raising an Exception.
    # We're not testing for correctness here (yet)
    fpath = os.path.join(data_directory, "reduce", "180706_HA_DG2.xrdml")
    xray.reduce_xrdml(fpath)
