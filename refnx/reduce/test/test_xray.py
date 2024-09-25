import pytest
import refnx.reduce.xray as xray


def test_reduction_runs(data_directory):
    # just ensure that the reduction occurs without raising an Exception.
    # We're not testing for correctness here (yet)
    fpath = data_directory / "reduce" / "180706_HA_DG2.xrdml"
    xray.reduce_xrdml(fpath)
