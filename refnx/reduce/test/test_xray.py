import os
import refnx.reduce.xray as xray


class TestXray(object):

    def setup_method(self):
        self.pth = os.path.dirname(os.path.abspath(__file__))

    def test_reduction_runs(self):
        # just ensure that the reduction occurs without raising an Exception.
        # We're not testing for correctness here (yet)

        fpath = os.path.join(self.pth, '180706_HA_DG2.xrdml')
        xray.reduce_xrdml(fpath)
