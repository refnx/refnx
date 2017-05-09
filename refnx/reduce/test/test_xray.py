import unittest
import os
import refnx.reduce.xray as xray


class TestXray(unittest.TestCase):

    def setUp(self):
        path = os.path.dirname(os.path.realpath(__file__))
        self.path = path

        return 0

    def test_reduction_runs(self):
        # just ensure that the reduction occurs without raising an Exception.
        # We're not testing for correctness here (yet)

        fpath = os.path.join(self.path, '180706_HA_DG2.xrdml')
        xray.reduce_xrdml(fpath)


if __name__ == '__main__':
    unittest.main()
