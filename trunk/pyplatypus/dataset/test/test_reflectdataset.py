import unittest
import pyplatypus.reduce.reflectdataset as reflectdataset
import numpy as np
import numpy.testing as npt

class TestReflectDataset(unittest.TestCase):

    def setUp(self):
        pass
             
    def test_load_reflectivity_XML(self):
        '''
            test reflectivity calculation
            with values generated from Motofit
        
        '''
        dataset = reflectdataset.ReflectDataset()
        with open('pyplatypus/reduce/test/c_PLP0000708.xml') as f:
            dataset.load_reflectivity_XML(f)
        
        self.assertEqual(dataset.numpoints, 90)
        self.assertEqual(90, np.size(dataset.W_q))
        
if __name__ == '__main__':
    unittest.main()