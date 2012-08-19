import unittest
import pyplatypus.dataset.reflectdataset as reflectdataset
import numpy as np
import numpy.testing as npt

class TestReflectDataset(unittest.TestCase):

    def setUp(self):
        pass
             
    def test_load(self):
        '''
            test reflectivity calculation
            with values generated from Motofit
        
        '''
        dataset = reflectdataset.ReflectDataset()
        with open('pyplatypus/dataset/test/c_PLP0000708.xml') as f:
            dataset.load(f)
        
        self.assertEqual(dataset.numpoints, 90)
        self.assertEqual(90, np.size(dataset.W_q))
        
        dataset1 = reflectdataset.ReflectDataset()
        with open('pyplatypus/dataset/test/c_PLP0000708.dat') as f:
            dataset1.load(f)
        
        self.assertEqual(dataset1.numpoints, 90)
        self.assertEqual(90, np.size(dataset1.W_q))
        
        
if __name__ == '__main__':
    unittest.main()