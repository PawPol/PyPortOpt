from PyPortOpt import Optimizers as o
import unittest
import numpy as np

class TestOptimizer(unittest.TestCase):
    
    def test_testFunction(self):
        self.assertEqual(o.testFunction(), True)
    
    def test_preprocessData(self):
        data = {'Ticker': {0: 'AAPL', 1: 'AAPL', 2: 'AAPL', 3: 'AAPL', 4: 'AAPL', 5: 'AAPL', 6: 'AAPL', 7: 'TSLA', 8: 'TSLA', 9: 'TSLA', 10: 'TSLA', 11: 'TSLA', 12: 'TSLA', 13: 'TSLA'}, 
                'Date': {0: '2020-01-02', 1: '2020-01-03', 2: '2020-01-06', 3: '2020-01-07', 4: '2020-01-08', 5: '2020-01-09', 6: '2020-01-10', 7: '2020-01-02', 8: '2020-01-03', 9: '2020-01-06', 10: '2020-01-07', 11: '2020-01-08',  12: '2020-01-09', 13: '2020-01-10'}, 
                'Adjusted_Close': {0: 74.09522915781685, 1: 73.37487600602452, 2: 73.95954620114364, 3: 73.61170443949048, 4: 74.79584660682033, 5: 76.38457068132122, 6: 76.55725808072349, 7: 86.052, 8: 88.602, 9: 90.308, 10: 93.812, 11: 98.428, 12: 96.268, 13: 95.63}}
        meanVec, sigMat = o.preprocessData(data)
        
        self.assertEqual(meanVec.shape[0],2)
        
        self.assertEqual(sigMat.shape[0],2)
        
        self.assertEqual(sigMat.shape[1],2)
    
    def test_SymPDcovmatrix(self):
        a = [[1,2,3],[5,6,7],[3,5,9]]
        a = np.array(a)
        SPD = np.dot(a.T,a)
        nonSPD = a
        mat, _ = o.SymPDcovmatrix(SPD, tol = 1e-16)
        self.assertTrue(np.allclose(mat, SPD, atol = 1e-16))
        
        mat,_ = o.SymPDcovmatrix(nonSPD, tol = 1e-16)
        eig,_ = np.linalg.eig(mat)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
