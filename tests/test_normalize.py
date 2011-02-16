import unittest
from flowvb.utils import arrays_almost_equal
from flowvb.normalize import normalize_logspace
import numpy as np
from numpy.random import random

NROW = 1000
NCOL = 20
ACC = 1e-7


class TestNormalizeLogspace(unittest.TestCase):
    """Test `normalize_logspace`"""

    def testC(self):
        """Test if `normalize_logspace` works with a C-contiguous array"""
        np.random.seed(0)
        mat = random((NROW, NCOL))
        self.assertTrue(mat.flags['C_CONTIGUOUS'])
        mat_out = normalize_logspace(mat)
        row_sum = mat_out.sum(1)
        approx_equal = arrays_almost_equal(row_sum, np.ones(NROW),
                                           accuracy=ACC)
        self.assertTrue(approx_equal)

    def testFortran(self):
        """Test if `normalize_logspace` works with a F-contiguous array"""
        np.random.seed(0)
        mat = random((NCOL, NROW)).T
        self.assertTrue(mat.flags['F_CONTIGUOUS'])
        mat_out = normalize_logspace(mat)
        row_sum = mat_out.sum(1)
        approx_equal = arrays_almost_equal(row_sum, np.ones(NROW),
                                           accuracy=ACC)
        self.assertTrue(approx_equal)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNormalizeLogspace)
    unittest.TextTestRunner(verbosity=2).run(suite)
