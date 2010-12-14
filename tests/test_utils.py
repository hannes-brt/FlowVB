import unittest
from flowvb.utils import repeat, normalize_logspace
from random import random, randint
import numpy as np


class TestNormalizeLogspace(unittest.TestCase):

    N_REPEATS = 100
    MAX_LEN_VECTOR = 1e6

    def testSumToOne(self):
        for _ in range(self.N_REPEATS):
            len_vector = randint(10, self.MAX_LEN_VECTOR)
            x = repeat(len_vector, random)
            x = normalize_logspace(x)
            s = np.sum(x)
            self.assertAlmostEqual(s, 1.0, 12)


if __name__ == '__main__':
        suite = unittest.TestLoader().loadTestsFromTestCase(
            TestNormalizeLogspace)
        unittest.TextTestRunner(verbosity=2).run(suite)
