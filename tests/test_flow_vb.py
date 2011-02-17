import unittest
from os.path import join
from flowvb import FlowVBAnalysis
from scipy.io import loadmat
from flowvb.utils import arrays_almost_equal
import numpy as np
from numpy.random import multivariate_normal as rmvnorm

TEST_DATA_LOC = join('tests', 'data', 'old_faithful')


class TestFaithul(unittest.TestCase):
    """Test with the Old Faithful data """

    def setUp(self):
        self.data = loadmat(join(TEST_DATA_LOC, 'faithful.mat'))['data']
        self.init = loadmat(join(TEST_DATA_LOC, 'faithful_init.mat'),
                            squeeze_me=True)
        self.model = FlowVBAnalysis(self.data,
                       init_mean=self.init['init_mean'],
                       init_covar=self.init['init_covar'],
                       init_mixweights=self.init['init_mixweights'],
                       thresh=1e-5, max_iter=200, verbose=False)

    def testMean(self):
        result = loadmat(join(TEST_DATA_LOC, 'faithful_final_mean.mat'),
                         squeeze_me=True)

        approx_equal = arrays_almost_equal(self.model.ESS.smm_mean,
                                           result['smm_mean'],
                                           accuracy=1e-1)

        self.assertTrue(approx_equal)


class TestRandom(unittest.TestCase):
    """Test with some synthetic data """

    def setUp(self):
        np.random.seed(0)

        self.mean = np.array([[0., -2.], [0., 0.], [0., 2.]])
        self.cov = np.array([[2., 0.], [0., .2]])

        n_obs = 2000

        data = np.vstack([np.array(rmvnorm(self.mean[k, :], self.cov, n_obs))
                          for k in range(self.mean.shape[0])])

        self.model = FlowVBAnalysis(data, 6, thresh=1e-6, init_method='random')

    def runTest(self):

        # Test for the correct number of components
        self.assertEqual(self.model.ESS.num_comp, 3)

        # Test the means
        sorted_mean = np.sort(self.model.ESS.smm_mean, 0)
        approx_equal = arrays_almost_equal(sorted_mean,
                                           self.mean,
                                           accuracy=1e-1)
        self.assertTrue(approx_equal)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFaithul)
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestRandom))
    unittest.TextTestRunner(verbosity=2).run(suite)
