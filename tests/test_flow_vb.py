from os.path import join
import unittest

import numpy as np
from numpy.random import multivariate_normal as rmvnorm
from scipy.io import loadmat

from flowvb import FlowVBAnalysis
from flowvb.core.flow_vb import Options
from flowvb.initialize import RandomInitialiser
from flowvb.utils import arrays_almost_equal

TEST_DATA_LOC = join('../', 'tests', 'data', 'old_faithful')

class FlowVBAnalysisTest(unittest.TestCase):
    pass

class TestFaithul(FlowVBAnalysisTest):
    """Test with the Old Faithful data """

    def setUp(self):
        self.data = loadmat(join(TEST_DATA_LOC, 'faithful.mat'))['data']
        self.init = loadmat(join(TEST_DATA_LOC, 'faithful_init.mat'),
                            squeeze_me=True)
                        
        init_params = {}
        init_params['mean'] = self.init['init_mean']
        init_params['covar'] = self.init['init_covar']
        init_params['mixweights'] = self.init['init_mixweights']
        
        options = Options(init_params)        
        self.model = FlowVBAnalysis(self.data, options)

    def testMean(self):
        result = loadmat(join(TEST_DATA_LOC, 'faithful_final_mean.mat'),
                         squeeze_me=True)

        approx_equal = arrays_almost_equal(self.model.ess.smm_mean,
                                           result['smm_mean'],
                                           accuracy=1e-1)

        self.assertTrue(approx_equal)


class TestRandom(FlowVBAnalysisTest):
    """Test with some synthetic data """

    def setUp(self):
        np.random.seed(0)

        self.mean = np.array([[0., -2.], [0., 0.], [0., 2.]])
        self.cov = np.array([[2., 0.], [0., .2]])

        n_obs = 2000

        data = np.vstack([np.array(rmvnorm(self.mean[k, :], self.cov, n_obs))
                          for k in range(self.mean.shape[0])])                        

        initialiser = RandomInitialiser()
        init_params = initialiser.initialise_parameters(data, 6)
        
        options = Options(init_params)
        
        self.model = FlowVBAnalysis(data, options)

    def runTest(self):

        # Test for the correct number of components
        self.assertEqual(self.model.ess.num_comp, 3)

        # Test the means
        sorted_mean = np.sort(self.model.ess.smm_mean, 0)
        approx_equal = arrays_almost_equal(sorted_mean,
                                           self.mean,
                                           accuracy=1e-1)
        self.assertTrue(approx_equal)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFaithul)
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestRandom))
    unittest.TextTestRunner(verbosity=2).run(suite)
