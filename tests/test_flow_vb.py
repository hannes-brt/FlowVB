import unittest
from os.path import join
from flowvb import FlowVB
from scipy.io import loadmat
from flowvb.utils import arrays_almost_equal

TEST_DATA_LOC = join('tests', 'data', 'old_faithful')


class TestFaithul(unittest.TestCase):
    def setUp(self):
        self.data = loadmat(join(TEST_DATA_LOC, 'faithful.mat'))['data']
        self.init = loadmat(join(TEST_DATA_LOC, 'faithful_init.mat'),
                            squeeze_me=True)

    def testMean(self):
        result = loadmat(join(TEST_DATA_LOC, 'faithful_final_mean.mat'),
                         squeeze_me=True)

        model = FlowVB(self.data,
                       init_mean=self.init['init_mean'],
                       init_covar=self.init['init_covar'],
                       init_mixweights=self.init['init_mixweights'],
                       thresh=1e-5, max_iter=200, verbose=False)

        approx_equal = arrays_almost_equal(model.ESS.smm_mean,
                                           result['smm_mean'],
                                           accuracy=1e-1)

        self.assertTrue(approx_equal)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFaithul)
    unittest.TextTestRunner(verbosity=2).run(suite)
