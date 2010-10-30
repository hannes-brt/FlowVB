import unittest
import random as rd
import numpy as np
from scipy.io import loadmat
from os.path import join
from flowvb.core._posterior import _Posterior
from flowvb.utils import normalize, arrays_almost_equal

TEST_ACCURACY = 3
MAX_DIFF = pow(10, -TEST_ACCURACY)
TEST_DATA_LOC = join('tests', 'data', 'old_faithful')


class TestSetUp(unittest.TestCase):
    def setUp(self):
        rd.seed(1)
        self.repeats = 100
        self.max_num_obs = 1e6
        self.max_num_comp = 20


class TestUpdateDirichlet(TestSetUp):
    def testReturnValue(self):
        """ Test return values of `_update_dirichlet` with random numbers """
        for i in range(self.repeats):
            num_obs = rd.randint(1, self.max_num_obs)
            num_comp = rd.randint(1, self.max_num_comp)
            e_mixweights = normalize([rd.uniform(0, 1) for i in
                                      range(num_comp)])
            prior_dirichlet = [rd.uniform(0, 1) for i in range(num_comp)]

            posterior_dirichlet = _Posterior._update_dirichlet(
                num_obs, e_mixweights, prior_dirichlet)

            self.assertEqual(len(posterior_dirichlet), num_comp)
            self.assertEqual(type(posterior_dirichlet), type(np.array(1)))

    def testFaithful(self):
        """ Test `_update_dirichlet` with some data from Old Faithful """
        dl = loadmat(join(TEST_DATA_LOC, 'posterior_dirichlet.mat'),
                     squeeze_me=True)
        posterior_dirichlet_test = _Posterior._update_dirichlet(
            dl['num_obs'], dl['smm_mixweights'], dl['prior_dirichlet'])

        approx_equal = arrays_almost_equal(dl['posterior_dirichlet'],
                                           posterior_dirichlet_test,
                                           accuracy=MAX_DIFF)
        self.assertTrue(approx_equal)


class TestUpdateNwsScale(TestSetUp):
    def testFaithful(self):
        """Test `_update_nws_scale` with some data from Old Faithful """
        nwss = loadmat(join(TEST_DATA_LOC, 'posterior_nws_scale.mat'),
                       squeeze_me=True)

        nws_scale_test = _Posterior._update_nws_scale(nwss['num_obs'],
                            nwss['latent_scaled_resp'],
                            nwss['prior_nws_scale'])

        approx_equal = arrays_almost_equal(nwss['posterior_nws_scale'],
                                           nws_scale_test,
                                           MAX_DIFF)
        self.assertTrue(approx_equal)


class TestUpdateNwsMean(TestSetUp):
    def testFaithful(self):
        """ Test `_update_nws_mean` with some data from Old Faithful """
        nwsm = loadmat(join(TEST_DATA_LOC, 'posterior_nws_mean.mat'),
                       squeeze_me=True)

        nws_mean_test = _Posterior._update_nws_mean(
            nwsm['num_obs'],
            nwsm['num_comp'],
            nwsm['latent_scaled_resp'],
            nwsm['smm_mean'],
            nwsm['prior_nws_scale'],
            nwsm['posterior_nws_scale'],
            nwsm['prior_nws_mean'])

        self.assertEqual(nws_mean_test.shape,
                         (nwsm['num_comp'], nwsm['num_dim']))

        approx_equal = arrays_almost_equal(nws_mean_test,
                                           nwsm['posterior_nws_mean'],
                                           accuracy=MAX_DIFF)
        self.assertTrue(approx_equal)


class TestUpdateNwsDof(TestSetUp):
    def testFaithful(self):
        """ Test `_update_nws_dof` with some data from Old Faithful """
        nwsd = loadmat(join(TEST_DATA_LOC, 'posterior_nws_dof.mat'),
                       squeeze_me=True)

        nws_dof_test = _Posterior._update_nws_dof(nwsd['num_obs'],
                                                  nwsd['smm_mixweights'],
                                                  nwsd['prior_nws_dof'])

        self.assertEqual(nws_dof_test.size, nwsd['num_comp'])
        approx_equal = arrays_almost_equal(nws_dof_test,
                                           nwsd['posterior_nws_dof'],
                                           accuracy=MAX_DIFF)
        self.assertTrue(approx_equal)


class TestUpdateNwsScaleMatrix(TestSetUp):
    def testFaithful(self):
        """ Test `_update_nws_scale_matrix` with some
        data from Old Faithful """
        nwssm = loadmat(join(TEST_DATA_LOC, 'posterior_nws_scale_matrix.mat'),
                        squeeze_me=True)

        nws_scale_matrix_test = _Posterior._update_nws_scale_matrix(
            nwssm['num_obs'], nwssm['num_comp'], nwssm['smm_mean'],
            nwssm['prior_nws_mean'], nwssm['latent_scaled_resp'],
            nwssm['smm_covar'], nwssm['prior_nws_scale'],
            nwssm['posterior_nws_scale'], nwssm['prior_nws_scale_matrix'])

        self.assertEqual(nws_scale_matrix_test.shape,
                         (nwssm['num_comp'],
                          nwssm['num_dim'],
                          nwssm['num_dim']))
        approx_equal = arrays_almost_equal(nws_scale_matrix_test,
                                           nwssm['posterior_nws_scale_matrix'],
                                           accuracy=MAX_DIFF)
        self.assertTrue(approx_equal)


class TestUpdateSmmDof(TestSetUp):
    def testFaithful(self):
        """ Test `_update_smm_dof` with some data from Old Faithful """
        sd = loadmat(join(TEST_DATA_LOC, 'smm_dof.mat'),
                     squeeze_me=True)

        smm_dof_test = _Posterior._update_smm_dof(sd['smm_dof_old'],
                                                  sd['num_obs'],
                                                  sd['num_comp'],
                                                  sd['smm_mixweights'],
                                                  sd['latent_resp'],
                                                  sd['latent_scale'],
                                                  sd['latent_log_scale'])

        self.assertEqual(smm_dof_test.size, sd['num_comp'])
        approx_equal = arrays_almost_equal(smm_dof_test,
                                           sd['smm_dof'],
                                           accuracy=MAX_DIFF)
        self.assertTrue(approx_equal)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUpdateDirichlet)
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateNwsScale))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateNwsMean))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateNwsDof))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateNwsScaleMatrix))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateSmmDof))
    unittest.TextTestRunner(verbosity=2).run(suite)
