import random as rd
import numpy as np
import unittest
from flowvb.core._posterior import _Posterior
from flowvb.utils import normalize

TEST_ACCURACY = 3


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
        from data.old_faithful.setup_test_data.posterior import dirichlet as dl

        posterior_dirichlet_test = _Posterior._update_dirichlet(
            dl.num_obs, dl.smm_mixweights, dl.prior_dirichlet)

        [self.assertAlmostEqual(dl.posterior_dirichlet[k],
                                posterior_dirichlet_test[k],
                                TEST_ACCURACY)
         for k in range(dl.num_comp)]


class TestUpdateNwsScale(TestSetUp):
    def testFaithful(self):
        """Test `_update_nws_scale` with some data from Old Faithful """
        from data.old_faithful.setup_test_data.posterior \
             import nws_scale as nwss

        nws_scale_test = _Posterior._update_nws_scale(nwss.num_obs,
                                                      nwss.scaled_resp,
                                                      nwss.prior_nws_scale)

        [self.assertAlmostEqual(nwss.nws_scale[k],
                                nws_scale_test[k],
                                TEST_ACCURACY)
         for k in range(nwss.num_comp)]


class TestUpdateNwsMean(TestSetUp):
    def testFaithful(self):
        """ Test `_update_nws_mean` with some data from Old Faithful """
        from data.old_faithful.setup_test_data.posterior \
             import nws_mean as nwsm

        nws_mean_test = _Posterior._update_nws_mean(nwsm.num_obs,
                                                    nwsm.num_comp,
                                                    nwsm.scaled_resp,
                                                    nwsm.smm_mean,
                                                    nwsm.prior_nws_scale,
                                                    nwsm.nws_scale,
                                                    nwsm.prior_nws_mean)

        self.assertEqual(nws_mean_test.shape,
                         (nwsm.num_comp, nwsm.num_dim))

        [[self.assertAlmostEqual(nwsm.nws_mean[k, d],
                                nws_mean_test[k, d],
                                TEST_ACCURACY)
         for d in range(nwsm.num_dim)]
         for k in range(nwsm.num_comp)]

 
class TestUpdateNwsDof(TestSetUp):
    def testFaithful(self):
        """ Test `_update_nws_dof` with some data from Old Faithful """
        from data.old_faithful.setup_test_data.posterior \
             import nws_dof as nwsd

        nws_dof_test = _Posterior._update_nws_dof(nwsd.num_obs,
                                                  nwsd.smm_mixweights,
                                                  nwsd.prior_nws_dof)

        self.assertEqual(nws_dof_test.size, nwsd.num_comp)
        [self.assertAlmostEqual(nwsd.nws_dof[k], nws_dof_test[k],
                                TEST_ACCURACY)
         for k in range(nwsd.num_comp)]


class TestUpdateNwsScaleMatrix(TestSetUp):
    def testFaithful(self):
        """ Test `_update_nws_scale_matrix` with some
        data from Old Faithful """
        from data.old_faithful.setup_test_data.posterior \
             import nws_scale_matrix as nwssm

        nws_scale_matrix_test = _Posterior._update_nws_scale_matrix(
            nwssm.num_obs, nwssm.num_comp, nwssm.smm_mean,
            nwssm.prior_nws_mean, nwssm.scaled_resp,
            nwssm.smm_covar, nwssm.prior_nws_scale,
            nwssm.nws_scale, nwssm.prior_nws_scale_matrix)

        self.assertEqual(nws_scale_matrix_test.shape,
                         (nwssm.num_comp, nwssm.num_dim, nwssm.num_dim))
        [[[self.assertAlmostEqual(nwssm.nws_scale_matrix[k, i, j],
                                nws_scale_matrix_test[k, i, j],
                                TEST_ACCURACY)
         for j in range(nwssm.num_dim)]
         for i in range(nwssm.num_dim)]
         for k in range(nwssm.num_comp)]


class TestUpdateSmmDof(TestSetUp):
    def testFaithful(self):
        """ Test `_update_smm_dof` with some data from Old Faithful """
        from data.old_faithful.setup_test_data.posterior \
             import smm_dof as sd

        smm_dof_test = _Posterior._update_smm_dof(sd.smm_dof_old,
                                                  sd.num_obs,
                                                  sd.num_comp,
                                                  sd.smm_mixweights,
                                                  sd.latent_resp,
                                                  sd.latent_scale,
                                                  sd.latent_log_scale)

        self.assertEqual(smm_dof_test.size, sd.num_comp)
        [self.assertAlmostEqual(sd.smm_dof[k],
                                smm_dof_test[k],
                                TEST_ACCURACY)
         for k in range(sd.num_comp)]

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
