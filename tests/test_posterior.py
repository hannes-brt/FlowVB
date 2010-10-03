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
        """ Test return values with random numbers """
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
        """ Test with some data from Old Faithful """
        from data.old_faithful.setup_test_data.posterior import dirichlet as dl

        posterior_dirichlet_test = _Posterior._update_dirichlet(
            dl.num_obs, dl.smm_mixweights, dl.prior_dirichlet)

        [self.assertAlmostEqual(dl.posterior_dirichlet[k],
                                posterior_dirichlet_test[k],
                                TEST_ACCURACY)
         for k in range(dl.num_comp)]

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUpdateDirichlet)
    unittest.TextTestRunner(verbosity=2).run(suite)
