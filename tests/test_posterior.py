import random as rd
import numpy as np
import unittest
from flowvb.core._posterior import _Posterior
from flowvb.utils import normalize

class TestUpdatePosteriorDirichlet(unittest.TestCase):

    def setUp(self):
        rd.seed(1)
        self.repeats = 100
        self.max_num_obs = 1e6
        self.max_num_comp = 20

    def testReturnValue(self):
        for i in range(self.repeats):
            num_obs = rd.randint(1, self.max_num_obs)
            num_comp = rd.randint(1, self.max_num_comp)
            e_mixweights = normalize([rd.uniform(0, 1) for i in
                                      range(num_comp)])
            prior_dirichlet = [rd.uniform(0, 1) for i in range(num_comp)]

            posterior_dirichlet = _Posterior._update_posterior_dirichlet(
                num_obs, e_mixweights, prior_dirichlet)

            self.assertEqual(len(posterior_dirichlet), num_comp)
            self.assertEqual(type(posterior_dirichlet), type(np.array(1)))
            
if __name__ == '__main__':
    unittest.main()
