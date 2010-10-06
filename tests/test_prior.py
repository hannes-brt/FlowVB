import unittest
import numpy as np
import os.path
from flowvb.core._prior import _Prior


class TestPrior(unittest.TestCase):
    def testTypes(self):
        """ Tests traits of `_Prior` class for correctness """

        num_comp = 6
        data = np.genfromtxt(os.path.join('tests', 'data', 'old_faithful',
                                          'faithful.txt'))
        num_obs = data.shape[0]
        num_features = data.shape[1]
        prior_dirichlet = 1e-3
        nws_mean = 0
        nws_scale = 1e-3
        nws_dof = 20
        nws_scale_matrix = 1 / 200.

        PriorObj = _Prior(data, num_comp, prior_dirichlet, nws_mean, nws_scale,
                          nws_dof, nws_scale_matrix)

        nws_scale_matrix = nws_scale_matrix * np.mat(np.eye(num_features))

        self.assertEqual(PriorObj.num_obs, num_obs)
        self.assertEqual(PriorObj.num_comp, num_comp)
        self.assertEqual(PriorObj.num_features, num_features)
        self.assertEqual(PriorObj.dirichlet, prior_dirichlet)
        self.assertEqual(PriorObj.nws_mean, nws_mean)
        self.assertEqual(PriorObj.nws_scale, nws_scale)
        self.assertEqual(PriorObj.nws_dof, nws_dof)
        [[self.assertEqual(PriorObj.nws_scale_matrix[i, j],
                           nws_scale_matrix[i, j])
         for j in range(num_features)]
         for i in range(num_features)]


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPrior)
    unittest.TextTestRunner(verbosity=2).run(suite)
