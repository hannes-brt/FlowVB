import unittest
from os.path import join
from flowvb.core._posterior import _Posterior
from tests.test_old_faithful import makeTestFaithful

TEST_ACCURACY = 3
MAX_DIFF = pow(10, -TEST_ACCURACY)
TEST_DATA_LOC = join('tests', 'data', 'old_faithful')

TestUpdateDirichlet = makeTestFaithful('posterior_dirichlet.mat',
            _Posterior._update_dirichlet,
            ('num_obs', 'smm_mixweights', 'prior_dirichlet'),
            'posterior_dirichlet')


TestUpdateNwsScale = makeTestFaithful('posterior_nws_scale.mat',
        _Posterior._update_nws_scale,
        ('num_obs', 'latent_scaled_resp', 'prior_nws_scale'),
        'posterior_nws_scale')


TestUpdateNwsMean = makeTestFaithful('posterior_nws_mean.mat',
    _Posterior._update_nws_mean,
    ('num_obs', 'num_comp', 'latent_scaled_resp', 'smm_mean',
     'prior_nws_scale', 'posterior_nws_scale', 'prior_nws_mean'),
    'posterior_nws_mean')

TestUpdateNwsDof = makeTestFaithful('posterior_nws_dof.mat',
        _Posterior._update_nws_dof,
        ('num_obs', 'smm_mixweights', 'prior_nws_dof'),
        'posterior_nws_dof')

TestUpdateNwsScaleMatrix = makeTestFaithful('posterior_nws_scale_matrix.mat',
        _Posterior._update_nws_scale_matrix,
        ('num_obs', 'num_comp', 'smm_mean',
         'prior_nws_mean', 'latent_scaled_resp',
         'smm_covar', 'prior_nws_scale',
         'posterior_nws_scale', 'prior_nws_scale_matrix'),
        'posterior_nws_scale_matrix')

TestUpdateSmmDof = makeTestFaithful('smm_dof.mat',
        _Posterior._update_smm_dof,
        ('smm_dof_old', 'num_obs', 'num_comp', 'smm_mixweights',
         'latent_resp',  'latent_scale', 'latent_log_scale'),
        'smm_dof')

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
