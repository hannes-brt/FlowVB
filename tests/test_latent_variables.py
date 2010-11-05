import unittest
from os.path import join
from flowvb.core._latent_variables import _LatentVariables
from test_old_faithful import makeTestFaithful

TEST_ACCURACY = 3
MAX_DIFF = pow(10, -TEST_ACCURACY)
TEST_DATA_LOC = join('tests', 'data', 'old_faithful')


TestUpdateLatentResp = makeTestFaithful('latent_resp.mat',
        _LatentVariables._update_latent_resp,
        ('smm_dof', 'posterior_nws_scale', 'log_smm_mixweight',
         'log_det_precision', 'scatter'),
        'latent_resp', load_data=True)

TestUpdateLatentScale = makeTestFaithful('latent_scale.mat',
        _LatentVariables._update_latent_scale,
        ('gamma_param_alpha', 'gamma_param_beta'),
        'latent_scale')

TestUpdateLatentLogScale = makeTestFaithful('latent_log_scale.mat',
        _LatentVariables._update_latent_log_scale,
        ('gamma_param_alpha', 'gamma_param_beta'),
        'latent_log_scale')

TestUpdateLatentScaledResp = makeTestFaithful('latent_scaled_resp.mat',
    _LatentVariables._update_latent_scaled_resp,
    ('num_obs', 'latent_resp', 'latent_scale'),
    'latent_scaled_resp')

TestUpdateLogSmmMixweight = makeTestFaithful('log_smm_mixweight.mat',
        _LatentVariables._update_log_smm_mixweight,
        ('posterior_dirichlet',),
        'log_smm_mixweight')

TestUpdateLogDetPrecision = makeTestFaithful('log_det_precision.mat',
        _LatentVariables._update_log_det_precision,
        ('num_features', 'num_comp', 'posterior_nws_dof',
         'posterior_nws_scale_matrix'),
        ('log_det_precision'), max_diff=1e-1)

TestUpdateGammaParamAlpha = makeTestFaithful('gamma_param_alpha.mat',
        _LatentVariables._update_gamma_param_alpha,
        ('num_features', 'smm_dof'),
        'gamma_param_alpha')

TestUpdateGammaParamBeta = makeTestFaithful('gamma_param_beta.mat',
        _LatentVariables._update_gamma_param_beta,
        ('posterior_nws_dof', 'posterior_nws_scale', 'scatter'),
        'gamma_param_beta', )


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUpdateLatentResp)
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateLatentScale))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateLatentLogScale))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(
            TestUpdateLatentScaledResp))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateLogSmmMixweight))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateLogDetPrecision))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateGammaParamAlpha))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateGammaParamBeta))
    unittest.TextTestRunner(verbosity=2).run(suite)
