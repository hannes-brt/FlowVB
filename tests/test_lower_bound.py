import unittest
from os.path import join
from flowvb.core._lower_bound import _LowerBound
from tests.test_old_faithful import makeTestFaithful

TEST_ACCURACY = 3
MAX_DIFF = pow(10, -TEST_ACCURACY)
TEST_DATA_LOC = join('tests', 'data', 'old_faithful')

TestLogDirichletNormalizationPrior = makeTestFaithful(
    'log_dirichlet_normalization_prior.mat',
    _LowerBound._log_dirichlet_normalization_prior,
    ('num_comp', 'prior_dirichlet'),
    'log_dirichlet_normalization_prior')

TestLogDirichletNormalization = makeTestFaithful(
    'log_dirichlet_normalization.mat',
    _LowerBound._log_dirichlet_normalization,
    ('num_obs', 'num_comp', 'prior_dirichlet'),
    'log_dirichlet_normalization')

TestLogDirichletConst = makeTestFaithful('log_dirichlet_const.mat',
    _LowerBound._log_dirichlet_const,
    ('prior_dirichlet',), 'log_dirichlet_const')

TestLogWishartConst = makeTestFaithful('log_wishart_const.mat',
    _LowerBound._log_wishart_const,
    ('num_features', 'nws_dof', 'nws_scale_matrix'),
    'log_wishart_const', max_diff=1)

TestExpectLogPx = makeTestFaithful('expect_log_px.mat',
    _LowerBound._expect_log_px,
    ('num_obs', 'num_features', 'num_comp', 'latent_resp',
    'latent_scale', 'latent_log_scale', 'latent_scaled_resp',
    'posterior_nws_mean', 'posterior_nws_scale_matrix_inv',
    'posterior_nws_dof', 'posterior_nws_scale',
    'smm_mixweights', 'log_det_precision'),
    'expect_log_px', load_data=True)

TestExpectLogPu = makeTestFaithful('expect_log_pu.mat',
    _LowerBound._expect_log_pu,
    ('num_obs', 'num_comp', 'smm_mixweights', 'smm_dof',
     'latent_resp', 'latent_log_scale', 'latent_scaled_resp',
     'expect_log_pu'),
    'expect_log_pu')

TestExpectLogPz = makeTestFaithful('expect_log_pz.mat',
    _LowerBound._expect_log_pz,
    ('num_comp', 'latent_resp', 'log_smm_mixweight'),
    'expect_log_pz')

TestExpectLogPTheta = makeTestFaithful('expect_log_ptheta.mat',
    _LowerBound._expect_log_ptheta,
    ('num_comp', 'num_features', 'prior_nws_mean', 'prior_dirichlet',
     'prior_nws_dof', 'prior_nws_scale', 'prior_nws_scale_matrix',
     'posterior_nws_mean', 'posterior_nws_dof', 'posterior_nws_scale',
     'posterior_nws_scale_matrix_inv', 'log_smm_mixweight',
     'log_det_precision', 'log_wishart_const_init',
     'log_dirichlet_normalization_prior'),
    'expect_log_ptheta')

TestExpectLogQu = makeTestFaithful('expect_log_qu.mat',
    _LowerBound._expect_log_qu,
    ('num_obs', 'num_comp', 'gamma_param_alpha', 'gamma_param_beta',
     'latent_resp', 'smm_mixweights'),
    'expect_log_qu')

TestExpectLogQz = makeTestFaithful('expect_log_qz.mat',
    _LowerBound._expect_log_qz,
    ('latent_resp',), 'expect_log_qz')

TestExpectLogQTheta = makeTestFaithful('expect_log_qtheta.mat',
    _LowerBound._expect_log_qtheta,
    ('num_comp', 'num_features', 'log_wishart_const',
     'log_dirichlet_normalization', 'posterior_dirichlet',
     'posterior_nws_scale', 'posterior_nws_dof',
     'posterior_nws_scale_matrix', 'log_smm_mixweight',
     'log_det_precision'),
    'expect_log_qtheta')

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestLogDirichletNormalizationPrior)
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(
            TestLogDirichletNormalization))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestLogDirichletConst))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestLogWishartConst))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestExpectLogPx))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestExpectLogPu))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestExpectLogPz))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestExpectLogPTheta))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestExpectLogQu))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestExpectLogQz))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestExpectLogQTheta))
    unittest.TextTestRunner(verbosity=2).run(suite)
