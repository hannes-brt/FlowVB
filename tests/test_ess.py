import unittest
from flowvb.core._ess import ExpectedSufficientStatistics
from tests.test_old_faithful import makeTestFaithful

TestUpdateSmmMean = makeTestFaithful('smm_mean.mat',
    ExpectedSufficientStatistics._update_smm_mean,
    ('num_obs', 'num_comp', 'latent_scaled_resp',
     'latent_resp', 'latent_scale'),
    'smm_mean', load_data=True)

TestUpdateSmmCovar = makeTestFaithful('smm_covar.mat',
    ExpectedSufficientStatistics._update_smm_covar,
    ('num_obs', 'num_features', 'num_comp',
     'latent_resp', 'latent_scale', 'latent_scaled_resp', 'smm_mean'),
    'smm_covar', load_data=True)

TestUpdateSmmMixweights = makeTestFaithful('smm_mixweights.mat',
    ExpectedSufficientStatistics._update_smm_mixweights,
    ('num_obs', 'latent_resp'),
    'smm_mixweights')

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestUpdateSmmMean)
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateSmmCovar))
    suite.addTest(
        unittest.TestLoader().loadTestsFromTestCase(TestUpdateSmmMixweights))
    unittest.TextTestRunner(verbosity=2).run(suite)
