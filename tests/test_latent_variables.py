import unittest
from os.path import join

import numpy as np
import numpy.testing as npt

from scipy.io.matlab.mio import loadmat

from flowvb.core.latent_variables import LatentVariables

TEST_ACCURACY = 3
MAX_DIFF = pow(10, -TEST_ACCURACY)
TEST_DATA_LOC = join('data', 'old_faithful')
PATH_TO_DATA = join('data', 'old_faithful', 'faithful.mat')

#
#TestUpdateLatentResp = makeTestFaithful('latent_resp.mat',
#        LatentVariables._update_latent_resp,
#        ('smm_dof', 'posterior_nws_scale', 'posterior_nws_dof',
#         'log_smm_mixweight', 'log_det_precision', 'scatter'),
#        'latent_resp', load_data=True)
#
#TestUpdateLatentScale = makeTestFaithful('latent_scale.mat',
#        LatentVariables._update_latent_scale,
#        ('gamma_param_alpha', 'gamma_param_beta'),
#        'latent_scale')
#
#TestUpdateLatentLogScale = makeTestFaithful('latent_log_scale.mat',
#        LatentVariables._update_latent_log_scale,
#        ('gamma_param_alpha', 'gamma_param_beta'),
#        'latent_log_scale')
#
#TestUpdateLatentScaledResp = makeTestFaithful('latent_scaled_resp.mat',
#    LatentVariables._update_latent_scaled_resp,
#    ('num_obs', 'latent_resp', 'latent_scale'),
#    'latent_scaled_resp')
#
#TestUpdateLogSmmMixweight = makeTestFaithful('log_smm_mixweight.mat',
#        LatentVariables._update_log_smm_mixweight,
#        ('posterior_dirichlet',),
#        'log_smm_mixweight')
#
#TestUpdateLogDetPrecision = makeTestFaithful('log_det_precision.mat',
#        LatentVariables._update_log_det_precision,
#        ('num_features', 'num_comp', 'posterior_nws_dof',
#         'posterior_nws_scale_matrix'),
#        'log_det_precision')
#
#TestUpdateGammaParamAlpha = makeTestFaithful('gamma_param_alpha.mat',
#        LatentVariables._update_gamma_param_alpha,
#        ('num_features', 'smm_dof'),
#        'gamma_param_alpha')
#
#TestUpdateGammaParamBeta = makeTestFaithful('gamma_param_beta.mat',
#        LatentVariables._update_gamma_param_beta,
#        ('num_features', 'smm_dof', 'posterior_nws_dof',
#         'posterior_nws_scale', 'scatter'),
#        'gamma_param_beta')
#
#TestGetScatter = makeTestFaithful('scatter.mat',
#        LatentVariables._get_scatter,
#        ('posterior_nws_scale_matrix_inc', 'posterior_nws_mean'),
#        'scatter', load_data=True)

class Test(unittest.TestCase):
    def setUp(self):
        self.lv = LatentVariables()
        
        mat = loadmat(PATH_TO_DATA, squeeze_me=True)
        self.data = mat['data']
    
    def test_get_scatter(self):
        x = loadmat(join(TEST_DATA_LOC, 'scatter.mat'))
        
        parameters = {}
        parameters['nws_mean'] = x['posterior_nws_mean'].swapaxes(0, 1)
        parameters['nws_scale_matrix_inv'] = np.rollaxis(x['posterior_nws_scale_matrix_inc'], 0, 3)
        parameters['nws_scale_matrix_inv'] = np.array(parameters['nws_scale_matrix_inv'].tolist())
        
        ncomp = parameters['nws_scale_matrix_inv'].shape[2]
        parameters['nws_scale_matrix'] = np.zeros(())
        for i in range(ncomp):
            

        
        test_scatter = self.lv._get_scatter(self.data, parameters)
        real_scatter = x['scatter'].swapaxes(0, 1)
        
        npt.assert_array_almost_equal(test_scatter, real_scatter, 5)
        
    def test_get_alpha(self):
        x = loadmat(join(TEST_DATA_LOC, 'gamma_param_alpha.mat'))
        
        parameters = {}
        parameters['smm_dof'] = x['smm_dof'].reshape(1, 6)
        
        test = self.lv._get_alpha(self.data, parameters)
        real = x['gamma_param_alpha']
        
        npt.assert_array_almost_equal(test, real, 5)
   
    def test_get_beta(self):
        x = loadmat(join(TEST_DATA_LOC, 'gamma_param_beta.mat'))
        
        parameters = {}
        parameters['smm_dof'] = x['smm_dof'].reshape(1, 6)
        
        parameters['nws_dof'] = x['posterior_nws_dof'].reshape(1, 6)
        parameters['nws_scale'] = x['posterior_nws_scale'].reshape(1, 6)
        
        test = self.lv._get_beta(self.data, parameters)
        real = x['gamma_param_beta']
        
        npt.assert_array_almost_equal(test, real, 5)
             
        
        
