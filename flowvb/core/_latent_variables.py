import numpy as np
from scipy.special import gammaln, psi

from flowvb.normalize import normalize_logspace
from flowvb.utils import logdet, multipsi
import scipy.spatial.distance


class LatentVariables(object):
    '''
    Class to compute and store the latent variables.
    '''
    def update(self, data, parameters):
        '''
        Update latent variables.
        '''
        self.ndim = data.shape[0]
        
        smm_dof = parameters['smm_dof']
        
        nws_dof = parameters['nws_dof']
        nws_scale = parameters['nws_scale']
        nws_scale_matrix = parameters['nws_scale_matrix']
        
        dirichlet = parameters['dirichlet']
                
        
        
        scatter = self._get_scatter(data, parameters)
        
        # Eqn (24)
        alpha = (smm_dof + self.ndims) / 2
        
        # Eqn (25)        
        beta = (nws_dof / 2) * scatter + \
               ndim / (2 * nws_scale) + \
               smm_dof / 2
                    
        expected_log_pi = psi(dirichlet) - psi(dirichlet.sum())
                
        expected_log_det_precision = multipsi(nws_dof, ndim) + \
                                     ndim * np.log(2) - \
                                     logdet(nws_scale_matrix)
        # Responsibilities eqn (19)
        resp = gammaln(self.alpha) + \
                      (smm_dof / 2) * np.log(smm_dof) - \
                      gammaln(smm_dof) + \
                      expected_log_pi + \
                      expected_log_det_precision - \
                      alpha * np.log(beta)
                      
        resp = np.exp(resp)        
        resp = normalize_logspace(resp)
        
        # Scale time responsibilities
        scaled_resp = np.log(resp) + np.log(alpha) - np.log(beta)
        scaled_resp = np.exp(scaled_resp)
        
        # log(scale) times responsibilities
        log_scaled_resp = np.log(resp) + np.log(psi(alpha)) - np.log(np.log(beta))
        log_scaled_resp = np.exp(log_scaled_resp)
        
        # Only return variable required to compute ESS.           
        latent_variables = {}
        latent_variables['resp'] = resp
        latent_variables['scaled_resp'] = scaled_resp
        latent_variables['log_scaled_resp'] = log_scaled_resp      
        
        return latent_variables
    
    def _get_alpha(self, data, parameters):
        ndim = data.shape[1]
        
        smm_dof = parameters['smm_dof']        
        
        alpha = (smm_dof + ndim) / 2
        
        return alpha
    
    def _get_beta(self, data, parameters):
        ndim = data.shape[1]
        
        smm_dof = parameters['smm_dof']
        
        nws_dof = parameters['nws_dof']
        nws_scale = parameters['nws_scale']
        
        scatter = self._get_scatter(data, parameters)
        
        beta = (nws_dof / 2) * scatter + \
               ndim / (2 * nws_scale) + \
               smm_dof / 2
               
        return beta
        
#    def _get_scatter(self, data, parameters):
#        """ Compute the scatter """
#        nws_mean = parameters['nws_mean']
#        nws_scale_matrix_inv = parameters['nws_scale_matrix_inv']
#        
#        num_obs = data.shape[0]
#        num_comp = nws_mean.shape[1]
#
#        def update(k):
#            data_center = data - nws_mean[:, k]
#            
#            prod = np.dot(data_center,
#                          nws_scale_matrix_inv[:, :, k])
#            return np.sum(prod * data_center, 1)
#
#        scatter = np.array([update(k) for k in range(num_comp)])
#        return scatter.swapaxes(0, 1)
    def _get_scatter(self, data, parameters):
        '''
        Compute scatter matrix.
        '''        
        nws_mean = parameters['nws_mean']
        nws_scale_matrix = parameters['nws_scale_matrix']       
        
        nobs = data.shape[0]
        ndim = data.shape[1]
        ncomp = nws_mean.shape[1]
        
        scatter = np.zeros((nobs, ncomp))
        
        for i in range(ncomp):
            u = data
            v = nws_mean[:, i]
            v = v.reshape((1, ndim))            
            
            VI = nws_scale_matrix[:, :, i]
            
            x = scipy.spatial.distance.cdist(u, v, 'mahalanobis', VI=VI)

            scatter[:, i] = x.flatten()
        
        return scatter
