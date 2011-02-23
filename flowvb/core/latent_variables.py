import numpy as np
from scipy.special import gammaln, psi

from flowvb.normalize import normalize_logspace
from flowvb.utils import logdet, multipsi


class LatentVariables(object):
    '''
    Class to compute and store the latent variables.
    '''
    def update(self, data, parameters):
        '''
        Update latent variables.
        '''
        scalar_shape = (1, 6)
        
        smm_dof = parameters['smm_dof']
        smm_dof = smm_dof.reshape(scalar_shape)
        
        nws_dof = parameters['nws_dof']
        nws_scale = parameters['nws_scale']
        nws_scale_matrix = parameters['nws_scale_matrix']
        
        dirichlet = parameters['dirichlet']
                
        ndim = data.shape[0]
        
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
    
    def _get_scatter(self, data, parameters):
        '''
        Compute scatter matrix.
        '''        
        nws_mean = parameters['nws_mean']
        nws_scale_matrix = parameters['nws_scale_matrix']       
        
        ndim = nws_mean.shape[0]
        ncomp = nws_mean.shape[1]
        
        scatter = np.zeros((ndim, ndim, ncomp))
        
        for i in range(ndim):        
            dist = (data - nws_mean[:, i])
            
            nws_scale_matrix_inv = np.linalg.inv(nws_scale_matrix[:, :, i])
            
            temp = np.dot(dist.T, nws_scale_matrix_inv) 
            scatter[:, :, i] = np.dot(temp, dist)
        
        return scatter
