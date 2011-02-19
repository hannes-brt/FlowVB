import numpy as np

class ExpectedSufficientStatistics(object):
    '''
    Class to compute expected sufficient statistics.
    '''
    
    def update_parameters(self, data, latent_variables):
        '''
        Update sufficient statistics.
        '''        
        resp = latent_variables['resp']        
        scaled_resp = latent_variables['scaled_resp']
        log_scaled_resp = latent_variables['log_scaled_resp']
        
        N = data.shape[0]
        ndim = resp.shape[0]
        ncomp = resp.shape[1]
        
        # Eqn (34)
        smm_mix_weights = (1 / N) * np.sum(resp, axis=0)
        
        # Eqn (35)
        smm_scaled_mix_weights = (1 / N) * np.sum(scaled_resp, axis=0)
        
        # Eqn (32)
        smm_mean = np.zeros((ndim, ncomp))        
        
        for i in range(ncomp):
            scaled_resp_i = scaled_resp[:, i]
            smm_mean[:, i] = (1 / N) * np.sum(scaled_resp_i[:, np.newaxis] * data, axis=0)
        
        # Eqn (33)
        smm_covar = np.zeros((ndim, ndim, ncomp))
        
        for i in range(ncomp):
            scaled_resp_i = scaled_resp[:, i]
            smm_mean_i = smm_mean[:, i]
            
            dist = (data - smm_mean_i)       
            temp = np.outer(dist, dist)
            
            smm_covar[:, :, i] = 1 / (N * smm_scaled_mix_weights[:, i]) * \
                                 np.sum(scaled_resp_i[:, np.newaxis] * temp)
    
        # The only latent variable term in  the update equation for dof eqn (36)
        smm_dof_root = 1 / (N * smm_mix_weights) * np.sum(log_scaled_resp - scaled_resp, axis=0)
    
        ess = {}
        ess['N'] = N
        ess['smm_mix_weights'] = smm_mix_weights
        ess['smm_scaled_mix_weights'] = smm_scaled_mix_weights
        ess['smm_mean'] = smm_mean
        ess['smm_covar'] = smm_covar
        ess['smm_dof_root'] = smm_dof_root
        
        return ess
