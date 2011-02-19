from math import log

import numpy as np

from scipy.special import psi, erf, polygamma

from scipy.optimize import newton

class Posterior(object):
    '''
    Class to compute posterior parameters of the model.
    '''
    def __init__(self, priors, use_approx=False):
        '''
        Initialize posterior parameters.
        '''
        self.priors = priors
        self.use_approx = use_approx

    def update_parameters(self, ess):
        '''
        Update posterior parameters.
        '''        
        prior_dirichlet = self.priors['dirichlet']
        prior_nws_scale = self.priors['nws_scale']
        
        prior_nws_dof = self.priors['prior_nws_dof']
        prior_nws_mean = self.priors['nws_mean']
        
        N = ess['N']
        smm_mix_weights = ess['smm_mix_weights']
        smm_scaled_mix_weights = ess['smm_scaled_mix_weights']
        smm_mean = ess['smm_mean']

        
        # Eqn (27)
        dirichlet = N * smm_mix_weights + prior_dirichlet
        
        # Eqn (28)
        nws_scale = N * smm_scaled_mix_weights + prior_nws_scale
        
        # Eqn (29)
        nws_mean = N * smm_scaled_mix_weights * smm_mean + \
                   prior_nws_scale * prior_nws_mean
        nws_mean = nws_mean / nws_scale

        # Eqn (30)
        nws_dof = N * smm_mix_weights + prior_nws_dof
        
        # Eqn (31)
        nws_scale_matrix = self._get_nws_scale_matrix(ess)

        if self.use_approx:
            smm_dof = self._update_smm_dof_approx(ess)
        else:
            smm_dof = self._update_smm_dof(ess)
            
        parameters = {}
        parameters['dirichlet'] = dirichlet
        parameters['nws_scale'] = nws_scale
        parameters['nws_mean'] = nws_mean
        parameters['nws_dof'] = nws_dof
        parameters['nws_scale_matrix'] = nws_scale_matrix
        parameters['smm_dof'] = smm_dof
        
        return parameters

    def _get_nws_scale_matrix(self, ess):
        '''
        Update nws_scale_matrix.
        '''
        prior_nws_mean = self.priors['nws_mean']
        prior_nws_scale = self.priors['nws_scale']
        prior_nws_scale_matrix = self.priors['nws_scale_matrix']
        
        N = ess['N']
        smm_mean = ess['smm_mean']
        smm_covar = ess['smm_covar']
        smm_scaled_mix_weights = ess['smm_scaled_mix_weights']
        
        shape = smm_covar.shape
        ncomp = smm_covar.shape[2]
        
        nws_scale_matrix = np.zeros(shape)
        
        for i in range(ncomp):
            smm_mean_i = smm_mean[:, i]
            smm_scaled_mix_weights_i = smm_scaled_mix_weights[:, i]
            
            # First term in sum
            temp = N * smm_scaled_mix_weights * smm_covar[:, :, i]
            
            dist = (smm_mean_i - prior_nws_mean)       
            temp = np.outer(dist, dist)
            
            numerator = N * smm_scaled_mix_weights_i * prior_nws_scale
            denominator = N * smm_scaled_mix_weights_i + prior_nws_scale
            
            # Second term in sum
            temp = temp + numerator / denominator * temp
            
            # Third term in sum
            temp = temp + prior_nws_scale_matrix
            
            nws_scale_matrix[:, :, i] = temp        

        return nws_scale_matrix

    def _update_smm_dof(self, priors, ess):
        '''
         Update smm_dof using gradient descent.
        '''
        x0 = 4
        
        smm_dof_root = ess['smm_dof_root']
        
        ncomp = smm_dof_root.shape[0] 

        smm_dof = np.zeros(ncomp)

        for k in range(ncomp):
            objective = lambda dof: log(dof / 2) + 1 - psi(dof / 2) + smm_dof_root[k]
            objective_deriv = lambda dof: 1 / dof - polygamma(1, dof / 2)

            smm_dof_new = newton(objective, x0, objective_deriv)

            smm_dof = smm_dof[k] = smm_dof_new

        return smm_dof

#    @staticmethod
#    def _update_smm_dof_approx(self, priors, ess):
#        ''' 
#        Update smm_dof (Eq 36 in Arch2007) using the
#        approximation of Shoham (2002)
#        '''
#        smm_dof = smm_dof_old
#
#        y = (-np.sum(latent_resp * 
#                    (latent_log_scale - latent_scale), 0) / 
#             (num_obs * smm_mixweights))
#        smm_dof_new = (2 / (y + np.log(y) - 1) + 0.0416 * 
#                       (1 + erf(0.6594 * np.log(2.1971 / 
#                                                (y + np.log(y) - 1)))))
#
#        smm_dof[~np.isnan(smm_dof_new)] = smm_dof_new[~np.isnan(smm_dof_new)]
#
#        return smm_dof
