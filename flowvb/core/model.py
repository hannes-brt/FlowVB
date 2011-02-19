'''
Created on 2011-02-17

@author: Andrew Roth
'''
import numpy as np

from flowvb.core.ess import ExpectedSufficientStatistics
from flowvb.core.latent_variables import LatentVariables
from flowvb.core.lower_bound import LowerBound
from flowvb.core.posterior import Posterior

class Model(object):
    def __init__(self, priors):
        self._priors = priors
        
        self._ess = ExpectedSufficientStatistics()
        self._latent_variables = LatentVariables()
        self._lower_bound = LowerBound(priors)
        self._posterior = Posterior(priors)
        
    def e_step(self, data):
        self.latent_variables = self._latent_variables.update(data, self.parameters)
    
    def m_step(self, data):
        self.ess = self._ess.update(data, self.latent_variables)
        self.parameters = self._posterior.update(data, self.ess)
        
    def remove_empty_clusters(self, cluster_indices):
        '''
        Remove empty clusters from the model. e_step should be called after runing this.
        '''
        
        # Scalar parameters
        self.parameters['dirichlet'] = np.delete(self.parameters['dirichlet'], cluster_indices, 0)
        self.parameters['nws_scale'] = np.delete(self.parameters['nws_scale'], cluster_indices, 0)
        self.parameters['nws_dof'] = np.delete(self.parameters['nws_dof'], cluster_indices, 0)
        self.parameters['smm_dof'] = np.delete(self.parameters['smm_dof'], cluster_indices, 0)
        
        # Vector parameters
        self.parameters['nws_mean'] = np.delete(self.parameters['nws_mean'], cluster_indices, 1)
        
        # Matrix parameters
        self.parameters['nws_scale_matrix'] = np.delete(self.parameters['nws_scale_matrix'], cluster_indices, 2)

        
