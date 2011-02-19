'''
Created on 2011-02-17

@author: Andrew Roth
'''
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
        
    def vbe_step(self, data):
        self.latent_variables = self._latent_variables.update(data, self.parameters)
    
    def vbm_step(self, data):
        self.ess = self._ess.update(data, self.latent_variables)
        self.parameters = self._posterior.update(data, self.ess)
