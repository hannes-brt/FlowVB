'''
Created on 2011-02-17

@author: Andrew Roth
'''
from flowvb.core._ess import _ESS
from flowvb.core._latent_variables import _LatentVariables
from flowvb.core._lower_bound import _LowerBound
from flowvb.core._posterior import _Posterior

class Model(object):
    def __init__(self, priors):
        self._priors = priors
        
        self._ess = _ESS()
        self._latent_variables = _LatentVariables()
        self._lower_bound = _LowerBound(priors)
        self._posterior = _Posterior(priors)
        
    def vbe_step(self, data):
        self.latent_variables = self._latent_variables.update_parameters(data, self.parameters)
    
    def vbm_step(self, data):
        self.ess = self._ess.update_parameters(data, self.latent_variables)
        self.parameters = self._posterior.update_parameters(data, self.ess)
