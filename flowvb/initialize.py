import numpy as np
from numpy.random import multinomial, uniform

from scipy.spatial.distance import mahalanobis
from scipy.cluster.vq import kmeans2

from flowvb.normalize import normalize_logspace
from flowvb.utils import element_weights, classify_by_distance

def init_d2_weighting(data, num_comp):

    num_obs = data.shape[0]

    cov_inv = np.linalg.inv(np.cov(data, rowvar=0))

    select_prob = np.ones(num_obs) / num_obs
    shortest_dist = np.inf * np.ones(num_obs)
    centroid = np.ones(num_comp)

    for k in range(num_comp):
        # Select a random data point as centroid
        centroid[k] = np.nonzero(multinomial(1, select_prob))[0]

        # Recompute distances
        for i, d in enumerate(shortest_dist):
            d_new = mahalanobis(data[centroid[k], :], data[i, :], cov_inv)
            if d_new < d: shortest_dist[i] = d_new

        select_prob = normalize_logspace(
            pow(shortest_dist.reshape(1, len(shortest_dist)), 2, 1))
        select_prob = select_prob.flatten()

    return centroid

#=======================================================================================================================
# Classes
#=======================================================================================================================
class Initialiser(object):
    def initialise_parameters(self, data, num_comp):
        raise NotImplemented
    
    def _get_covar(self, data, labels, *args, **kargs):
        '''
        Compute the covariance in all clusters
        '''
        elements = range(max(labels) + 1)

        def covar(m):
            # Make sure, a dxd-matrix is returned, even when there are
            # only zero or one observations
            if len(m.shape) > 1:
                d, n = m.shape
            elif len(m.shape) == 1:
                n = 1
                d = m.shape[2]
            if n > 1:
                return np.cov(m, *args, **kargs)
            else:
                return np.zeros([d, d])

        return np.array([covar(data[labels == l, :].T)
                         for l in elements])
    
class RandomInitialiser(Initialiser):
    def initialise_parameters(self, data, num_comp):
        '''
        Initialize randomly
        '''
        D = data.shape[1]
        
        data_lims = np.array([[data[:, d].min(), data[:, d].max()]
                              for d in range(D)])

        init_mean = np.array([uniform(*data_lims[d, :], size=num_comp)
                              for d in range(D)]).T

        covar_init = np.repeat([np.diag([1] * D)], num_comp, 0)

        labels = classify_by_distance(data,
                                      init_mean,
                                      covar_init)
        labels = labels.flatten()
        
        init_covar = self._get_covar(data, labels)
        
        init_mixweights = element_weights(labels)
        
        init_params = {}
        init_params['mean'] = init_mean
        init_params['covar'] = init_covar
        init_params['mixweights'] = init_mixweights
            
        return init_params
    
class D2Initialiser(Initialiser):
    def initialise_parameters(self, data, num_comp):
        """Initialize using D2-weighting
        """

        centroids_idx = init_d2_weighting(data, num_comp)

        init_mean = np.array([data[k, :] for k in centroids_idx])
                
        init_covar = np.cov(data, rowvar=0)
        
        init_covar = np.repeat(np.array([init_covar]), num_comp, 0)

        labels = classify_by_distance(data,
                                      init_mean,
                                      init_covar)
        
        labels = labels.flatten()

        init_covar = self._get_covar(data, labels)
        
        init_mixweights = element_weights(labels)
        
        init_params = {}
        init_params['mean'] = init_mean
        init_params['covar'] = init_covar
        init_params['mixweights'] = init_mixweights
            
        return init_params
    
class KMeansInitialiser(Initialiser):
    def initialise_parameters(self, data, num_comp):
        '''
        Initialize using k-means
        '''
        (init_mean, labels) = kmeans2(data, num_comp)
        
        init_covar = self._get_covar(data, labels)
        
        init_mixweights = element_weights(labels)
        
        init_params = {}
        init_params['mean'] = init_mean
        init_params['covar'] = init_covar
        init_params['mixweights'] = init_mixweights
            
        return init_params
    
class UserParameterInitialiser(Initialiser):
    def initialise_parameters(self, data, init_mean, init_covar, init_mixweights):
        '''
        If starting solution supplied initialise from it.
        '''                
        if init_mixweights is None:
            labels = classify_by_distance(data,
                                          init_mean,
                                          init_covar)
            
            init_mixweights = element_weights(labels)
        
        if init_covar is None:
            init_covar = self._get_covar(data, labels)
        
        init_params = {}
        init_params['mean'] = init_mean
        init_params['covar'] = init_covar
        init_params['mixweights'] = init_mixweights
            
        return init_params
