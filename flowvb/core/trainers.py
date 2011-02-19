'''
Created on 2011-02-17

@author: Andrew Roth
'''
import numpy as np
from numpy.random import multinomial, uniform

from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import mahalanobis

from flowvb.normalize import normalize_logspace
from flowvb.utils import classify_by_distance
from flowvb.core._graphics import plot_clustering

class ModelTrainer(object):
    '''
    Abstract class for learning model parameters. This class should not be called directly. Subclasses should
    implemented the _init_responsibilities method to initialise VBEM algorithm.
    '''
    def __init__(self, init_num_comp=10, max_iters=200, thresh=1e-5):
        self.num_comp = init_num_comp
        
        self.max_iters = max_iters
        
        self.tolerance = self.thresh
                                    
        self.lower_bound_values = []
        self.lower_bound_values.append(float('-inf'))
        
        self.eps = np.finfo(np.float).eps
        
    def fit_model(self, data, model, debug=False):
        '''
        Fit model to data.
        
        Input : data - data used for fitting.
                model - a Model object to train.
                
        Output : model - return fitted model.
        '''
        self.model = model
        
        self.converged = False
        
        self.iters = 0
        
        self.model.responsibilities = self._init_responsibilities(data)
        
        while not self.converged:            
            self.model.m_step(data)
            self.model.e_step(data)

            self._convergence_test()
            
            if debug:
                self._print_diagnostic_message()
                                   
            self.iters += 1
            
            if self._remove_empty_clusters():
                self.e_step(data)
                     
        return self.model
        
    def _convergence_test(self):
        '''
        Test if iteration has converged
        '''
        self.lower_bound_values.append(self.model.lower_bound)

        fval = self.lower_bound_values[-1]
        previous_fval = self.lower_bound_values[-2]

        delta_fval = abs(fval - previous_fval)
        avg_fval = (abs(fval) + abs(previous_fval) + self.eps) / 2
        
        if (delta_fval / avg_fval) < self.thresh:
            self.converged = True
        elif self.iters >= self.max_iters:
            self.converged = True
            
    
    def _remove_empty_clusters(self):
        '''
        Remove components with insufficient support from the model.
        Return true if we remove clusters false otherwise.
        '''
        empty_cluster_indices = np.where(self.model.smm_mixweights < self.remove_comp_thresh)[0]
        empty_cluster_indices = set(empty_cluster_indices)

        self.model.remove_clusters(empty_cluster_indices)
        
        if len(empty_cluster_indices) > 0:
            return True
        else:
            return False
        
    def _print_diagnostic_message(self):
        print "#" * 100
        print "# Diagnostics."
        print "#" * 100
        print "Number of iterations : ", self.iters
        print "New lower bound : ", self.lower_bound_values[-1]
        print "Old lower bound : ", self.lower_bound_values[-2]
        print "Lower bound change : ", self.lower_bound_values[-1] - self.lower_bound_values[-2]
    
        print "Parameters :"
        
        for param_name, param_value in self.model.parameters.items():
            print param_name, param_value
    
    def _init_responsibilities(self, data):
        '''
        This should be implemented by subclass.
        '''
        raise NotImplemented
    
    def _labels_to_responsibilitis(self, labels):
        nrows = labels.size
        ncols = self.num_comp
        
        shape = (nrows, ncols)
        responsibilities = np.zeros(shape)
        
        for id in range(ncols):
            index = (labels == id)

            responsibilities[index, id] = 1.        

        return responsibilities

#=======================================================================================================================
# Subclasses of ModelTrainer implementing various initialisation strategies.
#=======================================================================================================================
class KMeansTrainer(ModelTrainer):
    def _init_responsibilities(self, data):
        '''
        Initialize using k-means
        '''
        init_mean, labels = kmeans2(self.data, self.num_comps)

        return self._labels_to_responsibilitis(labels)

class D2Trainer(ModelTrainer):
    def _init_responsibilities(self, data):
        '''
        Initialise using D2-weighting
        '''
        centroids_idx = self._init_d2_weighting(data)

        init_mean = np.array([self.data[k, :] for k in centroids_idx])
        init_covar = np.cov(self.data, rowvar=0)
        init_covar = np.repeat(np.array([init_covar]), self.num_comp, 0)

        labels = classify_by_distance(self.data,
                                      init_mean,
                                      init_covar)
        labels = labels.flatten()

        return self._labels_to_responsibilitis(labels)


    def _init_d2_weighting(self, data):
        num_obs = data.shape[0]
        num_comp = self.num_comps
        
        cov = np.cov(data, rowvar=0)
        cov_inv = np.linalg.inv(cov)
        
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

class RandomStartTrainer(ModelTrainer):
    def _init_responsibilities(self, data):
        '''
        Initialize randomly
        '''
        D = data.shape[1]
        
        data_lims = np.array([[data[:, d].min(), data[:, d].max()]
                              for d in range(D)])

        init_mean = np.array([uniform(*data_lims[d, :], size=self.num_comp)
                              for d in range(D)]).T

        covar_init = np.repeat([np.diag([1] * D)], self.num_comp, 0)

        labels = classify_by_distance(self.data,
                                      init_mean,
                                      covar_init)
        labels = labels.flatten()
        
        return self._labels_to_responsibilitis(labels)

#=======================================================================================================================
# Class for plotting model trainer data.
#=======================================================================================================================
class TrainerPlotter(object):
    def plot_result(self, colors=None, dim=(0, 1), title='', output='screen', plot_kwargs=dict(), savefig_kwargs=dict()):
        plot_clustering(self.data, self.codebook, colors, dim, title, output)

    def plot_clustering_ellipses(self, ESS=None, dims=[0, 1], scale=1):
        '''
        Make a scatterplot of the data with error ellipses
        '''
        if ESS is None:
            ESS = self.ESS

        plt.plot(self.data[:, dims[0]], self.data[:, dims[1]], 'o', ls='none')

        for k in range(ESS.num_comp):
            pos = ESS.smm_mean[k, :]
            cov = scale * ESS.smm_covar[k, :, :]
            plt.plot(pos[0], pos[1], 'r+')
            plot_ellipse(pos, cov, edge='red')

        plt.show()

    def _plot_monitor_init(self):
        '''
        Initialize plot monitor
        '''
        self.app = wx.App(False)
        self.frame = _MonitorPlot(self.data)
        self.frame.Show(True)
        self.app.Dispatch()

    def _plot_monitor_update(self, ESS):
        """Update plot monitor
        """
        self.frame.update_plot(ESS.smm_mean, ESS.smm_covar)


