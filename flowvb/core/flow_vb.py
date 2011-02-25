import numpy as np
from numpy.random import uniform

from scipy.cluster.vq import whiten, kmeans2
from flowvb.core._ess import _ESS
from flowvb.core._latent_variables import _LatentVariables
from flowvb.core._lower_bound import _LowerBound
from flowvb.core._posterior import _Posterior
from flowvb.core._prior import _Prior
from flowvb.core._monitor_plot import _MonitorPlot
from flowvb.core._graphics import plot_clustering
from flowvb.utils import element_weights, plot_ellipse, \
     classify_by_distance, codebook
from flowvb.initialize import init_d2_weighting
import matplotlib.pyplot as plt
import wx

EPS = np.finfo(np.float).eps


class FlowVBAnalysis(object):
    '''
    Gate flow cytometry data using mixtures of Student-t densities
    '''
    def __init__(self, data, args):
        '''
        Fit the model to the data using Variational Bayes
        '''        
        # Save options in an options object.
        self.options = Options(args)
        
        if self.options.whiten_data:
            data = whiten(data)

        self.data = data           

        self._initialise_model()

        # Initial M-step
        self.posterior.update_parameters(self.prior,
                                         self.ess,
                                         self.latent_variables)

        # Main loop
        iteration = 1
        done = False

        if args.plot_monitor:
            self._plot_monitor = PlotMonitor(data)

        while not done:
            # Update parameters
            self._update_step()

            # Converged?
            if iteration == 1:
                converged = False
            else:
                converged = self._convergence_test()

            done = converged or (iteration >= args.max_iter)

            if args.plot_monitor:
                self._plot_monitor.update(self.ess)

            if args.verbose:
                print('iteration %d, lower bound: %f' % 
                      (iteration, self.lower_bound.lower_bound[-1]))

            iteration += 1

        self.codebook = codebook(self.latent_variables.latent_resp)

        # Call main loop of wxFrame to keep the window from closing
        if self.options.plot_monitor:
            self._plot_monitor.end_of_iteration()

    def __repr__(self):
        import flowvb.core._flow_vb_str

        # Add data dimensions to data dictionary
        opt = self.options.copy()
        opt.update({'num_obs': self.data.shape[0],
                    'num_features': self.data.shape[1]})

        # Build summary string
        str_summary = flowvb.core._flow_vb_str.str_summary_data
        str_summary += flowvb.core._flow_vb_str.str_summary_options_init_all

        if self.options['init_mean'] is not None:
            str_summary += flowvb.core._flow_vb_str.str_summary_init_mean
            opt['init_mean'] = np.array2string(opt['init_mean'])
        if self.options['init_covar'] is not None:
            str_summary += flowvb.core._flow_vb_str.str_summary_init_covar
            opt['init_covar'] = np.array2string(opt['init_covar'])
        if self.options['init_mixweights'] is not None:
            str_summary += flowvb.core._flow_vb_str.str_summary_init_mixweights
            opt['init_mixweights'] = np.array2string(opt['init_mixweights'])

        str_summary += flowvb.core._flow_vb_str.str_summary_optim_display

        return str_summary % opt
    
    def get_soft_labels(self):
        return self.latent_variables.latent_resp
    
    def get_labels(self):
        labels = np.argmax(self.latent_variables.latent_resp, axis=1)
        
        return labels
    
    def _initialise_model(self):
        data = self.data        
        (num_obs, num_features) = np.shape(data)        
        options = self.options
        
        initial_parameters = self._initialise_model_parameters()
        
        # Initialize data structures
        self.prior = _Prior(self.data,
                            options.num_comp_init,
                            options.prior_dirichlet)

        self.ess = _ESS(data,
                        options.num_comp_init,
                        initial_parameters['mean'],
                        initial_parameters['covar'],
                        initial_parameters['mixweights'])

        self.latent_variables = _LatentVariables(data,
                                                 self.ess,
                                                 options.num_comp_init)

        self.posterior = _Posterior(self.prior,
                                    options.num_comp_init,
                                    options.dof_init,
                                    use_approx=options.use_approx)

        self.lower_bound = _LowerBound(data,
                                       num_obs,
                                       num_features,
                                       options.num_comp_init,
                                       self.prior)
    
    def _initialise_model_parameters(self):
        # Choose method to intialize the parameters
        if self.options.init_method == 'd2-weighting':
            initialiser = D2Initialiser()
        elif self.options.init_method == 'kmeans':
            initialiser = KMeansInitialiser()
        elif self.options.init_method == 'random':
            initialiser = RandomInitialiser()

        if self.options.init_mean is None:
            initial_parameters = initialiser.initialise_parameters(self.data,
                                                                   self.options.num_comp_init)
        else:
            initialiser = UserParameterInitialiser()
            initial_parameters = initialiser.initialise_parameters(self.data,
                                                                   self.options.init_mean,
                                                                   self.options.init_covar,
                                                                   self.options.init_mixweights)            

            self.options.num_comp_init = initial_parameters['mean'].shape[0]
            
        return initial_parameters

    @staticmethod
    def _remove_empty_clusters(Prior, LatentVariables, ESS, Posterior,
                             LowerBound, remove_comp_thresh):
        """Remove components with insufficient support from the model
        """
        empty_cluster_indices = np.nonzero(
            ESS.smm_mixweights < remove_comp_thresh)[0]
        empty_cluster_indices = set(empty_cluster_indices)

        if len(empty_cluster_indices) > 0:
            Prior.remove_clusters(empty_cluster_indices)
            LatentVariables.remove_clusters(empty_cluster_indices)
            ESS.remove_clusters(empty_cluster_indices)
            Posterior.remove_clusters(empty_cluster_indices)
            LowerBound.remove_clusters(empty_cluster_indices)

    def _update_step(self):
        """Update the paramters
        """

        # E-step
        self.latent_variables.update_parameters(self.posterior)

        # Compute ancilliary statistics
        self.ess.update_parameters(self.prior,
                                   self.latent_variables)

        # Remove empty cluster
        self._remove_empty_clusters(self.prior,
                                    self.latent_variables,
                                    self.ess,
                                    self.posterior,
                                    self.lower_bound,
                                    self.options.remove_comp_thresh)

        # M-step
        self.posterior.update_parameters(self.prior,
                                         self.ess,
                                         self.latent_variables)

        # Compute the lower bound
        self.lower_bound.get_lower_bound(self.ess,
                                         self.prior,
                                         self.posterior,
                                         self.latent_variables)

    def _convergence_test(self):
        '''
        Test if iteration has converged
        '''
        converged = False

        fval = self.lower_bound.lower_bound[-1]
        previous_fval = self.lower_bound.lower_bound[-2]

        delta_fval = abs(fval - previous_fval)
        avg_fval = (abs(fval) + abs(previous_fval) + EPS) / 2
        
        if (delta_fval / avg_fval) < self.options.thresh:
            converged = True

        return converged

class Options(object):
    def __init__(self, args):
        # Initialisation
        self.num_comp_init = args.num_comp_init
        self.init_method = args.init_method
        self.prior_dirichlet = args.prior_dirichlet
        self.dof_init = args.dof_init
        
        # Training
        self.max_iter = args.max_iter
        self.thresh = args.thresh
        self.remove_comp_thresh = args.remove_comp_thresh
        self.use_approx = not args.use_exact
        
        # Pre-processing
        self.whiten_data = args.whiten_data
        
        # Output
        self.verbose = args.verbose
        self.plot_monitor = args.plot_monitor
        
        self.init_mean = None
        self.init_covar = None
        self.init_mixweights = None

class PlotMonitor(object):
    def __init__(self, data):
        '''
        Initialize plot monitor
        '''
        self.app = wx.App(False)
        self.frame = _MonitorPlot(data)
        self.frame.Show(True)
        self.app.Dispatch()

    def update(self, ess):
        '''
        Update plot monitor
        '''
        self.frame.update_plot(ess.smm_mean, ess.smm_covar)     
    
    def end_of_iteration(self):
        self.frame.end_of_iteration()
        self.app.MainLoop()
        
class AnalysisPlotter(object):
    def plot_result(self,
                    analysis,
                    colors=None,
                    dim=(0, 1),
                    title='',
                    output='screen',
                    plot_kwargs=dict(),
                    savefig_kwargs=dict()):
        
        plot_clustering(analysis.data, analysis.codebook, colors, dim, title, output)

    def plot_clustering_ellipses(self, data, ess, dims=[0, 1], scale=1):
        '''
        Make a scatterplot of the data with error ellipses
        '''
        plt.plot(data[:, dims[0]],
                 data[:, dims[1]],
                 'o',
                 ls='none')

        for k in range(ess.num_comp):
            pos = ess.smm_mean[k, :]
            cov = scale * ess.smm_covar[k, :, :]
            
            plt.plot(pos[0], pos[1], 'r+')
            
            plot_ellipse(pos, cov, edge='red')

        plt.show()

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
        
        init_parameters = {}
        init_parameters['mean'] = init_mean
        init_parameters['covar'] = init_covar
        init_parameters['mixweights'] = init_mixweights
            
        return init_parameters
    
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
        
        init_parameters = {}
        init_parameters['mean'] = init_mean
        init_parameters['covar'] = init_covar
        init_parameters['mixweights'] = init_mixweights
            
        return init_parameters
    
class KMeansInitialiser(Initialiser):
    def initialise_parameters(self, data, num_comp):
        '''
        Initialize using k-means
        '''
        (init_mean, labels) = kmeans2(data, num_comp)
        
        init_covar = self._get_covar(data, labels)
        
        init_mixweights = element_weights(labels)
        
        init_parameters = {}
        init_parameters['mean'] = init_mean
        init_parameters['covar'] = init_covar
        init_parameters['mixweights'] = init_mixweights
            
        return init_parameters
    
class UserParameterInitialiser(Initialiser):
    def _init_from_user_parameters(self, data, init_mean, init_covar, init_mixweights):
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
        
        init_parameters = {}
        init_parameters['mean'] = init_mean
        init_parameters['covar'] = init_covar
        init_parameters['mixweights'] = init_mixweights
            
        return init_parameters
