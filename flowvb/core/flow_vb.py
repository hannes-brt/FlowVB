from enthought.traits.api import HasTraits, Instance
import numpy as np
from numpy.random import uniform
from scipy.cluster.vq import whiten, kmeans2
from flowvb.core._ess import ESS
from flowvb.core._latent_variables import LatentVariables
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


class FlowVBAnalysis(HasTraits):
    """Gate flow cytometry data using mixtures of Student-t densities

    """

    Prior = Instance(_Prior)
    Posterior = Instance(_Posterior)
    ESS = Instance(ESS)
    LatentVariables = Instance(LatentVariables)
    LowerBound = Instance(_LowerBound)

    def __init__(self, data,
                 num_comp_init=10,
                 max_iter=200,
                 thresh=1e-5,
                 verbose=False,
                 init_mean=None,
                 init_covar=None,
                 init_mixweights=None,
                 init_method='d2-weighting',
                 prior_dirichlet=1e-3,
                 dof_init=2,
                 remove_comp_thresh=1e-2,
                 whiten_data=False,
                 plot_monitor=False,
                 use_approx=True):

        """Fit the model to the data using Variational Bayes

        """

        (num_obs, num_features) = np.shape(data)

        if whiten_data:
            data = whiten(data)

        self.data = data

        # Save options in a dictionary
        self.options = {
            'num_comp_init': num_comp_init,
            'max_iter': max_iter,
            'thresh': thresh,
            'verbose': verbose,
            'init_mean': init_mean,
            'init_covar': init_covar,
            'init_mixweights': init_mixweights,
            'init_method': init_method,
            'prior_dirichlet': prior_dirichlet,
            'dof_init': dof_init,
            'remove_comp_thresh': remove_comp_thresh,
            'whiten_data': whiten_data,
            'plot_monitor': plot_monitor,
            'use_approx': use_approx
            }

        # Choose method to intialize the parameters
        if init_method == 'd2-weighting':
            init_method = self._init_d2_weighting
        elif init_method == 'kmeans':
            init_method = self._init_kmeans
        elif init_method == 'random':
            init_method = self._init_random

        if init_mean is None:
            # No starting solution was supplied
            # Initialize with `init_method`
            (init_mean, labels, init_covar, init_mixweights) = \
                        init_method(num_comp_init)
        else:
            # Starting solution was supplied
            num_comp_init = init_mean.shape[0]
            if init_mixweights is None:
                labels = classify_by_distance(data, init_mean,
                                              init_covar)
                init_mixweights = element_weights(labels)
            if init_covar is None:
                init_covar = self._get_covar(data, labels)

        # Initialize data structures
        Prior = _Prior(data, num_comp_init, prior_dirichlet)

        ESS = ESS(data, num_comp_init, init_mean, init_covar,
                   init_mixweights)

        LatentVariables = LatentVariables(data, ESS, num_comp_init)

        Posterior = _Posterior(Prior, num_comp_init, dof_init,
                               use_approx=use_approx)

        LowerBound = _LowerBound(data, num_obs, num_features,
                                 num_comp_init, Prior)

        # Initial M-step
        Posterior.update(Prior, ESS, LatentVariables)

        # Main loop
        iteration = 1
        done = False

        if plot_monitor:
            self._plot_monitor_init()

        while not done:

            # Update parameters
            self._update_step(Prior, Posterior, ESS,
                              LatentVariables, LowerBound)

            # Converged?
            if iteration == 1:
                converged = False
            else:
                converged = self._convergence_test(LowerBound, thresh)

            done = converged or (iteration >= max_iter)

            if plot_monitor:
                self._plot_monitor_update(ESS)

            if verbose:
                print('iteration %d, lower bound: %f' % 
                      (iteration, LowerBound.lower_bound[-1]))

            iteration += 1

        self.Posterior = Posterior
        self.Prior = Prior
        self.LatentVariables = LatentVariables
        self.ESS = ESS
        self.LowerBound = LowerBound
        self.codebook = codebook(self.LatentVariables.latent_resp)

        # Call main loop of wxFrame to keep the window from closing
        if plot_monitor:
            self.frame.end_of_iteration()
            self.app.MainLoop()

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
        return self.LatentVariables.latent_resp
    
    def get_labels(self):
        labels = np.argmax(self.LatentVariables.latent_resp, axis=1)
        
        return labels





    @staticmethod
    def _get_covar(data, labels, *args, **kargs):
        """Compute the covariance in all clusters
        """
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


