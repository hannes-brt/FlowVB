from enthought.traits.api import HasTraits, Instance
import numpy as np
from scipy.cluster.vq import whiten, kmeans2
from flowvb.core._ess import _ESS
from flowvb.core._latent_variables import _LatentVariables
from flowvb.core._lower_bound import _LowerBound
from flowvb.core._posterior import _Posterior
from flowvb.core._prior import _Prior
from flowvb.core._monitor_plot import _MonitorPlot
from flowvb.utils import element_weights, plot_ellipse, classify_by_distance
import matplotlib.pyplot as plt
import wx
import pudb

EPS = np.finfo(np.float).eps


class FlowVB(HasTraits):
    """Gate flow cytometry data using mixtures of Student-t densities

    """

    Prior = Instance(_Prior)
    Posterior = Instance(_Posterior)
    ESS = Instance(_ESS)
    LatentVariables = Instance(_LatentVariables)
    LowerBound = Instance(_LowerBound)

    def __init__(self, data,
                 num_comp_init=10,
                 max_iter=200,
                 thresh=1e-5,
                 verbose=True,
                 init_mean=None,
                 init_covar=None,
                 init_mixweights=None,
                 prior_dirichlet=1e-3,
                 dof_init=2,
                 remove_comp_thresh=1e-2,
                 whiten_data=False,
                 plot_monitor=False):
        """Fit the model to the data using Variational Bayes

        """

        (num_obs, num_features) = np.shape(data)

        if whiten_data:
            data = whiten(data)

        self.data = data
        self.remove_comp_thresh = remove_comp_thresh

        if init_mean is None:
            # No starting solution was supplied
            # Initialize with k-means
            (init_mean, labels) = kmeans2(data, num_comp_init)
            init_covar = self._get_covar(data, labels)
            init_mixweights = element_weights(labels)

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

        ESS = _ESS(data, num_comp_init, init_mean, init_covar,
                   init_mixweights)

        LatentVariables = _LatentVariables(data, ESS, num_comp_init)

        Posterior = _Posterior(Prior, num_comp_init, dof_init)

        LowerBound = _LowerBound(data, num_obs, num_features,
                                 num_comp_init, Prior)

        # Initial M-step
        Posterior.update_parameters(Prior, ESS, LatentVariables)

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
                print Posterior.nws_mean

            iteration += 1

        self.Posterior = Posterior
        self.Prior = Prior
        self.LatentVariables = LatentVariables
        self.ESS = ESS
        self.LowerBound = LowerBound
        self.app.MainLoop()

    def plot_clustering_ellipses(self, ESS=None, dims=[0, 1], scale=1):
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
        self.app = wx.App(False)
        self.frame = _MonitorPlot(self.data)
        self.frame.Show(True)
        self.app.Dispatch()

    def _plot_monitor_update(self, ESS):
        self.frame.update_plot(ESS.smm_mean, ESS.smm_covar)

    @staticmethod
    def _remove_empty_clusters(Prior, LatentVariables, ESS, Posterior,
                             LowerBound, remove_comp_thresh):

        empty_cluster_indices = np.nonzero(
            ESS.smm_mixweights < remove_comp_thresh)[0]
        empty_cluster_indices = set(empty_cluster_indices)

        if len(empty_cluster_indices) > 0:
            Prior.remove_clusters(empty_cluster_indices)
            LatentVariables.remove_clusters(empty_cluster_indices)
            ESS.remove_clusters(empty_cluster_indices)
            Posterior.remove_clusters(empty_cluster_indices)
            LowerBound.remove_clusters(empty_cluster_indices)

    def _update_step(self, Prior, Posterior, ESS, LatentVariables, LowerBound):
        # E-step
        LatentVariables.update_parameters(Posterior)

        # Compute ancilliary statistics
        ESS.update_parameters(Prior, LatentVariables)

        # Remove empty cluster
        self._remove_empty_clusters(Prior, LatentVariables, ESS,
                                    Posterior, LowerBound,
                                    self.remove_comp_thresh)

        # M-step
        Posterior.update_parameters(Prior, ESS, LatentVariables)

        # Compute the lower bound
        LowerBound.get_lower_bound(ESS, Prior, Posterior, LatentVariables)

    @staticmethod
    def _convergence_test(LowerBound, thresh=1e-4):
        converged = False

        fval = LowerBound.lower_bound[-1]
        previous_fval = LowerBound.lower_bound[-2]

        delta_fval = abs(fval - previous_fval)
        avg_fval = (abs(fval) + abs(previous_fval) + EPS) / 2
        if (delta_fval / avg_fval) < thresh:
            converged = True

        return converged

    @staticmethod
    def _get_covar(data, labels, *args, **kargs):
        elements = range(max(labels) + 1)

        def covar(m):
            # Make sure, a dxd-matrix is returned, even when there are
            # only zero one observations
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
