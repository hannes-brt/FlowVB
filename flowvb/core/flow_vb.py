from enthought.traits.api import HasTraits, Instance
import numpy as np
from scipy.cluster.vq import whiten, kmeans2
from flowvb.core._ess import _ESS
from flowvb.core._latent_variables import _LatentVariables
from flowvb.core._lower_bound import _LowerBound
from flowvb.core._posterior import _Posterior
from flowvb.core._prior import _Prior
from flowvb.utils import element_weights, plot_ellipse
import matplotlib.pyplot as plt
from pylab import gca

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
                 prior_dirichlet=1e-3,
                 dof_init=2,
                 remove_comp_thresh=1e-2,
                 whiten_data=False):
        """Fit the model to the data using Variational Bayes

        """

        (num_obs, num_features) = np.shape(data)

        if whiten_data:
            data = whiten(data)

        self.data = data

        # Initialize with k-means
        (centroids, labels) = kmeans2(data, num_comp_init)
        smm_covar_init = [np.diag(np.ones(num_features))
                          for _ in range(num_comp_init)]
        smm_mixweights_init = element_weights(labels)

        # Initialize data structures
        Prior = _Prior(data, num_comp_init, prior_dirichlet)

        ESS = _ESS(data, num_comp_init, centroids, smm_covar_init,
                   smm_mixweights_init)

        LatentVariables = _LatentVariables(data, ESS, num_comp_init)

        Posterior = _Posterior(Prior, num_comp_init, dof_init)

        LowerBound = _LowerBound(data, num_obs, num_features,
                                 num_comp_init, Prior)

        # Initial M-step
        Posterior.update_parameters(Prior, ESS, LatentVariables)

        # Main loop
        iter = 1
        done = False

        while not done:

            # Remove empty cluster
            self.remove_empty_clusters(Prior, LatentVariables, ESS,
                                       Posterior, LowerBound,
                                       remove_comp_thresh)

            # E-step
            LatentVariables.update_parameters(Posterior)

            # Compute ancilliary statistics
            ESS.update_parameters(Prior, LatentVariables)

            # M-step
            Posterior.update_parameters(Prior, ESS, LatentVariables)

            # Compute the lower bound
            LowerBound.get_lower_bound(ESS, Prior, Posterior, LatentVariables)

            # Converged?
            if iter == 1:
                converged = False
            else:
                converged = self.convergence_test(LowerBound, thresh)

            done = converged or (iter > max_iter)

            if verbose:
                print('iteration %d, lower bound: %f' %
                      (iter, LowerBound.lower_bound[-1]))

            iter += 1

        self.Posterior = Posterior
        self.Prior = Prior
        self.LatentVariables = LatentVariables
        self.ESS = ESS
        self.LowerBound = LowerBound

    def plot_clustering_ellipses(self, Posterior=None, dims=[0, 1]):
        if Posterior is None:
            Posterior = self.Posterior

        plt.plot(self.data[:, dims[0]], self.data[:, dims[1]], 'o', ls='none')

        for k in range(Posterior.num_comp):
            pos = Posterior.nws_mean[k, :]
            cov = Posterior.nws_scale_matrix[k, :, :]
            plot_ellipse(pos, cov, edge='red')

        plt.show()

    @staticmethod
    def convergence_test(LowerBound, thresh=1e-4):
        converged = False

        fval = LowerBound.lower_bound[-1]
        previous_fval = LowerBound.lower_bound[-2]

        delta_fval = abs(fval - previous_fval)
        avg_fval = (abs(fval) + abs(previous_fval) + EPS) / 2
        if (delta_fval / avg_fval) < thresh:
            converged = True

        return converged

    @staticmethod
    def remove_empty_clusters(Prior, LatentVariables, ESS, Posterior,
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
