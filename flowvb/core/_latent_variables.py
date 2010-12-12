from enthought.traits.api import HasTraits, Int, Array
import numpy as np
from numpy import log
from scipy.special import gammaln, psi
from math import pi
from flowvb.utils import logdet, ind_retain_elements
from flowvb.normalize import normalize_logspace


class _LatentVariables(HasTraits):
    """Class to compute and store the latent variables.

    """

    num_obs = Int()
    num_features = Int()
    num_comp = Int()

    latent_resp = Array()
    latent_scale = Array()
    latent_log_scale = Array()
    latent_scaled_resp = Array()

    log_smm_mixweight = Array()
    log_det_precision = Array()

    gamma_param_alpha = Array()
    gamma_param_beta = Array()

    data = Array()

    def __init__(self, data, ESS, num_comp):
        """Initialize latent variables.

        """

        self.num_obs = data.shape[0]
        self.num_features = data.shape[1]
        self.num_comp = num_comp

        self.latent_resp = np.nan * np.zeros([self.num_obs, self.num_comp])
        self.latent_scale = np.ones([self.num_obs, self.num_comp])
        self.latent_log_scale = (np.nan * np.
                                 zeros([self.num_obs, self.num_comp]))
        self.latent_scaled_resp = ESS.smm_mixweights

        self.log_smm_mixweight = np.nan * np.zeros([self.num_comp])
        self.log_det_precision = np.nan * np.zeros([self.num_comp])

        self.gamma_param_alpha = np.nan * np.zeros([self.num_comp])
        self.gamma_param_beta = (np.nan *
                                 np.zeros([self.num_obs, self.num_comp]))

        self.data = data

    def update_parameters(self, Posterior):
        """Update latent variables.

        """

        scatter = self._get_scatter(self.data,
                            Posterior.nws_scale_matrix_inv,
                            Posterior.nws_mean)

        self.log_smm_mixweight = self._update_log_smm_mixweight(
                            Posterior.dirichlet)

        self.log_det_precision = self._update_log_det_precision(
                            self.num_features,
                            self.num_comp,
                            Posterior.nws_dof,
                            Posterior.nws_scale_matrix)

        self.gamma_param_alpha = self._update_gamma_param_alpha(
                            self.num_features,
                            Posterior.smm_dof)

        self.gamma_param_beta = self._update_gamma_param_beta(
                            self.num_features,
                            Posterior.smm_dof,
                            Posterior.nws_dof,
                            Posterior.nws_scale,
                            scatter)

        self.latent_resp = self._update_latent_resp(self.data,
                            Posterior.smm_dof,
                            Posterior.nws_scale,
                            Posterior.nws_dof,
                            self.log_smm_mixweight,
                            self.log_det_precision,
                            scatter)

        self.latent_scale = self._update_latent_scale(self.gamma_param_alpha,
                            self.gamma_param_beta)

        self.latent_log_scale = self._update_latent_log_scale(
                            self.gamma_param_alpha,
                            self.gamma_param_beta)

        self.latent_scaled_resp = self._update_latent_scaled_resp(self.num_obs,
                            self.latent_resp,
                            self.latent_scale)

    def remove_clusters(self, indices):
        """Remove clusters with insufficient support.

        """
        keep_indices = ind_retain_elements(indices, self.num_comp)

        self.num_comp = self.num_comp - len(indices)

        self.latent_resp = self.latent_resp[:, keep_indices]
        self.latent_scale = self.latent_scale[:, keep_indices]
        self.latent_log_scale = self.latent_log_scale[:, keep_indices]
        self.latent_scaled_resp = self.latent_scaled_resp[:, keep_indices]

        self.log_smm_mixweight = self.log_smm_mixweight[keep_indices]
        self.log_det_precision = self.log_det_precision[keep_indices]

        self.gamma_param_alpha = self.gamma_param_alpha[keep_indices]
        self.gamma_param_beta = self.gamma_param_beta[:, keep_indices]

    @staticmethod
    def _update_latent_resp(data, smm_dof, posterior_nws_scale,
                            posterior_nws_dof, log_smm_mixweight,
                            log_det_precision, scatter):
        """ Update `latent_resp` (Eq 22 in Arch2007) """
        num_features = data.shape[1]
        num_comp = len(log_smm_mixweight)

        def get_exp_latent(k):
            return (gammaln((num_features + smm_dof[k]) / 2) -
                    gammaln(smm_dof[k] / 2) -
                    (num_features / 2) * log(smm_dof[k] * pi) +
                    log_smm_mixweight[k] + log_det_precision[k] / 2 -
                    ((num_features + smm_dof[k]) / 2) *
                    log(1 + (posterior_nws_dof[k] / smm_dof[k]) *
                        scatter[k, :] +
                        num_features / (smm_dof[k] * posterior_nws_scale[k])))

        exp_latent = np.array([get_exp_latent(k) for k in range(num_comp)]).T

        latent_resp = normalize_logspace(exp_latent)
        return latent_resp

    @staticmethod
    def _update_latent_scale(gamma_param_alpha, gamma_param_beta):
        """ Update `latent_scale` """
        num_obs = np.shape(gamma_param_beta)[0]

        latent_scale = (np.tile(gamma_param_alpha, (num_obs, 1)) /
                        gamma_param_beta)
        return latent_scale

    @staticmethod
    def _update_latent_log_scale(gamma_param_alpha, gamma_param_beta):
        """ Update `latent_log_scale` """
        num_obs = np.shape(gamma_param_beta)[0]

        latent_log_scale = (np.tile(psi(gamma_param_alpha), (num_obs, 1)) -
                            log(gamma_param_beta))
        return latent_log_scale

    @staticmethod
    def _update_latent_scaled_resp(num_obs, latent_resp, latent_scale):
        """Update `latent_scaled_resp` (Eq 35 in Arch2007) """

        latent_scaled_resp = np.sum(latent_resp * latent_scale, 0) / num_obs
        return latent_scaled_resp

    @staticmethod
    def _update_log_smm_mixweight(posterior_dirichlet):
        """ Update `log_smm_mixweight` """
        log_smm_mixweight = (psi(posterior_dirichlet) -
                             psi(np.sum(posterior_dirichlet)))
        return log_smm_mixweight

    @staticmethod
    def _update_log_det_precision(num_features, num_comp, posterior_nws_dof,
                                  posterior_nws_scale_matrix):
        """ Update `log_det_precision` """

        update = lambda k: (np.sum(psi((posterior_nws_dof[k] +
                                        1 - range(1, num_features + 1)) / 2)) +
                            num_features * log(2) -
                            logdet(posterior_nws_scale_matrix[k, :, :]))
        log_det_precision = [update(k) for k in range(num_comp)]
        return log_det_precision

    @staticmethod
    def _update_gamma_param_alpha(num_features, smm_dof):
        """ Update `gamma_param_alpha` """
        gamma_param_alpha = (num_features + smm_dof) / 2
        return gamma_param_alpha

    @staticmethod
    def _update_gamma_param_beta(num_features, smm_dof, posterior_nws_dof,
                                 posterior_nws_scale, scatter):
        """ Update `gamma_param_beta` """
        num_comp = np.shape(posterior_nws_dof)[0]

        update = lambda k: ((posterior_nws_dof[k] / 2) * scatter[k, :] +
                            num_features / (2 * posterior_nws_scale[k]) +
                            smm_dof[k] / 2)

        gamma_param_beta = np.array([update(k) for k in range(num_comp)]).T
        return gamma_param_beta

    @staticmethod
    def _get_scatter(data, posterior_nws_scale_matrix_inv,
                      posterior_nws_mean):
        """ Compute the scatter """
        num_obs = np.shape(data)[0]
        num_comp = np.shape(posterior_nws_mean)[0]

        def update(k):
            data_center = data - np.tile(posterior_nws_mean[k, :],
                                         (num_obs, 1))
            prod = np.dot(data_center,
                          posterior_nws_scale_matrix_inv[k, :, :])
            return np.sum(prod * data_center, 1)

        scatter = np.array([update(k) for k in range(num_comp)])
        return scatter
