from enthought.traits.api import HasTraits, Array, Int, Float
import numpy as np
from numpy import log, dot
from scipy.special import gammaln, psi
from flowvb.utils import logdet, mvt_gamma_ln
from math import pi

EPS = np.finfo(np.float).eps


class _LowerBound(HasTraits):
    """Class to compute and store the lower bound.

    """

    num_obs = Int()
    num_features = Int()
    num_comp = Int()

    log_dirichlet_const_init = Float()
    log_wishart_const_init = Float()

    log_dirichlet_norm_init = Float()
    log_dirichlet_norm = Float()

    lower_bound = Array()

    data = Array()

    def __init__(self, data, num_obs, num_features, num_comp, Prior):
        """Initialize lower bound.

        """
        self.num_obs = num_obs
        self.num_features = num_features
        self.num_comp = num_comp

        self.data = data

        self.log_dirichlet_const_init = self._log_dirichlet_const(
                                        Prior.dirichlet)
        self.log_wishart_const_init = self._log_wishart_const(num_features,
                                        Prior.nws_dof,
                                        Prior.nws_scale_matrix)

        self.log_dirichlet_norm_init = \
                self._log_dirichlet_normalization_prior(num_comp,
                                                       Prior.dirichlet)
        self.log_dirichlet_norm = \
                self._log_dirichlet_normalization(num_obs, num_comp,
                                                 Prior.dirichlet)

    def remove_clusters(self, indices):
        self.num_comp = self.num_comp - len(indices)

    def get_lower_bound(self, ESS, Prior, Posterior, LatentVariables):
        """Update the lower bound """

        log_wishart_const = [self._log_wishart_const(self.num_features,
                                        Posterior.nws_dof[k],
                                        Posterior.nws_scale_matrix[k, :, :])
                             for k in range(self.num_comp)]

        expect_log_px = self._expect_log_px(self.data,
                                            self.num_obs,
                                            self.num_features,
                                            self.num_comp,
                                            LatentVariables.latent_resp,
                                            LatentVariables.latent_scale,
                                            LatentVariables.latent_log_scale,
                                            LatentVariables.latent_scaled_resp,
                                            Posterior.nws_mean,
                                            Posterior.nws_scale_matrix_inv,
                                            Posterior.nws_dof,
                                            Posterior.nws_scale,
                                            ESS.smm_mixweights,
                                            LatentVariables.log_det_precision)

        expect_log_pu = self._expect_log_pu(self.num_obs,
                                            self.num_comp,
                                            ESS.smm_mixweights,
                                            Posterior.smm_dof,
                                            LatentVariables.latent_resp,
                                            LatentVariables.latent_log_scale,
                                            LatentVariables.latent_scaled_resp)

        expect_log_pz = self._expect_log_pz(self.num_comp,
                                            LatentVariables.latent_resp,
                                            LatentVariables.log_smm_mixweight)

        expect_log_ptheta = self._expect_log_ptheta(self.num_comp,
                                            self.num_features,
                                            Prior.nws_mean,
                                            Prior.dirichlet,
                                            Prior.nws_dof,
                                            Prior.nws_scale,
                                            Prior.nws_scale_matrix,
                                            Posterior.nws_mean,
                                            Posterior.nws_dof,
                                            Posterior.nws_scale,
                                            Posterior.nws_scale_matrix_inv,
                                            LatentVariables.log_smm_mixweight,
                                            LatentVariables.log_det_precision,
                                            self.log_wishart_const_init,
                                            self.log_dirichlet_norm_init)

        expect_log_qu = self._expect_log_qu(self.num_obs,
                                            self.num_comp,
                                            LatentVariables.gamma_param_alpha,
                                            LatentVariables.gamma_param_beta,
                                            LatentVariables.latent_resp,
                                            ESS.smm_mixweights)

        expect_log_qz = self._expect_log_qz(LatentVariables.latent_resp)

        expect_log_qtheta = self._expect_log_qtheta(self.num_comp,
                                            self.num_features,
                                            log_wishart_const,
                                            self.log_dirichlet_norm,
                                            Posterior.dirichlet,
                                            Posterior.nws_scale,
                                            Posterior.nws_dof,
                                            Posterior.nws_scale_matrix,
                                            LatentVariables.log_smm_mixweight,
                                            LatentVariables.log_det_precision)

        self.lower_bound = np.append(self.lower_bound,
                                     expect_log_px + expect_log_pu +
                                     expect_log_pz + expect_log_ptheta -
                                     expect_log_qu - expect_log_qz -
                                     expect_log_qtheta)

    @staticmethod
    def _log_dirichlet_normalization_prior(num_comp, prior_dirichlet):
        """Compute the normalization constant for the dirichlet distribution"""
        log_dirichlet_normalization_prior = gammaln(num_comp *
                                                    prior_dirichlet)
        return log_dirichlet_normalization_prior

    @staticmethod
    def _log_dirichlet_normalization(num_obs, num_comp, prior_dirichlet):
        """Compute the normalization constant for the dirichlet distribution"""
        log_dirichlet_normalization = gammaln(num_obs +
                                              num_comp * prior_dirichlet)
        return log_dirichlet_normalization

    @staticmethod
    def _log_dirichlet_const(prior_dirichlet):
        """Compute `log_dirichlet_const` """
        log_dirichlet_const = (gammaln(np.sum(prior_dirichlet)) -
                               np.sum(gammaln(prior_dirichlet)))
        return log_dirichlet_const

    @staticmethod
    def _log_wishart_const(num_features, nws_dof,
                           nws_scale_matrix):
        """Compute `log_wishart_const_init` """
        log_wishart_const = ((nws_dof / 2) * logdet(nws_scale_matrix) -
                               (nws_dof * num_features / 2) * log(2) -
                               mvt_gamma_ln(num_features, nws_dof / 2))
        return log_wishart_const

    @staticmethod
    def _expect_log_px(data, num_obs, num_features, num_comp, latent_resp,
                       latent_scale, latent_log_scale, latent_scaled_resp,
                       posterior_nws_mean, posterior_nws_scale_matrix_inv,
                       posterior_nws_dof, posterior_nws_scale,
                       smm_mixweights, log_det_precision):
        """Compute `expect_log_px` (Eq 40 in Arch2007) """

        def update(k):
            #Centered data
            data_center = (data -
                           np.tile(posterior_nws_mean[k, :], (num_obs, 1)))

            # Mahalanobis distance
            S = np.sum(np.dot(data_center,
                              posterior_nws_scale_matrix_inv[k, :, :]) *
                       data_center, 1)
            mahal_dist = np.sum(latent_resp[:, k] * latent_scale[:, k] * S)

            return (num_features *
                    np.sum(latent_resp[:, k] * latent_log_scale[:, k]) +

                    num_obs * log_det_precision[k] * smm_mixweights[k] -

                    posterior_nws_dof[k] * mahal_dist -

                    num_obs * num_features *
                    (latent_scaled_resp[k] / posterior_nws_scale[k]))

        expect_log_px = np.array([update(k) for k in range(num_comp)])
        expect_log_px = (-(num_obs * num_features / 2) * log(2 * pi) +
                         0.5 * sum(expect_log_px))
        return expect_log_px

    @staticmethod
    def _expect_log_pu(num_obs, num_comp, smm_mixweights, smm_dof,
                       latent_resp, latent_log_scale, latent_scaled_resp):
        """Compute `expect_log_pu` (Eq 41 in Arch2007) """
        def update(k):
            return (0.5 * num_obs * smm_mixweights[k] * smm_dof[k] *
                    log(smm_dof[k] / 2) -

                    num_obs * smm_mixweights[k] * gammaln(smm_dof[k] / 2) +
                    (smm_dof[k] / 2 - 1) *
                    np.sum(latent_resp[:, k] * latent_log_scale[:, k]) -

                    0.5 * num_obs * smm_dof[k] * latent_scaled_resp[k])

        expect_log_pu = np.sum([update(k) for k in range(num_comp)])
        return expect_log_pu

    @staticmethod
    def _expect_log_pz(num_comp, latent_resp, log_smm_mixweight):
        """Compute `expect_log_pz` (Eq 42 in Arch2007) """
        update = lambda k: np.sum(latent_resp[:, k] * log_smm_mixweight[k])

        expect_log_pz = np.sum([update(k) for k in range(num_comp)])
        return expect_log_pz

    @staticmethod
    def _expect_log_ptheta(num_comp, num_features, prior_nws_mean,
                           prior_dirichlet, prior_nws_dof, prior_nws_scale,
                           prior_nws_scale_matrix, posterior_nws_mean,
                           posterior_nws_dof, posterior_nws_scale,
                           posterior_nws_scale_matrix_inv,
                           log_smm_mixweight, log_det_precision,
                           log_wishart_const_init,
                           log_dirichlet_normalization_prior):
        """Compute `expect_log_ptheta` (Eq 43 in Arch2007) """

        def update(k):
            mc = posterior_nws_mean[k, :] - prior_nws_mean

            return ((prior_dirichlet - 1) * log_smm_mixweight[k] -

                    (prior_nws_scale * posterior_nws_dof[k] / 2) *
                    dot(dot(mc, posterior_nws_scale_matrix_inv[k, :, :]),
                        mc.T) -

                    0.5 * num_features * prior_nws_scale /
                    posterior_nws_scale[k] +

                    0.5 * (prior_nws_dof - num_features) *
                    log_det_precision[k] -

                    0.5 * posterior_nws_dof[k] *
                    np.trace(dot(prior_nws_scale_matrix,
                                 posterior_nws_scale_matrix_inv[k, :, :])) -

                    gammaln(prior_dirichlet))

        constant_factors = (log_dirichlet_normalization_prior +

                            num_comp * log_wishart_const_init -
                            0.5 * num_comp * num_features *
                            log(2 * pi * prior_nws_scale))

        expect_log_ptheta = (constant_factors +
                             np.sum([update(k) for k in range(num_comp)]))
        return expect_log_ptheta

    @staticmethod
    def _expect_log_qu(num_obs, num_comp, gamma_param_alpha, gamma_param_beta,
                       latent_resp, smm_mixweights):
        """Compute `expect_log_qu` (Eq 44 in Arch2007) """

        update = lambda k: num_obs * (-gammaln(gamma_param_alpha[k]) *
                                      smm_mixweights[k] +

                                      smm_mixweights[k] *
                                      psi(gamma_param_alpha[k]) *
                                      (gamma_param_alpha[k] - 1) +

                                      np.sum(latent_resp[:, k] *
                                             log(gamma_param_beta[:, k]))
                                      / num_obs -

                                      smm_mixweights[k] * gamma_param_alpha[k])

        expect_log_qu = np.sum([update(k) for k in range(num_comp)])
        return expect_log_qu

    @staticmethod
    def _expect_log_qz(latent_resp):
        """Compute `expect_log_qz` (Eq 45 in Arch2007) """
        expect_log_qz = np.sum(latent_resp * log(latent_resp + EPS))
        return expect_log_qz

    @staticmethod
    def _expect_log_qtheta(num_comp, num_features, log_wishart_const,
                           log_dirichlet_normalization, posterior_dirichlet,
                           posterior_nws_scale, posterior_nws_dof,
                           posterior_nws_scale_matrix, log_smm_mixweight,
                           log_det_precision):
        """Compute `expect_log_qtheta` (Eq 46 in Arch2007) """

        update = lambda k: ((posterior_dirichlet[k] - 1) *
                            log_smm_mixweight[k] +

                            0.5 * num_features * log(posterior_nws_scale[k]) +

                            log_wishart_const[k] +

                            0.5 * (posterior_nws_dof[k] - num_features) *
                            log_det_precision[k] -

                            0.5 * (posterior_nws_dof[k] * num_features) -

                            gammaln(posterior_dirichlet[k]))

        constant_factors = (log_dirichlet_normalization -
                            0.5 * num_comp * num_features * (log(2 * pi) + 1))

        expect_log_qtheta = (constant_factors +
                             np.sum([update(k) for k in range(num_comp)]))
        return expect_log_qtheta
