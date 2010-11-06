from enthought.traits.api import HasTraits, Int, Float, Instance
from _prior import _Prior
import numpy as np
from numpy import log
from scipy.special import gammaln
from flowvb.utils import logdet, mvt_gamma_ln
from math import pi
import pdb


class _LowerBound(HasTraits):
    """Class to compute and store the lower bound.

    """

    num_obs = Int()
    num_features = Int()
    num_comp = Int()

    Prior = Instance(_Prior)

    log_dirichlet_const_init = Float()  # Not sure if this is correct
    log_wishart_const_init = Float()

    log_dirichlet_norm_init = Float()
    log_dirichlet_norm = Float()

    lower_bound = Float()

    def __init__(self):
        """Initialize lower bound.

        """
        pass

    def update_lower_bound(self):
        """Update the lower bound """
        pass

    @staticmethod
    def _log_dirichlet_normalization_prior(num_comp, prior_dirichlet):
        """Compute the normalization constant for the dirichlet distribution"""
        log_dirichlet_normalization_prior = gammaln(num_comp *
                                                    prior_dirichlet[0])
        return log_dirichlet_normalization_prior

    @staticmethod
    def _log_dirichlet_normalization(num_obs, num_comp, prior_dirichlet):
        """Compute the normalization constant for the dirichlet distribution"""
        log_dirichlet_normalization = gammaln(num_obs +
                                              num_comp * prior_dirichlet[0])
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
                               mvt_gamma_ln(int(num_features), nws_dof / 2))
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
                       latent_resp, latent_log_scale, latent_scaled_resp,
                       expect_log_pu):
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
        pass

    @staticmethod
    def _expect_log_ptheta(num_comp, num_features, prior_nws_mean,
                           prior_dirichlet, prior_nws_dof, prior_nws_scale,
                           prior_nws_scale_matrix, posterior_nws_mean,
                           posterior_nws_dof, posterior_nws_scale,
                           posterior_nws_scale_matrix_inv):
        """Compute `expect_log_ptheta` (Eq 43 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_qu(num_obs, num_comp, gamma_param_alpha, gamma_param_beta,
                       latent_resp, smm_mixweights):
        """Compute `expect_log_qu` (Eq 44 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_qz(latent_resp):
        """Compute `expect_log_qz` (Eq 45 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_qtheta(num_comp, num_features, log_wishart_const,
                           log_dirichlet_normalization, posterior_dirichlet,
                           posterior_nws_scale, posterior_nws_dof,
                           posterior_nws_scale_matrix, log_smm_mixweight,
                           log_det_precision):
        """Compute `expect_log_qtheta` (Eq 46 in Arch2007) """
        pass
