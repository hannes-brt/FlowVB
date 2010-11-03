from enthought.traits.api import HasTraits, Int, Float, Instance
from _prior import _Prior


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
        pass

    @staticmethod
    def _log_dirichlet_normalization(num_obs, num_comp, prior_dirichlet):
        """Compute the normalization constant for the dirichlet distribution"""
        pass

    @staticmethod
    def _log_dirichlet_const(prior_dirichlet):
        """Compute `log_dirichlet_const` """
        pass

    @staticmethod
    def _log_wishart_const(num_features, posterior_nws_dof,
                           posterior_nws_scale_matrix):
        """Compute `log_wishart_const_init` """
        pass

    @staticmethod
    def _expect_log_px(num_obs, num_features, num_comp, latent_resp,
                       latent_scale, latent_log_scale,
                       posterior_nws_scale_matrix_inv, smm_mixweights,
                       latent_scaled_resp, posterior_nws_dof,
                       posterior_nws_scale, log_smm_mixweight):
        """Compute `expect_log_px` (Eq 40 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_pu(num_obs, num_comp, smm_mixweights, smm_dof,
                       latent_resp, latent_log_scale, latent_scaled_resp,
                       expect_log_pu):
        """Compute `expect_log_pu` (Eq 41 in Arch2007) """
        pass

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
