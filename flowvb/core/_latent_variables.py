from enthought.traits.api import HasTraits, Int, Array
import numpy as np


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

    def __init__(self, data, num_comp):
        """Initialize latent variables.

        """

        self.num_obs = data.shape[0]
        self.num_features = data.shape[1]
        self.num_comp = num_comp

        self.latent_resp = np.nan * np.zeros(self.num_obs, self, num_comp)
        self.latent_scale = np.nan * np.zeros(self.num_obs, self.num_comp)
        self.latent_log_scale = np.nan * np.zeros(self.num_obs, self.num_comp)

        self.log_smm_mixweight = np.nan * np.zeros(self.num_comp)
        self.log_det_precision = np.nan * np.zeros(self.num_comp)

        self.gamma_param_alpha = np.nan * np.zeros(self.num_comp)
        self.gamma_param_beta = np.nan * np.zeros(self.num_obs, self.num_comp)

    def update_parameters(self):
        """Update latent variables.

        """
        pass

    def remove_clusters(self):
        """Remove clusters with insufficient support.

        """

        pass

    @staticmethod
    def _update_latent_resp(data, smm_dof, posterior_nws_scale,
                            log_smm_mixweight, log_det_precision, scatter):
        """ Update `latent_resp` (Eq 22 in Arch2007) """
        pass

    @staticmethod
    def _update_latent_scale(gamma_param_alpha, gamma_param_beta):
        """ Update `latent_scale` """
        pass

    @staticmethod
    def _update_latent_log_scale(gamma_param_alpha, gamma_param_beta):
        """ Update `latent_log_scale` """
        pass

    @staticmethod
    def _update_latent_scaled_resp(num_obs, latent_resp, latent_scale):
        """Update `latent_scaled_resp` (Eq 35 in Arch2007) """

        latent_scaled_resp = np.sum(latent_resp * latent_scale, 0) / num_obs
        return latent_scaled_resp

    @staticmethod
    def _update_log_smm_mixweight(posterior_dirichlet):
        """ Update `log_smm_mixweight` """
        pass

    @staticmethod
    def _update_log_det_precision(num_features, num_comp, posterior_nws_dof,
                                  posterior_nws_scale_matrix):
        """ Update `log_det_precision` """
        pass

    @staticmethod
    def _update_gamma_param_alpha(num_features, smm_dof):
        """ Update `gamma_param_alpha` """
        pass

    @staticmethod
    def _update_gamma_param_beta(posterior_nws_dof, posterior_nws_scale,
                                 scatter):
        """ Update `gamma_param_beta` """
        pass

    @staticmethod
    def _get_scatter_(data, posterior_nws_scale_matrix_inv,
                      posterior_nws_mean):
        """ Compute the scatter """
        pass
