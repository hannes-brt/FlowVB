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
    def _update_latent_resp(data, Posterior,
                            log_smm_mixweight,
                            log_det_precision):
        """ Update `latent_resp` (Eq 22 in Arch2007) """
        pass

    @staticmethod
    def _update_latent_scale():
        """ Update `latent_scale` """
        pass

    @staticmethod
    def _update_log_scale():
        """ Update `latent_scale` """
        pass

    @staticmethod
    def _update_log_smm_mixweights():
        """ Update `log_smm_mixweight` """
        pass

    @staticmethod
    def _update_log_det_precision():
        """ Update `log_det_precision` """
        pass

    @staticmethod
    def _update_gamma_param_alpha():
        """ Update `gamma_param_alpha` """
        pass

    @staticmethod
    def _update_gamma_param_beta():
        """ Update `gamma_param_beta` """
        pass
