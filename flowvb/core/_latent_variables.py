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

    def update_parameters():
        """Update latent variables.

        """
        pass

    def remove_clusters():
        """Remove clusters with insufficient support.

        """
        pass
