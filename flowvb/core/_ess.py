from enthought.traits.api import HasTraits, Int, Array
import numpy as np


class _ESS(HasTraits):
    """Class to compute and store the sufficient statistics.

    """

    num_obs = Int()
    num_features = Int()
    num_comp = Int()

    smm_mean = Array()
    smm_covar = Array()
    smm_mixweights = Array()
    scaled_resp = Array()

    def __init__(self, data, num_comp, smm_mean, smm_covar, smm_mixweights):
        """Initialize sufficient statistics.

        """

        self.num_obs = data.shape[0]
        self.num_features = data.shape[1]
        self.num_comp = num_comp

        self.smm_mean = smm_mean
        self.smm_covar = smm_covar
        self.smm_mixweights = smm_mixweights
        self.scaled_resp = smm_mixweights

    def update_parameters():
        """Update sufficient statistics.

        """
        pass

    def remove_clusters():
        """Remove clusters with insufficient support.

        """
        pass

    @staticmethod
    def _update_smm_mean(data, num_obs, num_comp, scaled_resp,
                         latent_resp, latent_scale):
        """Update `smm_mean` (Eq 32 in Arch2007) """

        smm_mean = [np.sum([latent_resp[k, n] * latent_scale[k, n] * data[n, :]
                    for n in range(num_obs)], 0)
                    / (num_obs * scaled_resp[k, :])
                    for k in range(num_comp)]

        return smm_mean

    @staticmethod
    def _update_smm_covar(data, num_obs, num_features, num_comp,
                          latent_resp, latent_scale, smm_mean):
        """Update `smm_covar` (Eq 33 in Arch2007) """

        def update(k):
            data_center = data - np.tile(smm_mean[k, :], (num_features, 1))
            return np.sum([latent_resp[k, n] * latent_scale[k, n] *
                           np.dot(data_center, data_center.T)
                           for n in range(num_obs)], 0)
        smm_covar = [update(k) for k in range(num_comp)]
        return smm_covar

    @staticmethod
    def _update_smm_mixweights(num_obs, latent_resp):
        """Update `smm_mixweights` (Eq 34 in Arch2007) """

        smm_mixweights = 1 / num_obs * np.sum(latent_resp, 0)
        return smm_mixweights

    @staticmethod
    def _update_scaled_resp(num_obs, latent_resp, latent_scale):
        """Update `scaled_resp` (Eq 35 in Arch2007) """

        scaled_resp = 1 / num_obs * np.sum(latent_resp * latent_scale, 0)
        return scaled_resp
