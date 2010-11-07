from enthought.traits.api import HasTraits, Int, Array
import numpy as np
from flowvb.utils import ind_retain_elements


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

    data = Array()

    def __init__(self, data, num_comp, smm_mean, smm_covar, smm_mixweights):
        """Initialize sufficient statistics.

        """

        self.num_obs = data.shape[0]
        self.num_features = data.shape[1]
        self.num_comp = num_comp

        self.smm_mean = smm_mean
        self.smm_covar = smm_covar
        self.smm_mixweights = smm_mixweights

        self.data = data

    def update_parameters(self, Prior, LatentVariables):
        """Update sufficient statistics.

        """

        self.smm_mean = self._update_smm_mean(self.data,
                            self.num_obs,
                            self.num_comp,
                            LatentVariables.latent_scaled_resp,
                            LatentVariables.latent_resp,
                            LatentVariables.latent_scale)

        self.smm_covar = self._update_smm_covar(self.data,
                            self.num_obs,
                            self.num_features,
                            self.num_comp,
                            LatentVariables.latent_resp,
                            LatentVariables.latent_scale,
                            LatentVariables.latent_scaled_resp,
                            self.smm_mean)

        self.smm_mixweights = self._update_smm_mixweights(self.num_obs,
                            LatentVariables.latent_resp)

    def remove_clusters(self, indices):
        """Remove clusters with insufficient support.

        """
        keep_indices = ind_retain_elements(indices, self.num_comp)

        self.num_comp = self.num_comp - len(indices)

        self.smm_mean = self.smm_mean[keep_indices]
        self.smm_covar = self.smm_covar[keep_indices, :, :]

        self.smm_mixweights = self.smm_mixweights[keep_indices]

    @staticmethod
    def _update_smm_mean(data, num_obs, num_comp, latent_scaled_resp,
                         latent_resp, latent_scale):
        """Update `smm_mean` (Eq 32 in Arch2007) """
        smm_mean = [np.sum([latent_resp[n, k] * latent_scale[n, k] * data[n, :]
                    for n in range(num_obs)], 0)
                    / (num_obs * latent_scaled_resp[k])
                    for k in range(num_comp)]

        return np.array(smm_mean)

    @staticmethod
    def _update_smm_covar(data, num_obs, num_features, num_comp,
                          latent_resp, latent_scale, latent_scaled_resp,
                          smm_mean):
        """Update `smm_covar` (Eq 33 in Arch2007) """

        def update(k):
            data_center = data - np.tile(smm_mean[k, :], (num_obs, 1))
            prod = (data_center *
                    np.tile(latent_resp[:, k] * latent_scale[:, k], (2, 1)).T)
            return np.dot(prod.T, data_center) /\
                   (num_obs * latent_scaled_resp[k])

        smm_covar = [update(k) for k in range(num_comp)]
        return smm_covar

    @staticmethod
    def _update_smm_mixweights(num_obs, latent_resp):
        """Update `smm_mixweights` (Eq 34 in Arch2007) """

        smm_mixweights = np.sum(latent_resp, 0) / num_obs
        return smm_mixweights
