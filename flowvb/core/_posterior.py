from math import log
import numpy as np
from scipy.special import psi
from scipy.optimize import fsolve
from enthought.traits.api import HasTraits, Array, Float, Int, Bool, Instance
from _prior import _Prior


class _Posterior(HasTraits):
    """Class to compute and store the posterior parameters of the model.

    """

    num_obs = Int()
    num_comp = Int()
    num_features = Int()

    Prior = Instance(_Prior)

    dirichlet = Array()
    nws_mean = Array()
    nws_scale = Array()
    nws_dof = Array()
    nws_scale_matrix = Array()

    smm_dof = Array()
    smm_dof_init = Float()

    gausian = Bool()

    def __init__(self, Prior, num_comp, smm_dof_init=20, gaussian=False):
        """Initialize posterior parameters.

        """

        self.Prior = Prior

        self.num_obs = Prior.num_obs
        self.num_features = Prior.num_features
        self.num_comp = num_comp

        self.dirichlet = Prior.dirichlet * np.ones(self.num_comp)
        self.nws_mean = Prior.nws_mean * np.ones((self.num_comp,
                                                  self.num_features))
        self.nws_scale = Prior.nws_scale * np.ones(self.num_comp)
        self.nws_dof = Prior.nws_dof * np.ones(self.num_comp)

        self.smm_dof_init = smm_dof_init
        self.smm_dof = smm_dof_init * np.ones(self.num_comp)

        self.nws_scale_matrix = [Prior.nws_scale_matrix
                                 for k in range(self.num_comp)]

    def update_parameters():
        """Update posterior parameters.

        """
        pass

    def remove_clusters():
        """Remove clusters with insufficient support.

        """
        pass

    @staticmethod
    def _update_dirichlet(num_obs,
                          smm_mixweights,
                          prior_dirichlet):
        """ Update `dirichlet` (Eq 27 in Arch2007) """

        posterior_dirichlet = num_obs * smm_mixweights + prior_dirichlet
        return posterior_dirichlet

    @staticmethod
    def _update_nws_scale(num_obs,
                          scaled_resp,
                          prior_nws_scale):
        """ Update `nws_scale` (Eq 28 in Arch2007) """

        nws_scale = num_obs * scaled_resp + \
                              prior_nws_scale
        return nws_scale

    @staticmethod
    def _update_nws_mean(num_obs,
                         num_comp,
                         scaled_resp,
                         smm_mean,
                         prior_nws_scale,
                         nws_scale,
                         prior_nws_mean):
        """ Update `nws_mean` (Eq 29 in Arch2007) """

        update = lambda k: (num_obs * scaled_resp[k] *
                            smm_mean[k, :] + prior_nws_scale *
                            prior_nws_mean) / nws_scale[k]

        posterior_nws_mean = np.array([update(k) for k in range(num_comp)])
        return posterior_nws_mean

    @staticmethod
    def _update_nws_dof(num_obs,
                        smm_mixweights,
                        prior_nws_dof):
        """ Update `nws_dof` (Eq 30 in Arch2007) """

        posterior_nws_dof = num_obs * smm_mixweights + prior_nws_dof
        return posterior_nws_dof

    @staticmethod
    def _update_nws_scale_matrix(num_obs,
                                 num_comp,
                                 smm_mean,
                                 prior_nws_mean,
                                 scaled_resp,
                                 smm_covar,
                                 prior_nws_scale,
                                 nws_scale,
                                 prior_nws_scale_matrix):
        """ Update `nws_scale_matrix` (Eq 31 in Arch2007) """

        def update(k):
            scatter = (smm_mean[k, :] - prior_nws_mean).T * \
                      (smm_mean[k, :] - prior_nws_mean)

            return num_obs * scaled_resp[k] * \
                   smm_covar[:, :, k] + \
                   (num_obs * scaled_resp[k] *
                    prior_nws_scale) / \
                   nws_scale[k] * scatter + prior_nws_scale_matrix

        posterior_nws_scale_matrix = np.array([update(k)
                                               for k in range(num_comp)])
        return posterior_nws_scale_matrix

    @staticmethod
    def _update_smm_dof(smm_dof_old,
                        num_obs,
                        num_comp,
                        smm_mixweights,
                        latent_resp,
                        latent_scale,
                        latent_log_scale):
        """ Update `smm_dof` (Eq 36 in Arch2007) """

        smm_dof = np.array([])

        for k in range(num_comp):
            frac = (1 / (num_obs * smm_mixweights[k])) * \
                   sum(latent_resp[k, :] * \
                       (latent_log_scale[k, :] - latent_scale[k, :]))
            objective_func = lambda dof: log(dof / 2) + 1 - psi(dof / 2) + frac

            try:
                smm_dof = np.append(smm_dof,
                                        fsolve(objective_func,
                                               smm_dof_old[k]))
            except:
                smm_dof = np.append(smm_dof,
                                        smm_dof_old[k])

        return smm_dof
