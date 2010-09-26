from math import log
import numpy as np
from scipy.special import psi
from scipy.optimize import fsolve


class _Posterior(object):
    """Class to compute and store the posterior parameters of the model.

    """

    def __init__(self):
        """Initialize posterior parameters.

        """
        pass

    def update_parameters():
        """Update posterior parameters.

        """
        pass

    def remove_clusters():
        """Remove clusters with insufficient support.

        """
        pass

    @staticmethod
    def _update_posterior_dirichlet(num_obs,
                                    smm_mixweights,
                                    prior_dirichlet):
        """ Update `posterior_dirichlet` (Eq 27 in Arch2007) """

        posterior_dirichlet = num_obs * smm_mixweights + prior_dirichlet
        return posterior_dirichlet

    @staticmethod
    def _update_posterior_nws_scale(num_obs,
                                    latent_scaled_resp,
                                    prior_nws_scale):
        """ Update `posterior_nws_scale` (Eq 28 in Arch2007) """

        posterior_nws_scale = num_obs * latent_scaled_resp + \
                              prior_nws_scale
        return posterior_nws_scale

    @staticmethod
    def _update_posterior_nws_mean(num_obs,
                                   num_comp,
                                   latent_scaled_resp,
                                   smm_mean,
                                   prior_nws_scale,
                                   posterior_nws_scale,
                                   prior_nws_mean):
        """ Update `posterior_nws_mean` (Eq 29 in Arch2007) """

        update = lambda k: (num_obs * latent_scaled_resp[k] *
                            smm_mean[k, :] + prior_nws_scale *
                            prior_nws_mean) / posterior_nws_scale(k)

        posterior_nws_mean = np.array([update(k) for k in range(num_comp)])
        return posterior_nws_mean

    @staticmethod
    def _update_posterior_nws_dof(num_obs,
                                  smm_mixweights,
                                  prior_nws_dof):
        """ Update `normal_wishart_dof_posterior` (Eq 30 in Arch2007) """

        posterior_nws_dof = num_obs * smm_mixweights + prior_nws_dof
        return posterior_nws_dof

    @staticmethod
    def _update_posterior_nws_scale_matrix(num_obs,
                                           num_comp,
                                           smm_mean,
                                           prior_nws_mean,
                                           latent_scaled_resp,
                                           smm_covar,
                                           prior_nws_scale,
                                           posterior_nws_scale,
                                           prior_nws_scale_matrix):
        """ Update `posterior_nws_scale_matrix` (Eq 31 in Arch2007) """

        def update(k):
            scatter = (smm_mean[k, :] - prior_nws_mean).T * \
                      (smm_mean[k, :] - prior_nws_mean)

            return num_obs * latent_scaled_resp[k] * \
                   smm_covar[:, :, k] + \
                   (num_obs * latent_scaled_resp[k] *
                    prior_nws_scale) / \
                   posterior_nws_scale[k] * scatter + prior_nws_scale_matrix

        posterior_nws_scale_matrix = np.array([update(k)
                                               for k in range(num_comp)])
        return posterior_nws_scale_matrix

    @staticmethod
    def _update_smm_dof(smm_dof_old,
                        num_obs,
                        num_comp,
                        smm_mixweights,
                        latent_responsabilites,
                        latent_scale,
                        log_scale_student):
        """ Update `smm_dof` (Eq 36 in Arch2007) """

        smm_dof = np.array()

        for k in range(num_comp):
            frac = (1 / (num_obs * smm_mixweights[k])) * \
                   sum(latent_responsabilites[k, :] * \
                       (log_scale_student[k, :] - latent_scale[k, :]))
            objective_func = lambda dof: log(dof / 2) + 1 - psi(dof / 2) + frac

            try:
                smm_dof = np.append(smm_dof,
                                        fsolve(objective_func,
                                               smm_dof_old[k]))
            except:
                smm_dof = np.append(smm_dof,
                                        smm_dof_old[k])

        return smm_dof
