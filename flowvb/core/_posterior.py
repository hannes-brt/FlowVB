from math import log
import numpy as np
from numpy.linalg import inv
from scipy.special import psi, erf
from scipy.optimize import fsolve
from enthought.traits.api import HasTraits, Array, Float, Int, Bool
from flowvb.utils import ind_retain_elements


class _Posterior(HasTraits):
    """Class to compute and store the posterior parameters of the model.

    """

    num_obs = Int()
    num_comp = Int()
    num_features = Int()

    dirichlet = Array()
    nws_mean = Array()
    nws_scale = Array()
    nws_dof = Array()
    nws_scale_matrix = Array()
    nws_scale_matrix_inv = Array()

    smm_dof = Array()
    smm_dof_init = Float()

    gausian = Bool()

    def __init__(self, Prior, num_comp, smm_dof_init=20,
                 gaussian=False, use_approx=True):
        """Initialize posterior parameters.

        """

        self.num_obs = Prior.num_obs
        self.num_features = Prior.num_features
        self.num_comp = num_comp

        self.dirichlet = Prior.dirichlet * np.ones(self.num_comp)
        self.nws_mean = np.tile(Prior.nws_mean, (self.num_comp, 1))
        self.nws_scale = Prior.nws_scale * np.ones(self.num_comp)
        self.nws_dof = Prior.nws_dof * np.ones(self.num_comp)

        self.smm_dof_init = smm_dof_init
        self.smm_dof = smm_dof_init * np.ones(self.num_comp)

        self.nws_scale_matrix = [Prior.nws_scale_matrix
                                 for k in range(self.num_comp)]

        self.nws_scale_matrix_inv = [inv(self.nws_scale_matrix[k, :, :])
                                         for k in range(self.num_comp)]

        self.use_approx = use_approx

    def update_parameters(self, Prior, ESS, LatentVariables):
        """Update posterior parameters.

        """
        self.dirichlet = self._update_dirichlet(self.num_obs,
                            ESS.smm_mixweights,
                            Prior.dirichlet)

        self.nws_scale = self._update_nws_scale(self.num_obs,
                            LatentVariables.latent_scaled_resp,
                            Prior.nws_scale)

        self.nws_mean = self._update_nws_mean(self.num_obs,
                            self.num_comp,
                            LatentVariables.latent_scaled_resp,
                            ESS.smm_mean,
                            Prior.nws_scale,
                            self.nws_scale,
                            Prior.nws_mean)

        self.nws_dof = self._update_nws_dof(self.num_obs,
                            ESS.smm_mixweights,
                            Prior.nws_dof)

        self.nws_scale_matrix = self._update_nws_scale_matrix(self.num_obs,
                            self.num_comp,
                            ESS.smm_mean,
                            Prior.nws_mean,
                            LatentVariables.latent_scaled_resp,
                            ESS.smm_covar,
                            Prior.nws_scale,
                            self.nws_scale,
                            Prior.nws_scale_matrix)

        self.nws_scale_matrix_inv = [inv(self.nws_scale_matrix[k, :, :])
                                     for k in range(self.num_comp)]

        if self.use_approx:
            self.smm_dof = self._update_smm_dof_approx(self.smm_dof,
                                self.num_obs,
                                self.num_comp,
                                ESS.smm_mixweights,
                                LatentVariables.latent_resp,
                                LatentVariables.latent_scale,
                                LatentVariables.latent_log_scale)
        else:
            self.smm_dof = self._update_smm_dof(self.smm_dof,
                                self.num_obs,
                                self.num_comp,
                                ESS.smm_mixweights,
                                LatentVariables.latent_resp,
                                LatentVariables.latent_scale,
                                LatentVariables.latent_log_scale)

    def remove_clusters(self, indices):
        """Remove clusters with insufficient support.

        """
        keep_indices = ind_retain_elements(indices, self.num_comp)

        self.num_comp = self.num_comp - len(indices)

        self.dirichlet = self.dirichlet[keep_indices]
        self.nws_scale = self.nws_scale[keep_indices]
        self.nws_dof = self.nws_dof[keep_indices]
        self.nws_mean = self.nws_mean[keep_indices, :]

        self.smm_dof = self.smm_dof[keep_indices]

        self.nws_scale_matrix = self.nws_scale_matrix[keep_indices, :, :]
        self.nws_scale_matrix_inv = \
                                  self.nws_scale_matrix_inv[keep_indices, :, :]

    @staticmethod
    def _update_dirichlet(num_obs,
                          smm_mixweights,
                          prior_dirichlet):
        """ Update `dirichlet` (Eq 27 in Arch2007) """

        posterior_dirichlet = num_obs * smm_mixweights + prior_dirichlet
        return posterior_dirichlet

    @staticmethod
    def _update_nws_scale(num_obs,
                          latent_scaled_resp,
                          prior_nws_scale):
        """ Update `nws_scale` (Eq 28 in Arch2007) """

        nws_scale = num_obs * latent_scaled_resp + prior_nws_scale
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
            scatter = np.outer((smm_mean[k, :] - prior_nws_mean),
                      (smm_mean[k, :] - prior_nws_mean))
            return (num_obs * scaled_resp[k] *
                    smm_covar[k, :, :] +
                    (num_obs * scaled_resp[k] *
                     prior_nws_scale) /
                    nws_scale[k] * scatter + prior_nws_scale_matrix)

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
                   sum(latent_resp[:, k] * \
                       (latent_log_scale[:, k] - latent_scale[:, k]))

            # By using the absolute value of dof, we prevent the
            # algorithm from failing when it tries dof < 0
            objective_func = lambda dof: (log(np.absolute(dof) / 2) + 1 -
                                          psi(np.absolute(dof) / 2) + frac)

            try:
                smm_dof_new = np.absolute(
                    fsolve(objective_func, smm_dof_old[k]))
            except ValueError:
                smm_dof_new = smm_dof_old[k]

            smm_dof = np.append(smm_dof, smm_dof_new)

        return smm_dof

    @staticmethod
    def _update_smm_dof_approx(smm_dof_old,
                               num_obs,
                               num_comp,
                               smm_mixweights,
                               latent_resp,
                               latent_scale,
                               latent_log_scale):
        """ Update `smm_dof` using the approximation of Shoham (2002) """

        smm_dof = smm_dof_old

        for k in range(num_comp):
            y = - (np.sum(latent_resp[:, k] * (latent_log_scale[:, k] -
                                             latent_scale[:, k])) /
                   (num_obs * smm_mixweights[k]))
            smm_dof_new = (2 / (y + log(y) - 1) +
                          0.0416 * (1 + erf(0.6594 *
                                            log(2.1971 / (y + log(y) - 1)))))
            if not np.isnan(smm_dof_new):
                smm_dof[k] = smm_dof_new

        return smm_dof
