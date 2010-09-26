import numpy as np
from scipy.special import psi
from scipy.optimize import fsolve

class _Posterior(object):
    """Class to compute and store the posterior parameters of the model.

    Attributes used by this class:
    - dirichlet_parameter_posterior    - Parameter to the Dirichlet distribution
                                         (kappa_m in Arch2007)
    - scale_prec_posterior             - Scale parameter for the precision matrix
                                         (eta_m in Arch2007)
    - wishart_dof_posterior            - Degrees of freedom of the Wishart distribution
                                         (gamma_m in Arch2007)
    - normal_mean_posterior            - Expectation of the Normal distribution
                                         (m_m in Arch2007)
    - wishart_precision_posterior      - Precision matrix of the Wishart distribution
                                         (S_m in Arch2007)
    - dof                              - Degrees of freedom of the components
                                         (nu_m in Arch2007)

    - no_observations                  - Number of observations
    - no_components                    - No of components
    
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
                                    e_mixweights,
                                    prior_dirichlet):
        """ Update `posterior_dirichlet` (Eq 27 in Arch2007) """

        posterior_dirichlet = num_obs * e_mixweights + prior_dirichlet
        return posterior_dirichlet


    @staticmethod
    def _update_posterior_nws_scale(num_obs,
                                    e_scaled_responsabilities,
                                    prior_nws_scale):
        """ Update `posterior_nws_scale` (Eq 28 in Arch2007) """

        posterior_nws_scale = num_obs * e_scaled_responsabilities \
                              + prior_nws_scale
        return posterior_nws_scale


    @staticmethod
    def _update_posterior_nws_mean(num_obs,
                                   num_comp,
                                   e_scaled_responsabilities,
                                   prior_nws_scale,
                                   posterior_nws_scale,
                                   prior_nws_mean):
        """ Update `posterior_nws_mean` (Eq 29 in Arch2007) """

        update = lambda k: (num_obs * e_scaled_responsabilities[k] * 
                            e_component_mean[k,:] + prior_nws_scale * 
                            prior_nws_mean) / posterior_nws_scale(k)
        
        posterior_nws_mean = np.array([ update(k) for k in range(num_comp) ])
        return posterior_nws_mean

    
    @staticmethod
    def _update_posterior_nws_dof(num_obs,
                                  e_mixweights,
                                  prior_nws_dof):
        """ Update `normal_wishart_dof_posterior` (Eq 30 in Arch2007) """

        posterior_nws_dof = num_obs * e_mixweights + prior_nws_dof
        return posterior_nws_dof

    @staticmethod
    def _update_posterior_nws_scale_matrix(num_obs,
                                           num_comp,
                                           e_component_mean,
                                           prior_nws_mean,
                                           e_scaled_responsabilities,
                                           e_component_covar,
                                           prior_nws_scale,
                                           posterior_nws_scale,
                                           prior_nws_scale_matrix):
        """ Update `posterior_nws_scale_matrix` (Eq 31 in Arch2007) """

        def update(k):
            scatter = (e_component_mean[k,:] - prior_nws_mean).T * \
                      (e_component_mean[k,:] - prior_nws_mean)

            return num_obs * e_scaled_responsabilities[k] * \
                   e_component_covar[:,:,k] + \
                   (num_obs * e_scaled_responsabilities[k] *
                    prior_nws_scale) / \
                   posterior_nws_scale[k] * scatter + prior_nws_scale_matrix

        posterior_nws_scale_matrix = np.array([ update(k) for k in range(num_comp) ])
        return posterior_nws_scale_matrix

    @staticmethod
    def _update_student_dof(student_dof_old,
                            num_obs,
                            num_comp,
                            mixweights,
                            e_responsabilities,
                            e_scale_student,
                            e_log_scale_student):
        """ Update `student_dof` (Eq 36 in Arch2007) """

        student_dof = np.array()

        for k in range(num_comp):
            frac = (1 / (num_obs * mixweights[k])) * \
                   sum(e_responsabilities[k,:] * \
                       (e_log_scale_student[k,:] - e_scale_student[k,:]))
            objective_func = lambda dof: log(dof / 2) + 1 - psi(dof / 2) + frac

            try:
                student_dof = np.append(student_dof,
                                        fsolve(objective_func, self.student_dof[k]))
            else:
                student_dof = np.append(student_dof,
                                        student_dof_old[k])

        return student_dof
