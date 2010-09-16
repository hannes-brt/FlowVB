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

    def __init__():
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
    def _update_dirichlet_parameter_posterior(no_observations,
                                              mixweights,
                                              dirichlet_parameter_prior):
        """ Update `dirichlet_parameter_posterior` (Eq 27 in Arch2007) """
        pass

    @staticmethod
    def _update_normal_wishart_scale_posterior(no_observations,
                                               e_scaled_responsabilities,
                                               normal_wishart_scale_prior):
        """ Update `scale_prec_posterior` (Eq 28 in Arch2007) """
        pass

    @staticmethod
    def _update_normal_wishart_mean_posterior(no_observations,
                                              e_scaled_responsabilities,
                                              normal_wishart_scale_prior,
                                              normal_wishart_scale_posterior,
                                              normal_wishart_mean_prior):
        """ Update `normal_mean_posterior` (Eq 29 in Arch2007) """
        pass

    @staticmethod
    def _update_normal_wishart_dof_posterior(no_observations,
                                             mixweights,
                                             normal_wishart_dof_prior):
        """ Update `normal_wishart_dof_posterior` (Eq 30 in Arch2007) """
        pass

    @staticmethod
    def _update_normal_wishart_scale_matrix_posterior(no_observations,
                                                      e_component_mean,
                                                      normal_wishart_mean_prior,
                                                      e_scaled_responsabilities,
                                                      e_component_covar,
                                                      normal_wishart_scale_prior,
                                                      normal_wishart_scale_posterior,
                                                      normal_wishart_scale_matrix_prior):
        """ Update `normal_wishart_scale_matrix_posterior` (Eq 31 in Arch2007) """
        pass

    @staticmethod
    def _update_student_dof(no_observations,
                            no_components,
                            mixweights,
                            e_responsabilities,
                            e_scale_student,
                            e_log_scale_student):
        """ Update `student_dof` (Eq 36 in Arch2007) """
        pass
                                            
                                            
