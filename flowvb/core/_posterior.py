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
