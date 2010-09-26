class _Prior(object):
    """Class to compute and store the prior parameters of the model.

    Attributes used by this class:

    - dirichlet_parameter_prior            - Parameter for the Dirichlet distribution
                                             (kappa_0 in Arch2007)
    - scale_prec_prior                     - Scale parameter for the precision matrix
                                             (eta_0 in Arch2007)
    - wishart_dof_posterior                - Degrees of freedom of the Wishart distribution
                                             (gamma_0 in Arch2007)
    - normal_mean_prior                    - Expectation of the Normal distribution
                                             (m_0 in Arch2007)
    - wishart_precision_prior              - Precision matrix of the Wishart distribution
                                             (S_0 in Arch2007)
    - dof_init                             - Degrees of freedom of the components
                                             
    """

    def __init__(self):
        """Initialize prior paramters.

        """
        pass

    def update_parameters():
        """Update prior parameters.

        """
        pass

    def remove_clusters():
        """Remove clusters with insufficient support.

        """
        pass
