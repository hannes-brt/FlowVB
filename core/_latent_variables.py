class _LatentVariables(object):
    """Class to compute and store the latent variables.
    """

    responsabilities = None                    # rho in Arch2007
    scale_prec = None                          # u in Arch2007
    log_scale_prec = None                      # log(u) in Arch2007

    log_pi_tilde = None                        
    log_lambda_tilde = None

    num_observations = None                    # N in Arch2007
    num_dimensions = None                      # D in Arch2007
    num_components = None                      # K in Arch2007 
    
    
    def __init__():
        """Initialize latent variables.
        """
        pass

    def update_parameters():
        """Update latent variables.
        """
        pass

    def remove_clusters():
        """Remove clusters with insufficient support.
        """
        pass
