from _ess import _ESS
from _latent_variables import _LatentVariables
from _lower_bound import _LowerBound
from _posterior import _Posterior
from _prior import _Prior 

class FlowVB(object):
    """Gate flow cytometry data using mixtures of Student-t densities

    Attributes used by this class:
    
    - prior_obj                            - Prior parameter
    - ess_obj                              - Ancilliary statistics
    - latent_variables_obj                 - Latent variables
    - posterior_obj                        - Posterior parameters
    - lower_bound_obj                      - Variables for the lower bound
    - lower_bound_hist                     - The lower bound

    - data                                 - The data (NxD-dimensional)

    - no_components_init                   - Number of initial components
    - no_max_iter                          - Maximum number of iterations
    - threshold_stop_iter                  - Threshold to stop the iteration
    - verbose                              - Be verbose?
    - kappa_prior                          - Prior parameter for kappa 
    - approximate_dof                      - Use approximation for the update of the degrees of freedom?
    - threshold_remove_component           - Threshold to remove a component
    - gaussian                             - Use gaussian density (instead of Student-t)
    _ no_random_restarts                   - Number of random restarts
    - standardize                          - Standardize data?
    - principal_components_analysis        - Use principal components analysis?
    - pca_cutoff                           - PCA cutoff
    
    - no_components                        - Number of estimated components
    - responsabilities                     - Responsabilities
    - codebook                             - Hard labels
    - mixweights                           - Mixing weights of the components
    - component_centers                    - Centers of the components
    - component_precisions                 - Covariance matrices of the components
    - component_dof                        - Degrees of freedom of the components

    - no_components_merged                 - Number of components after flowMerge
    - responsabilities_merges              - Responsabilities after flowMerge
    - codebook_merged                      - Hard labels after flowMerge    

    """    

    def __init__(self):
        """Fit the model to the data using Variational Bayes
        
        """
        pass

    def plot():
        """Plot
        
        """
