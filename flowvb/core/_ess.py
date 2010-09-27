from enthought.traits.api import HasTraits, Int, Array


class _ESS(HasTraits):
    """Class to compute and store the sufficient statistics.

    """

    num_obs = Int()
    num_features = Int()
    num_comp = Int()

    smm_mean = Array()
    smm_covar = Array()
    smm_mixweights = Array()

    def __init__(self, data, num_comp, smm_mean, smm_covar, smm_mixweights):
        """Initialize sufficient statistics.

        """

        self.num_obs = data.shape[0]
        self.num_features = data.shape[1]
        self.num_comp = num_comp

        self.smm_mean = smm_mean
        self.smm_covar = smm_covar
        self.smm_mixweights = smm_mixweights

    def update_parameters():
        """Update sufficient statistics.

        """
        pass

    def remove_clusters():
        """Remove clusters with insufficient support.

        """
        pass
