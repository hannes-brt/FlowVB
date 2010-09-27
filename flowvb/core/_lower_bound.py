from enthought.traits.api import HasTraits, Int, Float, Instance
from _prior import _Prior


class _LowerBound(HasTraits):
    """Class to compute and store the lower bound.

    """

    num_obs = Int()
    num_features = Int()
    num_comp = Int()

    Prior = Instance(_Prior)

    log_dirichlet_const_init = Float()  # Not sure if this is correct
    log_wishart_const_init = Float()

    log_dirichlet_norm_init = Float()
    log_dirichlet_norm = Float()

    lower_bound = Float()

    def __init__(self):
        """Initialize lower bound.

        """
        pass
