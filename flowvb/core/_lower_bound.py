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

    def update_lower_bound(self):
        """Update the lower bound """
        pass

    @staticmethod
    def _log_dirichlet_const():
        """Compute `log_dirichlet_const` """
        pass

    @staticmethod
    def _log_wishart_const():
        """Compute `log_wishart_const_init` """
        pass

    @staticmethod
    def _expect_log_px():
        """Compute `expect_log_px` (Eq 40 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_pu():
        """Compute `expect_log_pu` (Eq 41 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_pz():
        """Compute `expect_log_pz` (Eq 42 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_ptheta():
        """Compute `expect_log_ptheta` (Eq 43 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_qu():
        """Compute `expect_log_qu` (Eq 44 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_qz():
        """Compute `expect_log_qz` (Eq 45 in Arch2007) """
        pass

    @staticmethod
    def _expect_log_qtheta():
        """"Compute `expect_log_qtheta` (Eq 46 in Arch2007) """
        pass
