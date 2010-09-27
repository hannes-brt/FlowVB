from numpy import eye, mat
from enthought.traits.api import HasTraits, Int, Float, Array


class _Prior(HasTraits):
    """Class to compute and store the prior parameters of the model.

    """

    num_obs = Int()
    num_comp = Int()
    num_features = Int()

    dirichlet = Float()
    nws_mean = Float()
    nws_scale = Float()
    nws_dof = Float()
    nws_scale_matrix = Array()

    def __init__(self, data, num_comp, prior_dirichlet=1e-3, nws_mean=0,
                 nws_scale=1, nws_dof=20, nws_scale_matrix=1 / 200.):
        """Initialize prior paramters.

        """

        self.num_obs = data.shape[0]
        self.num_features = data.shape[1]

        self.dirichlet = prior_dirichlet
        self.nws_mean = nws_mean
        self.nws_scale = nws_scale
        self.nws_dof = nws_dof
        self.nws_scale_matrix = nws_scale_matrix * mat(eye(self.num_features))
