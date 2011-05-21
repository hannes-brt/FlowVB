import numpy as np
from numpy.random import multivariate_normal as rmvnorm
from flowvb import FlowVBAnalysis
from flowvb.initialize import D2Initialiser
from argparse import Namespace

""" Demo using synthetic data with three wide but short clusters """

# np.random.seed(0)

mean = np.array([[0., -2.], [0., 0.], [0., 2.]])
cov = np.array([[2., 0.], [0., .2]])

n_obs = 2000

data = np.vstack([np.array(rmvnorm(mean[k, :], cov, n_obs))
                  for k in range(mean.shape[0])])

args = Namespace()

args.num_comp_init = 6
args.thresh = 1e-5
args.max_iter = 10000
args.verbose = False
        
args.prior_dirichlet = 1e-2
args.dof_init = 2
args.remove_comp_thresh = 1e-4
        
args.use_exact = False
args.whiten_data = False
args.plot_monitor = True

args.init_params = D2Initialiser().initialise_parameters(data, args.num_comp_init)

model = FlowVBAnalysis(data, args)
