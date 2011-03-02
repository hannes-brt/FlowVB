import numpy as np
from numpy.random import multivariate_normal as rmvnorm
from flowvb import FlowVBAnalysis
from flowvb.initialize import D2Initialiser
from argparse import Namespace

""" Demo using synthetic data with a large and a small cluster """

#np.random.seed(0)

mean = np.array([[0., -2.], [3., 3.]])
cov = np.array([[[2., 0.], [0., .2]], [[.2, 0], [0, .2]]])

n_obs = [2000, 100]

data = np.vstack([np.array(rmvnorm(mean[k, :], cov[k, :, :], n_obs[k]))
                  for k in range(mean.shape[0])])

args = Namespace()

args.num_comp_init = 2
args.thresh = 1e-5
args.max_iter = 10000
args.verbose = False
        
args.prior_dirichlet = 1e-2
args.dof_init = 2
args.remove_comp_thresh = 1e-6
        
args.use_exact = False
args.whiten_data = False
args.plot_monitor = True

args.init_params = D2Initialiser().initialise_parameters(data, args.num_comp_init)

model = FlowVBAnalysis(data, args)
