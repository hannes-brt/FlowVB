import numpy as np
from numpy.random import multivariate_normal as rmvnorm
from flowvb import FlowVBAnalysis

""" Demo using synthetic data with three wide but short clusters """

# np.random.seed(0)

mean = np.array([[0., -2.], [0., 0.], [0., 2.]])
cov = np.array([[2., 0.], [0., .2]])

n_obs = 2000

data = np.vstack([np.array(rmvnorm(mean[k, :], cov, n_obs))
                  for k in range(mean.shape[0])])

model = FlowVBAnalysis(data, 6, thresh=1e-5, init_method='d2-weighting',
               plot_monitor=True, max_iter=10000,
               remove_comp_thresh=1e-4)
