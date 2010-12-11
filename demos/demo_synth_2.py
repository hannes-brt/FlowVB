import numpy as np
from numpy.random import multivariate_normal as rmvnorm
from flowvb import FlowVB

""" Demo using synthetic data with a large and a small cluster """

#np.random.seed(0)

mean = np.array([[0., -2.], [3., 3.]])
cov = np.array([[[2., 0.], [0., .2]], [[.2, 0], [0, .2]]])

n_obs = [2000, 100]

data = np.vstack([np.array(rmvnorm(mean[k, :], cov[k, :, :], n_obs[k]))
                  for k in range(mean.shape[0])])

model = FlowVB(data, 2, thresh=1e-5, init_method='d2-weighting',
               plot_monitor=True, max_iter=10000)
