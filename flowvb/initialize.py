import numpy as np
from numpy.random import multinomial
from scipy.spatial.distance import mahalanobis
from flowvb.utils import normalize_logspace


def init_d2_weighting(data, num_comp):

    num_obs = data.shape[0]

    cov_inv = np.linalg.inv(np.cov(data, rowvar=0))

    select_prob = np.ones(num_obs) / num_obs
    shortest_dist = np.inf * np.ones(num_obs)
    centroid = np.ones(num_comp)

    for k in range(num_comp):
        # Select a random data point as centroid
        centroid[k] = np.nonzero(multinomial(1, select_prob))[0]

        # Recompute distances
        for i, d in enumerate(shortest_dist):
            d_new = mahalanobis(data[centroid[k], :], data[i, :], cov_inv)
            if d_new < d: shortest_dist[i] = d_new

        select_prob = normalize_logspace(
            pow(shortest_dist.reshape(1, len(shortest_dist)), 2, 1))
        select_prob = select_prob.flatten()

    return centroid
