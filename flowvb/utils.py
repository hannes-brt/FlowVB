"""Miscalleneous functions
"""
import math
from math import pi, sqrt

import numpy as np
from numpy import log
from numpy.linalg import cholesky

from matplotlib.patches import Ellipse
from pylab import gca

from scipy.special import gammaln, psi

def multipsi(x, d):
    '''
    Implements the multivariate digamma (psi) function.
    '''
    i = np.arange(1, d + 1)
    
    mp = psi((x + 1 - i) / 2)
    mp = mp.sum()
    
    return mp

def repeat(x, func, *args, **kargs):
    """Call the function `func` `x` times and accumulate the results in a list
    """
    return [func(*args, **kargs) for _ in range(x)]


def logdet(matrix):
    """Computes the log of the determinant of a matrix
    """

    U = cholesky(matrix)
    logdet = 2 * np.sum(np.diag(np.log(U)))
    return logdet


def mvt_gamma_ln(n, alpha):
    """ Returns the log of multivariate gamma(n, alpha) value.
    necessary for avoiding underflow/overflow problems
    alpha > (n-1)/2
    """
    n = float(n)
    logp = (((n * (n - 1)) / 4) * log(pi) + 
            np.sum(gammaln(np.tile(alpha, (n, 1)).T
                           + 0.5 * np.arange(0, -n, -1)), 1))
    return logp


def normalize(vector):
    """Normalizes a vector to have sum one
    """

    return np.array(vector) / sum(vector)


def normalize_logspace(mat):
    """ Normalizes the rows of a matrix while avoiding numerical underflow
    """

    if np.array(mat).ndim == 1:
        mat = np.array(mat).reshape(1, len(mat))

    L = logsumexp(mat)
    d = mat.shape[1]
    return np.exp(mat - np.tile(L, (d, 1)).T)


def logsumexp(mat, dim=1):
    """ Returns log(sum(exp(a))) while avoiding numerical underflow

    """
    max_dim = np.max(mat, dim)                      # Maximum of each row
    mat = mat - max_dim.reshape((len(max_dim), 1))  # Substract row maximum
    s = max_dim + np.log(np.sum(np.exp(mat), 1))

    idx_inf = np.nonzero(~np.isfinite(s))          # Deal with inf entries

    if (len(idx_inf) != 0):
        s[idx_inf] = max_dim[idx_inf]

    return s


def standardize(a, dim=0):
    """Standardize the vector `a` to mean zero and sd 1 """
    return (a - np.mean(a)) / np.std(a)


def arrays_almost_equal(a, b, accuracy=1e-3):
    """Check if two arrays are approximately equal.

    Parameters
    ----------
    a : array_like
    b : array_like
    accuracy : float (optional)
       The maximum difference of any array element for which the two arrays
       are assumed to be equal.

    Returns
    _______
    approx_equal : bool
       Whether the two arrays are approximately equal.
    """

    try:
        d = np.absolute(a - b)
        return (d < accuracy).all()
    except TypeError:
        return False


def element_weights(a):
    """Returns the proportions with which every element in a vector occurs
    """
    elements = range(max(a) + 1)
    a = np.array(a)

    mixweights = np.array([float(np.sum(a == x)) / len(a)
                           for x in elements])
    return mixweights


def ind_retain_elements(indices, num_comp):
    keep_indices = set(range(num_comp))
    keep_indices = list(keep_indices - indices)
    return keep_indices


def plot_ellipse(pos, P, edge='black', face='none'):
    """Plots an error ellipse.

    By Tinne De Laet
    <http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/
    msg14153.html>

    Parameters
    ----------
    pos : array_like
       A two-element vector giving the center of the ellipse.
    P : array_like
       A (2x2)-dimensional covariance matrix.
    edge : color_specification
    face: color_specification

    Returns
    -------
    ellipse : matplotlib.patches.Ellipse-object

    """
    U, s, Vh = np.linalg.svd(P)
    orient = math.atan2(U[1, 0], U[0, 0]) * 180 / pi
    ellipsePlot = Ellipse(xy=pos, width=2.0 * math.sqrt(s[0]),
                          height=2.0 * math.sqrt(s[1]),
                          angle=orient, facecolor=face,
                          edgecolor=edge)
    ax = gca()
    ax.add_patch(ellipsePlot)
    return ellipsePlot


def classify_by_distance(data, mean, covar):
    K = mean.shape[0]

    covar_inv = np.array([np.linalg.inv(covar[k, :, :]) for k in range(K)])

    def mahal_dist(x, k):
        return sqrt(np.dot(np.dot((x - mean[k, :]).T, covar_inv[k, :, :]),
                    (x - mean[k, :])))

    def classify_point(x):
        dist = np.array([mahal_dist(x, k) for k in range(K)])
        return np.nonzero(dist == min(dist))[0]

    labels = np.apply_along_axis(classify_point, 1, data)
    return labels


def codebook(resp):
    mmax = np.max(resp, 1)
    return np.array([np.nonzero(resp[i, :] == mmax[i])[0]
                     for i in range(resp.shape[0])]).flatten()
