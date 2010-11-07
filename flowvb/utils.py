"""Miscalleneous functions
"""

import numpy as np
from numpy import log
from numpy.linalg import cholesky
from scipy.maxentropy import logsumexp
from scipy.special import gammaln
from math import pi


def repeat(x, func, *args, **kargs):
    """Call the function `func` `x` times and accumulate the results in a list
    """
    return [func(*args, **kargs) for _ in range(x)]


def logdet(matrix):
    """Computes the log of the determinant of a matrix
    """

    U = cholesky(matrix)
    logdet = 2 * np.sum(log(np.diag(U)))
    return logdet


def mvt_gamma_ln(n, alpha):
    """ Returns the log of multivariate gamma(n, alpha) value.
    necessary for avoiding underflow/overflow problems
    alpha > (n-1)/2
    """

    logp = (((n * (n - 1)) / 4) * log(pi) +
            sum(gammaln(alpha + 0.5 * np.arange(0, -n, -1))))
    return logp


def normalize(vector):
    """Normalizes a vector to have sum one
    """

    return np.array(vector) / sum(vector)


def normalize_logspace(a):
    """Normalize the vector `a` in logspace """
    L = logsumexp(a)
    return a - L


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
