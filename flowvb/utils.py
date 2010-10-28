"""Miscalleneous functions
"""

import numpy as np
from scipy.maxentropy import logsumexp


def repeat(x, func, *args, **kargs):
    """Call the function `func` `x` times and accumulate the results in a list
    """
    return [func(*args, **kargs) for _ in range(x)]


def normalize(vector):
    """Normalizes a vector to have sum one
    """

    return np.array(vector) / sum(vector)


def normalize_logspace(a):
    """Normalize the vector `a` in logspace """
    L = logsumexp(a)
    return a - L


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
    d = a - b
    return (d < accuracy).all()
