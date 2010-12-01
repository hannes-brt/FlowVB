import numpy as np
from math import log, exp
cimport numpy as np


def normalize_logspace(mat):
    """ Normalizes the rows of a matrix while avoiding numerical underflow
    """
    L = logsumexp(mat)
    d = mat.shape[1]
    return np.exp(mat - np.tile(L, (d, 1)).T)


def logsumexp(np.ndarray[np.float64_t, ndim=2] mat):
    """ Returns log(sum(exp(a))) while avoiding numerical underflow

    """
    cdef Py_ssize_t i, j
    cdef double sum_exp
    cdef np.ndarray[np.float64_t] max_dim, s
    cdef Py_ssize_t n, d

    n = mat.shape[0]
    d = mat.shape[1]

    for i in range(n):                            # Maximum of each row
        max_dim[i] = -np.inf
        for j in range(d):
            if mat[i, j] > max_dim[i]:
                max_dim[i] = mat[i, j]

    for i in range(n):                            
        sum_exp = 0
        for j in range(d):
            sum_exp += exp(mat[i, j] - max_dim[i])

        s[i] = max_dim[i] + log(sum_exp)
            

    idx_inf = np.nonzero(~ np.isfinite(s))         # Deal with inf entries

    if (len(idx_inf) != 0):
        s[idx_inf] = max_dim[idx_inf]

    return s
    
