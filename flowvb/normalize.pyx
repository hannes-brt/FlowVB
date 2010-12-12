import numpy as np
cimport numpy as np

cdef extern from "c-utils.h":
     void normalize_logspace_matrix_c(Py_ssize_t nrow , Py_ssize_t ncol,
                                    char* mat)

def normalize_logspace(np.ndarray[np.double_t, ndim=2] mat):

    cdef Py_ssize_t n, d
    n = mat.shape[0]
    d = mat.shape[1]
    normalize_logspace_matrix_c(n, d, mat.data)
    return np.exp(mat)
    
