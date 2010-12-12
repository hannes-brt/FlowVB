import numpy as np
cimport numpy as np
import pudb

cdef extern from "c-utils.h":
     void normalize_logspace_matrix(Py_ssize_t nrow , Py_ssize_t ncol,
                                    Py_ssize_t rowstride, Py_ssize_t colstride,
                                    double* mat)

def normalize_logspace(np.ndarray[np.double_t, ndim=2] mat):
    cdef Py_ssize_t n, d
    n = mat.shape[0]
    d = mat.shape[1]
    rowstride = mat.strides[0] // mat.itemsize
    colstride = mat.strides[1] // mat.itemsize
    normalize_logspace_matrix(n, d, rowstride, colstride, <double*> mat.data)


    
