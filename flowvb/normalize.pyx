import numpy as np
cimport numpy as np

cdef extern from "c-utils.h":
     void normalize_logspace_matrix(Py_ssize_t nrow , Py_ssize_t ncol,
                                    Py_ssize_t rowstride, Py_ssize_t colstride,
                                    double* mat, double* mat_out)

def normalize_logspace(np.ndarray[np.double_t, ndim=2] mat):
    """Normalizes the rows of a matrix in logspace to prevent underflow.
    """
    
    cdef Py_ssize_t n, d
    cdef np.ndarray[np.double_t, ndim=2] mat_out
    cdef char *ordering
    
    n = mat.shape[0]
    d = mat.shape[1]

    # Get the strides for the rows and columns
    rowstride = mat.strides[0] // mat.itemsize
    colstride = mat.strides[1] // mat.itemsize

    # Create the output array in the same order as the input
    if mat.flags['C_CONTIGUOUS']:
        ordering = 'C'
    else:
        ordering = 'F'

    mat_out = np.empty((n, d), order=ordering)

    normalize_logspace_matrix(n, d, rowstride, colstride, 
                              <double*> mat.data, <double*> mat_out.data)

    return mat_out
