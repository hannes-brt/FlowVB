"""Miscalleneous functions
"""

import numpy as np

def normalize(vector):
    """Normalizes a vector to have sum one
    """
    
    return np.array(vector) / sum(vector)
       
