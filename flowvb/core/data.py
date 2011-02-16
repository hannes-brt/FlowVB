'''
Created on 2011-02-15

@author: Andrew Roth
'''
import numpy as np

class Data( object ):
    '''
    Class for representing flow cytometry data.
    '''
    def __init__( self, X, pre_processor=None ):
        '''
        X - A numpy array with each row being an observation. 
        '''
        if pre_processor is None:
            self.data = X
        else:
            self.data = pre_processor.pre_process( X )
            
        self.nrows = self.data.shape[0]
        self.ncols = self.data.shape[1]
    
    def pre_process( self, pre_processor ):
        pass
    
class PreProcessor:
    '''
    Class for pre-processing data.
    '''
    def pre_process( self, X ):
        '''
        All subclasses should implement this method. 
        Input is a design matrix X, and output is the processed matrix.
        '''
        raise NotImplemented
    
class Whitener( PreProcessor ):
    def pre_proccess( self, X ):
        """ Normalize a group of observations on a per feature basis.
    
        Before running k-means, it is beneficial to rescale each feature
        dimension of the observation set with whitening. Each feature is
        divided by its standard deviation across all observations to give
        it unit variance.
    
        :Parameters:
            obs : ndarray
                Each row of the array is an observation.  The
                columns are the features seen during each observation.
                ::
    
                          #   f0    f1    f2
                    obs = [[  1.,   1.,   1.],  #o0
                           [  2.,   2.,   2.],  #o1
                           [  3.,   3.,   3.],  #o2
                           [  4.,   4.,   4.]]) #o3
    
                XXX perhaps should have an axis variable here.
    
        :Returns:
            result : ndarray
                Contains the values in obs scaled by the standard devation
                of each column.
    
        Examples
        --------
    
        >>> from numpy import array
        >>> from scipy.cluster.vq import whiten
        >>> features  = array([[  1.9,2.3,1.7],
        ...                    [  1.5,2.5,2.2],
        ...                    [  0.8,0.6,1.7,]])
        >>> whiten(features)
        array([[ 3.41250074,  2.20300046,  5.88897275],
               [ 2.69407953,  2.39456571,  7.62102355],
               [ 1.43684242,  0.57469577,  5.88897275]])
    
        """
        
        std_dev = np.std( X, axis=0 )
        return X / std_dev
