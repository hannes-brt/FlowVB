import unittest
import types
import numpy as np
from scipy.io import loadmat
from os.path import join
from flowvb.utils import arrays_almost_equal

TEST_ACCURACY = 3
MAX_DIFF = pow(10, -TEST_ACCURACY)
TEST_DATA_LOC = join('tests', 'data', 'old_faithful')


def makeTestFaithful(mat_filename, test_function, argument_keys, result_key,
                     load_data=False, max_diff=MAX_DIFF,
                     test_data_loc=TEST_DATA_LOC):
    """ Factory function to produce classes that are test cases
    for the _update-methods.

    Parameters
    ----------
    mat_filename : string
        The .mat-file where the data for the test is stored.
    test_function : function
        The function to be tested
    argument_keys : tuple or list
        A tuple or a list of strings that are the names of the arguments to
        `test_function`, as they a are stored in `mat_filename`.
    result_key : strings
        The name of the return of `test_function` as stores in `mat_filename`.
    load_data : bool, optional
        Whether to also load the original data (will be passed as first
        positional argument to `test_function`.
    max_diff : float, optional
        The maximum difference to accept between the test data stored in
        `mat_filename` and the return value of `test_function`.
    test_data_loc : string, optional
        The location where the data is found.

    Return
    ------
    TestCase : class
        A class derived from superclass to use with unittest.
    """

    def testFaithful(self):
        test_data = loadmat(join(test_data_loc,
                                 mat_filename), squeeze_me=True)

        args = (test_data[arg] for arg in argument_keys)

        if load_data:
            data = np.genfromtxt(join(test_data_loc,
                                      "faithful.txt"), delimiter=",")
            args = (data, ) + tuple(args)

        test_result = test_function(*args)
        approx_equal = arrays_almost_equal(test_data[result_key],
                                           test_result,
                                           accuracy=max_diff)
        self.assertTrue(approx_equal)

    docstring_function = " Test `" + test_function.__name__ + \
                "` with some data from Old Faithful"
    testFaithful.__doc__ = docstring_function

    docstring_class = "Test `" + test_function.__name__ + "`"
    clsdict = {'testFaithful': testFaithful,
                '__doc__': docstring_class}
    return type('TestFaithful', (unittest.TestCase,), clsdict)
