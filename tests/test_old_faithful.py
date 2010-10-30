import unittest
import numpy as np
from scipy.io import loadmat
from os.path import join
from flowvb.utils import arrays_almost_equal

TEST_ACCURACY = 3
MAX_DIFF = pow(10, -TEST_ACCURACY)
TEST_DATA_LOC = join('tests', 'data', 'old_faithful')


def makeTestFaithful(data_filename, function, argument_keys, result_key,
                     load_data=False, max_diff=MAX_DIFF,
                     test_data_loc=TEST_DATA_LOC, doc=''):
    def testFaithful(self):
        test_data = loadmat(join(test_data_loc,
                                 data_filename), squeeze_me=True)

        args = (test_data[arg] for arg in argument_keys)

        if load_data:
            data = np.genfromtxt(join(test_data_loc,
                                      "faithful.txt"), delimiter=",")
            args = (data, ) + tuple(args)

        test_result = function(*args)
        approx_equal = arrays_almost_equal(test_data[result_key],
                                           test_result,
                                           accuracy=max_diff)
        self.assertTrue(approx_equal)

    docstring_function = " Test `" + function.__name__ + \
                "` with some data from Old Faithful"
    testFaithful.__doc__ = docstring_function

    docstring_class = "Test `" + function.__name__ + "`"
    clsdict = {'testFaithful': testFaithful, '__doc__': docstring_class}
    return type('TestFaithful', (unittest.TestCase,), clsdict)
