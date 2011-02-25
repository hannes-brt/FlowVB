from os.path import join
from flowvb import FlowVBAnalysis
from scipy.io import loadmat
import numpy as np


TEST_DATA_LOC = join('../', 'tests', 'data', 'old_faithful')

np.random.seed(0)

data = loadmat(join(TEST_DATA_LOC, 'faithful.mat'))['data']
model = FlowVBAnalysis(data,
               num_comp_init=6,
               thresh=1e-5,
               max_iter=200,
               verbose=False,
               plot_monitor=True)
