from os.path import join

import numpy as np
from scipy.io import loadmat

from flowvb import FlowVBAnalysis
from flowvb.core.flow_vb import Options
from flowvb.initialize import D2Initialiser

TEST_DATA_LOC = join('../', 'tests', 'data', 'old_faithful')

np.random.seed(0)

data = loadmat(join(TEST_DATA_LOC, 'faithful.mat'))['data']

num_comp_init = 6
init_params = D2Initialiser().initialise_parameters(data, num_comp_init)

# Use default options but turn plot monitor on.
options = Options(init_params)
options.plot_monitor = True

model = FlowVBAnalysis(data, options)
