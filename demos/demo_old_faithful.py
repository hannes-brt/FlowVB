from os.path import join
from flowvb import FlowVBAnalysis
from flowvb.initialize import D2Initialiser
from scipy.io import loadmat
import numpy as np
from argparse import Namespace


TEST_DATA_LOC = join('../', 'tests', 'data', 'old_faithful')

np.random.seed(0)

data = loadmat(join(TEST_DATA_LOC, 'faithful.mat'))['data']

args = Namespace()

args.num_comp_init = 6
args.thresh = 1e-5
args.max_iter = 200
args.verbose = False
        
args.prior_dirichlet = 1e-2
args.dof_init = 2
args.remove_comp_thresh = 1e-6
        
args.use_exact = False
args.whiten_data = False
args.plot_monitor = True

args.init_params = D2Initialiser().initialise_parameters(data, args.num_comp_init)

model = FlowVBAnalysis(data, args)
