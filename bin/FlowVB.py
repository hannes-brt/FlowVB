#!/usr/bin/env python

#=======================================================================================================================
# FlowVB
# Author : Hannes Bretschneider, Andrew Roth
#=======================================================================================================================

import argparse
from flowvb.run import run_flow_vb

parser = argparse.ArgumentParser(prog='FlowVB')

#===============================================================================
# Add analysis sub-command
#===============================================================================
parser.add_argument('input_file_name',
                             help='''Path to input file.''')

parser.add_argument('output_prefix',
                             help='''Prefix to be used for output files.''')

parser.add_argument('--file_format',
                              choices=['csv', 'fcs'],
                              default='fcs',
                              help='''Format of input file.''')

train_group = parser.add_argument_group(title='Training Parameters',
                                        description='Options for training the model.')

train_group.add_argument('--max_iter', default=200, type=int,
                          help='''Maximum number of iterations to used for training model. Default 200.''')

train_group.add_argument('--num_comp_init', default=10, type=int,
                          help='''Number of components to initally use in the model. Default 10.''')

train_group.add_argument('--thresh', default=1e-5, type=float,
                          help='''Convergence threshold for VBEM training. Default 1e-6''')

train_group.add_argument('--verbose', action='store_true', default=False,
                          help='''If set the program will output internal information.''')

train_group.add_argument('--init_method', choices=['d2-weighting', 'kmeans', 'random'],
                         default='d2-weighting', help='Model initialisation method. Default is d2-weighting.')

train_group.add_argument('--remove_comp_thresh', default=1e-2, type=float,
                          help='''Remove clusters if their mix-weights fall below this value. Default 1e-2.''')

train_group.add_argument('--whiten_data', action='store_true', default=False,
                          help='''If set the program will whiten the data set. Default False.''')

train_group.add_argument('--plot_monitor', action='store_true', default=False,
                          help='''If set the program will display a window plotting clusters as they are fit. Default
                          False.''')

train_group.add_argument('--use_exact', action='store_true', default=False,
                          help='''If set the will use an exact line search to estimate dof in student-t densities. By
                          default an accurate approximation is used instead.''')

train_group.add_argument('--dof_init', default=2, type=float,
                         help='''Initial dof value for studen-t densities.''')

prior_group = parser.add_argument_group(title='Prior Parameters',
                                        description='Prior values for the model.')

prior_group.add_argument('--prior_dirichlet', default=1e-2, type=float,
                         help='''Value of parameters in dirichlet prior distribution (<1 promotes sparsity). Default
                         1e-2.''')

parser.set_defaults(func=run_flow_vb)
#===============================================================================
# Run
#===============================================================================
args = parser.parse_args()

args.func(args)
