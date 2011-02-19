str_summary_data = """
FlowVB Object

Data:
=====
Observations:     %(num_obs)d
Dimensions:       %(num_features)d
Whiten data:      %(whiten_data)s

"""


str_summary_options_init_all = """
Options:
========

Initialization options:
-----------------------
Initial number of clusters             (num_comp_init)      : %(num_comp_init)d
Initial degrees of freedom             (dof_init)           : %(dof_init)7.5e
Dirichlet prior                        (prior_dirichlet)    : %(prior_dirichlet)7.5e
"""

str_summary_init_mean = """
Initial cluster mean    (init_mean):
%(init_mean)s
"""

str_summary_init_covar = """
Initial covariances     (init_covar):
%(init_covar)s
"""

str_summary_init_mixweights = """
Initial mixweights      (init_mixweights):
%(init_mixweights)s
"""

str_summary_optim_display = """
Optimization options:
---------------------
Threshold for stopping the iteration   (thresh)             : %(thresh)7.5e
Threshold to remove clusters           (remove_comp_thresh) : %(remove_comp_thresh)7.5e
Use approximation for to update dof    (use_approx)         : %(use_approx)7.5e

Display options:
----------------
Verbosity                              (verbose)            : %(verbose)s
Plot monitor                           (plot_monitor)       : %(plot_monitor)s
"""
