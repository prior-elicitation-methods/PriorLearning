# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 21:03:57 2023

@author: flobo
"""

import tensorflow_probability as tfp
tfd = tfp.distributions

from MakeMyPrior.discrepancy_measures import energy_loss

_global_variables = dict(
    method = "softmax_gumbel_trick", 
    softmax_gumbel_temp = 1.0,
    coupling_design = "Spline",
    coupling_layers = 7,
    units = 2**7, 
    activation = "relu",
    permutation = "fixed",
    loss_discrepancy = energy_loss,   #["energy", "gaussian"]     
    lr_step = 5,
    lr_perc = 0.80,
    clipnorm_val = 1.,
    task_balance_factor = 1.6,
    )     