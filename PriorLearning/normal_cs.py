# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:21:49 2024

@author: flobo
"""

import patsy as pa
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from MakeMyPrior.elicitation_wrapper import expert_model
from MakeMyPrior.training import trainer
from MakeMyPrior.helper_functions import group_obs, Exponential_unconstrained, Normal_unconstrained
from MakeMyPrior.user_config import target_config, target_input
from MakeMyPrior.plot_helpers import plot_loss, plot_convergence, plot_priors
from MakeMyPrior.discrepancy_measures import energy_loss, squared_error_loss

def sim_case_normal(loss_list, epochs, rep):
   # tf.random.set_seed(123)
    
    user_config = dict(                    
        B = 2**8,                          
        rep = rep,                         
        epochs = epochs,                      
        view_ep = 1,
        lr_decay = True,
        lr0 = 0.001, 
        lr_min = 0.0001, 
        loss_dimensions = "B:m,n",   
        loss_discrepancy = loss_list,       
        loss_scaling = "unscaled",         
        method = "normalizing_flow"  
        )
    
    # design matrix
    X =  pa.dmatrix("a*b", pa.balanced(a = 2, b = 3, repeat = 60))
    dmatrix = tf.cast(X, dtype = tf.float32)
    # contrast matrix
    cmatrix = dmatrix[0:dmatrix.shape[1], :]
    
    
    # true hyperparameter values for ideal_expert
    true_mu = [0.12, 0.15, -0.02, -0.03, -0.02, -0.04]
    true_sigma = [0.02, 0.02, 0.06, 0.06, 0.03, 0.03]
    true_nu = 9.
    
    # model parameters
    parameters_dict = dict()
    for i in range(6):
        parameters_dict[f"beta_{i}"] = {
                "family":  Normal_unconstrained(),
                "true": tfd.Normal(true_mu[i], true_sigma[i]),
                "initialization": [tfd.Normal(0.,0.1)]*2
                }
    parameters_dict["sigma"] = {
            "family": Exponential_unconstrained(user_config["rep"]),
            "true": tfd.Exponential(true_nu),
            "initialization": [tfd.Normal(0.,0.1)]
            }
    
    # generative model
    class GenerativeModel(tf.Module):
        def __call__(self, 
                     parameters, # obligatory: samples from prior distributions; tf.Tensor
                     dmatrix,    # optional: design matrix; tf.Tensor
                     cmatrix,    # optional: contrast matrix; tf.Tensor
                     **kwargs    # obligatory: possibility for further keyword arguments is needed 
                     ):  
            
            # compute linear predictor term
            epred = parameters[:,:,0:6] @ tf.transpose(dmatrix)
            
            # define likelihood
            likelihood = tfd.Normal(
                loc = epred, 
                scale = tf.expand_dims(parameters[:,:,-1], -1))
            
            # sample prior predictive data
            ypred = likelihood.sample()
            
            # compute custom target quantity (here: group-differences)
            samples_grouped = group_obs(ypred, dmatrix, cmatrix)
    
            # compute mean difference between groups
            effect_list = []
            diffs = [(3,0), (4,1), (5,2)]
            for i in range(len(diffs)):
                # compute group difference
                diff = tf.math.subtract(
                    samples_grouped[:, :, :, diffs[i][0]],
                    samples_grouped[:, :, :, diffs[i][1]],
                )
                # average over individual obs within each group
                diff_mean = tf.reduce_mean(diff, axis=2)
                # collect all mean group differences
                effect_list.append(diff_mean)
    
            mean_effects = tf.stack(effect_list, axis=-1)
            
            return dict(likelihood = likelihood,     # obligatory: likelihood; callable
                        ypred = ypred,               # obligatory: prior predictive data
                        epred = epred,               # obligatory: samples from linear predictor
                        mean_effects = mean_effects  # optional: custom target quantity
                        )
        
    # define a custom function using the output from the generative model   
    def custom_r2(ypred, epred, **kwargs):
        return tf.math.divide(tf.math.reduce_variance(epred, axis = -1), 
                              tf.math.reduce_variance(ypred, axis = -1))
    
    # specify target quantity, elicitation technique and loss combination
    t1 = target_config(target="R2", 
                       elicitation="histogram",
                       combine_loss="all",
                       custom_target_function = custom_r2)
    t2 = target_config(target="group_means", 
                       elicitation="quantiles", 
                       combine_loss="by-group", 
                       quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90))
    t3 = target_config(target="mean_effects", 
                       elicitation="quantiles",
                       combine_loss="by-group",
                       quantiles_specs = (10, 20, 30, 40, 50, 60, 70, 80, 90))
    
    target_info = target_input(t1, t2, t3)
    
    # ideal expert
    expert_res_list, prior_pred_res = expert_model(1, user_config["rep"],
                                   parameters_dict, GenerativeModel, target_info,
                                   method = "ideal_expert",
                                   dmatrix = dmatrix,
                                   cmatrix = cmatrix,
                                   dmatrix_fct = dmatrix)
                                   
    # simulation model and training
    res_dict = trainer(expert_res_list, user_config["B"], user_config["rep"],
                       parameters_dict, user_config["method"], GenerativeModel,
                       target_info, user_config, loss_balancing = True,
                       dmatrix = dmatrix, cmatrix = cmatrix, dmatrix_fct = dmatrix)
    
    return expert_res_list, prior_pred_res, res_dict, target_info, user_config

