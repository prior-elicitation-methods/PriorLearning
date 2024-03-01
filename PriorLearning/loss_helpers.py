# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:30:07 2023

@author: flobo
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from MakeMyPrior.helper_functions import scale_loss_component
from MakeMyPrior.discrepancy_measures import energy_loss

def test_value(samples):
    if len(samples.shape) > 2:
        test_val = tf.multiply(samples.shape[1], samples.shape[2]).numpy()
    elif len(samples.shape) == 2:
        test_val = samples.shape[1]
    else:
        print(f"Shape of tensor is {samples.shape}. Don't know how to handle this.")
    
    return test_val

def compute_loss(expert_res_dict, model_res_dict, 
                 loss_scaling, loss_dimensions, loss_discrepancy, num_loss_comp):
   
    loss = []
    expert_components = []
    model_components = []
    
    if loss_discrepancy == "energy":
        loss_discrepancy_list = [energy_loss()]*num_loss_comp
       
    for i in range(len(model_res_dict)):
      
        loss_measure = loss_discrepancy_list[i]   
        m = model_res_dict[i]        
        e_raw = expert_res_dict[i]
        if len(m.shape) == 1:
            m = tf.expand_dims(m, axis=-1)
        if len(e_raw.shape) == 1:
            e_raw = tf.expand_dims(e_raw, axis=-1)
        
        # broadcast such that first dimension is batch size
        e = tf.broadcast_to(e_raw, shape = (m.shape[0], e_raw.shape[1]))
        
        # apply scaling of elicited statistics of loss component
        if e.shape[1] != 1:
            e, m = scale_loss_component(loss_scaling, e, m)
       
        if loss_dimensions == "m,n:B":
            e = tf.transpose(e)
            m = tf.transpose(m)
            
        loss.append(loss_measure(
                        e, m, 
                        m.shape[0],         # B
                        e.shape[1],         # m
                        m.shape[1])         # n
                        )
        expert_components.append(e)
        model_components.append(m)
    
    return loss, (expert_components,model_components)

            

   


