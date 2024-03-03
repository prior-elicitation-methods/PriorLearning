# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:24:36 2024

@author: flobo
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from itertools import product

from MakeMyPrior.loss_helpers import test_value


def combi_none(elicits, target):
    loss_component = []
    test_num_losses = test_value(elicits[target])
    test_shape = (elicits[target].shape[0],)
    
    if tf.rank(elicits[target]) == 2:
        [loss_component.append(elicits[target][:,i]) for i in 
         range(elicits[target].shape[1])]
    
    if tf.rank(elicits[target]) == 3:
        [loss_component.append(elicits[target][:,i,j]) for i,j in 
         product(range(elicits[target].shape[1]),
                 range(elicits[target].shape[2]))]
    
    assert loss_component[0].shape == test_shape, f"Shape of loss component should be {test_shape} but is {loss_component[0].shape}."
    assert len(loss_component) == test_num_losses, f"Number of loss components should be {test_num_losses} but is {len(loss_component)}." 
    
    return loss_component

def combi_all(elicits, target):
   # test_shape = (elicits[target].shape[0], test_value(elicits[target]))
    #loss_component = 0
  
    if len(elicits[target].shape) == 2:
        loss_component = elicits[target]
    
    if len(elicits[target].shape) == 3:
        #print(tf.rank(elicits[target]))
        loss_component = elicits[target][:,:,0]
      
        for i in range(1, elicits[target].shape[-1]):
            loss_component = tf.concat([loss_component,
                elicits[target][:,:,i]
                ], axis = 1)
    
   # assert loss_component[0].shape == test_shape, f"Shape of loss component should be {test_shape} but is {loss_component.shape}."
    
    return loss_component


def combi_group(elicits, target):
    loss_component = []
        
    [loss_component.append(elicits[target][:,:,i]) for i in 
     range(elicits[target].shape[2])]
    
    return loss_component


def combi_stats(elicits, target):
    loss_component = []
    if len(elicits[target].shape) == 3:
        #test_num_losses =  elicits[target].shape[1]
        #test_shape = (elicits[target].shape[0], elicits[target].shape[2])
        
        [loss_component.append(elicits[target][:,i,:]) for i in 
         range(elicits[target].shape[1])]
    
    if len(elicits[target].shape) == 2:
        #test_num_losses =  elicits[target].shape[1]
        #test_shape = (elicits[target].shape[0], )
        
        [loss_component.append(elicits[target][:,i]) for i in 
         range(elicits[target].shape[1])]
    
    #assert loss_component[0].shape == test_shape, f"Shape of loss component should be {test_shape} but is {loss_component[0].shape}."
    #assert len(loss_component) == test_num_losses, f"Number of loss components should be {test_num_losses} but is {len(loss_component)}." 
    return loss_component 

def combine_loss_components(target_info, elicits):
    
    loss_dict = dict()
    for i, (combi, target) in enumerate(zip(target_info["combine_loss"],
                                            elicits.keys())):
    
        assert combi in ["all", "by-group", "by-stats", None], f"Combine_loss must be either 'all', 'by-group', 'by-stats' or None. Got {combi}."
        
        if combi is None:
            loss_dict[target] = combi_none(elicits, target)
            
        if combi == "all":
            loss_dict[target] = combi_all(elicits, target)
            
        if combi == "by-group":
            loss_dict[target] = combi_group(elicits, target)
            
        if combi == "by-stats":
            loss_dict[target] = combi_stats(elicits, target)

    return loss_dict
