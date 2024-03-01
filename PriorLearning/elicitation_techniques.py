"""
File: elicitation_techniques.py
Author: Florence Bockting
Date: 09.2023

Description: Compute elicited statistics from target quantities according to selected elicitation technique.
"""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def method_quantiles(target_samples, q):
   
    # compute quantiles
    quants = tfp.stats.percentile(x=target_samples, q=q, axis=1)
    # depending on rank of tensor reshape tensor such that quantile axis 
    # is last (not first)
    if len(quants.shape) == 2:
        quants_reshaped = tf.transpose(quants)
    if len(quants.shape) == 3:
        quants_reshaped = tf.transpose(quants, perm=[1, 0, 2])
    
    return quants_reshaped

def method_moments(target_samples, m):
    
    mom_list = []
    for i in m:
        if i == "mean":
            mom_list.append(tf.reduce_mean(target_samples, axis=1))
        if i == "sd":
            mom_list.append(tf.math.reduce_std(target_samples, axis=1))
        if i == "variance":
            mom_list.append(tf.math.reduce_variance(target_samples, axis=1))
    
    return tf.stack(mom_list,1)

class ElicitationTechnique(tf.Module):
    def __init__(self):
        super(ElicitationTechnique).__init__()
    
    def __call__(self, elicit_info, targets):
    
        elicit_dict = dict()
        # compute elicited statistics from target quantities according to
        # specified elicitation technique
        
        if "quantiles_specs" in elicit_info.keys(): 
            quant = 0
        
        if "moments_specs" in elicit_info.keys(): 
            mom = 0
        
        if "obs_specs" in elicit_info.keys():
            obs = 0
       
        for i, (elicit_method, target) in enumerate(zip(elicit_info["elicitation"], 
                                          targets.keys())):
            #print(i, elicit_method, target)
            if elicit_method == "histogram":
                elicit_dict[f"{target}_hist_{i}"] = targets[target]
    
            if elicit_method == "quantiles":
                q = elicit_info["quantiles_specs"][quant]
                quant += 1
                
                elicit_dict[f"{target}_quant_{i}"] = method_quantiles(
                    targets[target], list(q))

            if elicit_method == "moments":
                m = elicit_info["moments_specs"][mom]
                mom += 1
                
                elicit_dict[f"{target}_mom_{i}"] = method_moments(
                    targets[target], list(m))
                
            if elicit_method == "select_obs":
                o = elicit_info["obs_specs"][obs]
                yobs = targets[target]
                elicit_dict[f"{target}_obs_{i}"] = tf.gather(yobs, o, axis = 2)
                
        return elicit_dict
        
    
