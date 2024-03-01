# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 13:54:27 2023

@author: flobo
"""

import bayesflow as bf
import numpy as np
import math
import tensorflow as tf

bins = 16
B = 20
dim_out = 2
X = np.random.normal(size = (B,4))
units = 5
default_domain = (-5,5)
num_coupling_blocks = 7

spline_params_counts = {
            "left_edge": 1,
            "bottom_edge": 1,
            "widths": bins,
            "heights": bins,
            "derivatives": bins - 1,
        }

inference_net = bf.networks.InvertibleNetwork(
    num_params=dim_out, coupling_design="spline", num_coupling_layers=1
)

inference_net(X, condition = None)

u1 = tf.cast(X[:,:2], tf.float32)
u2 = tf.cast(X[:,2:], tf.float32)

x1 = np.copy(u1)
x2 = np.copy(u2)

splines = list()
for i in range(num_coupling_blocks):
    sp = spline(u1, u2)
    splines.append(sp)
    v1, v2 = sp()
    u1, u2 = sp()

plt.scatter(u1, u2, color = "red")
plt.scatter(x1, x2, color = "blue")



class spline():
    def __init__(self, u1, u2):
        self.u1 = u1
        self.u2 = u2
        
    def __call__(self):
        s1 = NeuralNetwork(tuple(self.u1.shape), sum(spline_params_counts.values()) * dim_out, 
                           dim_out, units, train_var_names = "nn")
        
        params = s1(self.u1)
        sh = params.shape
        params2 = tf.reshape(params, (sh[0], dim_out, -1))
        parameters = tf.split(params2, list(spline_params_counts.values()), axis=-1)
        
        left_edge, bottom_edge, widths, heights, derivatives = parameters
        
        # domain [-B,B]
        left_edge = left_edge + default_domain[0]
        bottom_edge = bottom_edge + default_domain[0]
        # theta_iw
        default_width = (default_domain[1] - default_domain[0]) / bins
        # theta_ih
        default_height = (default_domain[1] - default_domain[0]) / bins
        
        # Compute shifts for softplus function
        # softplus function: f(x) = log(1 + exp(x)) 
        xshift = tf.math.log(tf.math.exp(default_width) - 1)
        yshift = tf.math.log(tf.math.exp(default_height) - 1)
        
        widths = tf.math.softplus(widths + xshift)
        heights = tf.math.softplus(heights + yshift)
        
        # Compute spline derivatives
        shift = tf.math.log(math.e - 1.0)
        derivatives = tf.nn.softplus(derivatives + shift)
        
        # Add in edge derivatives
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)
        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        scale = total_height / total_width
        derivatives = tf.concat([scale, derivatives, scale], axis=-1)
    
        #%%
        target = self.u2
        
        result = tf.zeros_like(target)
        log_jac = tf.zeros_like(target)
        
        total_width = tf.reduce_sum(widths, axis=-1, keepdims=True)
        total_height = tf.reduce_sum(heights, axis=-1, keepdims=True)
        
        knots_x = tf.concat([left_edge, left_edge + tf.math.cumsum(widths, axis=-1)], axis=-1)
        knots_y = tf.concat([bottom_edge, bottom_edge + tf.math.cumsum(heights, axis=-1)], axis=-1)
        
        target_in_domain = tf.logical_and(knots_x[..., 0] < target, target <= knots_x[..., -1])
        higher_indices = tf.searchsorted(knots_x, target[..., None])
        
        target_in = target[target_in_domain]
        target_in_idx = tf.where(target_in_domain)
        target_out = target[~target_in_domain]
        target_out_idx = tf.where(~target_in_domain)
        
        if tf.size(target_in_idx) > 0:
            # Index crunching
            higher_indices = tf.gather_nd(higher_indices, target_in_idx)
            higher_indices = tf.cast(higher_indices, tf.int32)
            lower_indices = higher_indices - 1
            lower_idx_tuples = tf.concat([tf.cast(target_in_idx, tf.int32), lower_indices], axis=-1)
            higher_idx_tuples = tf.concat([tf.cast(target_in_idx, tf.int32), higher_indices], axis=-1)
        
            # Spline computation
            dk = tf.gather_nd(derivatives, lower_idx_tuples)
            dkp = tf.gather_nd(derivatives, higher_idx_tuples)
            xk = tf.gather_nd(knots_x, lower_idx_tuples)
            xkp = tf.gather_nd(knots_x, higher_idx_tuples)
            yk = tf.gather_nd(knots_y, lower_idx_tuples)
            ykp = tf.gather_nd(knots_y, higher_idx_tuples)
            x = target_in
            dx = xkp - xk
            dy = ykp - yk
            sk = dy / dx
            xi = (x - xk) / dx
        
        numerator = dy * (sk * xi**2 + dk * xi * (1 - xi))
        denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
        result_in = yk + numerator / denominator
        # Log Jacobian for in-domain
        numerator = sk**2 * (dkp * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
        denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
        log_jac_in = tf.math.log(numerator + 1e-10) - tf.math.log(denominator + 1e-10)
        log_jac = tf.tensor_scatter_nd_update(log_jac, target_in_idx, log_jac_in)
        
        result = tf.tensor_scatter_nd_update(result, target_in_idx, result_in)
        
        v1, v2 = self.u1, result
        
        return v1, v2



import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(constrained_layout = True, figsize = (7,7))
axs.scatter(x=knots_x[0,0,:],y=knots_y[0,0,:], color = "purple", label = "knots")
sns.lineplot(x=x[1::2], y=result_in[1::2], ax = axs, color = "red", label = "spline1")
sns.lineplot(x=x[0::2], y=result_in[0::2], ax = axs, color = "green", label = "spline2")
axs.axvline(x=np.unique(knots_x[0,0,:])[0], linestyle = "--", color = "darkgrey", alpha = 0.3, label = "bins")
[axs.axvline(x=np.unique(knots_x[0,0,:])[i], linestyle = "--", color = "darkgrey", alpha = 0.3) for i in range(1,len(np.unique(knots_x[0,0,:])))]
[axs.axhline(y=np.unique(knots_y[0,0,:])[i], linestyle = "--", color = "darkgrey", alpha = 0.3) for i in range(len(np.unique(knots_y[0,0,:])))]
axs.scatter(x[1::2], result_in[1::2], alpha = 0.3, color = "red")
axs.scatter(x[0::2], result_in[0::2], alpha = 0.3, color = "green")
axs.set_xlim(default_domain[0], default_domain[1])
axs.set_xticks(np.arange(default_domain[0],default_domain[1]+0.3,0.5))
axs.set_yticks(np.arange(default_domain[0],default_domain[1]+0.3,0.5))
plt.legend()
plt.show()


c = lambda y: y+5

c(4)
