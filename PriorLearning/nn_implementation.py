# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:25:56 2023

@author: flobo
"""
    
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tfd = tfp.distributions
import MakeMyPrior
from tensorflow import keras
from MakeMyPrior.MakeMyPrior.discrepancy_measures import energy_loss


# the neural network
def activation_fun(x): 
    return tf.math.maximum(0,x)

def permutation(input_dim, coupling_layers):
    permutation_vec = []
    for _ in range(coupling_layers):
        permutation_vec.append(np.random.permutation(input_dim))
    
    return permutation_vec

class NeuralNetwork(tf.Module):
    def __init__(self, input_dims, output_dims, hidden_layers, units,  train_var_names):
        super(NeuralNetwork, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.units = units
        self.hidden_layers = hidden_layers

        self.w0 = tf.Variable(initial_value = tf.zeros(shape=(self.input_dims, self.units))+0.001,
                          trainable=True, name = f"{train_var_names}_weights_input")
        self.w =  [tf.Variable(initial_value = tf.zeros(shape=(self.units, self.units))+0.001,
                         trainable=True, name = f"{train_var_names}_weights_{i}") for i in range(self.hidden_layers)]
        self.w_out = tf.Variable(initial_value = tf.zeros(shape=(self.units, self.output_dims))+0.001,
                          trainable=True, name = f"{train_var_names}_weights_out")
        
        self.b0 = tf.Variable(initial_value = tf.zeros(shape=(1,self.units))+0.001,
                          trainable=True, name = f"{train_var_names}_bias_input")
        self.b = [tf.Variable(initial_value = tf.zeros(shape=(1, self.units))+0.001,
                          trainable=True, name = f"{train_var_names}_bias_{i}") for i in range(self.hidden_layers)]
        self.b_out = tf.Variable(initial_value = tf.zeros(shape=(1, self.output_dims))+0.001,
                          trainable=True, name = f"{train_var_names}_bias_out")
    
    def __call__(self, z):
        z_t = z #tf.transpose(z, perm=(1,0))
   
        h = activation_fun(tf.matmul(z_t, self.w0) + self.b0)
   
        for i in range(self.hidden_layers):
            h = activation_fun(tf.matmul(h,self.w[i]) + self.b[i])
        o = activation_fun(tf.matmul(h, self.w_out) + self.b_out )
        
        return o#tf.transpose(o, perm = (1,0))

class CouplingBuildingBlock(tf.Module):
    def __init__(self, input_dims, output_dims, hidden_layers,  units, permutation_vec):
        
        super(CouplingBuildingBlock, self).__init__() 
        self.input_dims = input_dims
        self.output_dims = output_dims
        
        self.s1 = NeuralNetwork(self.input_dims[0],self.output_dims[0], 
                                hidden_layers, units, train_var_names = "s1")
        self.t1 = NeuralNetwork(self.input_dims[0],self.output_dims[0],
                                hidden_layers, units, train_var_names = "t1")
        self.s2 = NeuralNetwork(self.input_dims[1],self.output_dims[1],
                                hidden_layers, units, train_var_names = "s2")
        self.t2 = NeuralNetwork(self.input_dims[1],self.output_dims[1],
                                hidden_layers, units, train_var_names = "t2")
        
        self.permutation_vec = permutation_vec
    
    def __call__(self, x, it):
        
        x_perm = tf.gather(x, self.permutation_vec[it], axis = -1)
        
        # compute size of dimensions per split
        split1 = x_perm.shape[-1]//2       # floor division
        split2 = x_perm.shape[-1]-split1   # remaining dimensions
        
        # split latent variable
        z1, z2 = tf.split(x_perm, [split1, split2], axis = -1)
        u1 = z1 * tf.math.exp(self.s1(z2)) + self.t1(z2)
        u2 = z2 * tf.math.exp(self.s2(u1)) + self.t2(u1)
        
        u = tf.concat([u1,u2], axis = -1)
        
        return u

def expert_model(mu0, sigma0,mu1, sigma1, lambda0, B):
    beta0 = tfd.Normal(mu0, sigma0).sample((B,rep))
    beta1 = tfd.Normal(mu1, sigma1).sample((B,rep))
    sigma = tfd.Exponential(lambda0).sample((B,rep))
    
    return beta0, beta1, sigma                                

class CouplingLayers(tf.Module):
    def __init__(self, hidden_layers, units, coupling_layers, permutation_vec):
        super(CouplingLayers, self).__init__()

        self.permutation_vec = permutation_vec
        self.coupling_layers = coupling_layers
        self.ACB = [CouplingBuildingBlock((2,1),(1,2),hidden_layers, units,
                                          self.permutation_vec) for _ in 
                    range(self.coupling_layers)]
    
    def __call__(self, z):
        it = 0
        # first coupling layer
        out = self.ACB[0](z, it)
        
        if self.coupling_layers > 1:
            # remaining coupling layers
            for building_block in self.ACB[1::]:
                it += 1
                out = building_block(out, it)    
        
        return out


# set general variables
epochs = 4#00            # number of epochs
num_param = 3           # number of parameters
hidden_layers = 4       # number of hidden layers in s,t networks
B = 2**7                # batch size
rep = 50
units = 50             # units per layer in s,t networks 
coupling_layers = 7     # number of coupling blocks 

# simulate ideal expert
expert_data = expert_model(0.8, 0.5, 1.5, 0.8, 1., B)   

# sample from multivariate standard normal (base distribution)
z = tfd.MultivariateNormalDiag(loc = tf.zeros(num_param), 
                               scale_diag = tf.ones(num_param)).sample((B,rep))

# initialize permutation vector
permutation_vec = permutation(z.shape[-1], coupling_layers)

# initialize coupling blocks
ACBs = CouplingLayers(hidden_layers, units, coupling_layers, permutation_vec)

# initialize mmd loss with energy kernel
loss = energy_loss()

# initialize adam optimizer
optimizer = keras.optimizers.legacy.Adam(learning_rate = 0.0001,
                                         clipnorm = 1.0)

# run training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # run normalizing flow (forward path)
        out = ACBs(z)
        # allocate dimensions to parameters
        param0 = out[:,:,0]
        param1 = out[:,:,1]
        param2 = out[:,:,2]
        
        # compute loss value
        mmd = loss(expert_data[0], param0, B, rep, rep)
        mmd += loss(expert_data[1], param1, B, rep, rep)
        mmd += loss(expert_data[2], param2, B, rep, rep)
        
        # compute gradients
        g = tape.gradient(mmd, ACBs.trainable_variables)
        # update trainable_variables using gradient info with adam optimizer
        optimizer.apply_gradients(zip(g, ACBs.trainable_variables))
        print(f"epoch: {epoch}, mmd: {mmd:.2f}, b0: {tf.reduce_mean(param0):.2f}, b1: {tf.reduce_mean(param1):.2f}, sigma: {tf.reduce_mean(param2):.2f}")
    
tf.math.reduce_std(param0[0,:])
tf.math.reduce_std(param1[0,:])
tf.math.reduce_std(param2[0,:])

tf.math.reduce_std(z[0,:,0])
tf.math.reduce_std(z[0,:,1])
tf.math.reduce_std(z[0,:,2])

tf.math.reduce_mean(expert_data[0])
