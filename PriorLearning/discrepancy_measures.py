"""
File: discrepancy_measures.py
Author: Florence Bockting
Date: 09.2023

Description: Discrepancy measures for evaluating the discrepancy between the 
expert and simulated elicited statistics
"""
import tensorflow as tf
from functools import partial
from keras import losses


class squared_error_loss:
    def __init__(self):
        super(squared_error_loss, self).__init__()   
    
    def __call__(self, x, y, B, M, N, **kwargs):
        mean_x = x #tf.reduce_mean(x, 0)
        mean_y = y #tf.reduce_mean(y, 0)
        
        loss = self.squared_error(mean_x, mean_y)
        return tf.reduce_mean(loss)
    
    def squared_error(self, x, y):
        L = tf.square(x) - 2*x*y + tf.square(y)
        return L


def generate_weights(x, N):
    B, N, _ = x.shape
    return tf.divide(tf.ones(shape = (B, N), dtype = x.dtype), N)  

# (x*x - 2xy + y*y)
def squared_distances(x, y):
    # (B, N, 1)
    D_xx = tf.expand_dims(tf.math.reduce_sum(tf.multiply(x, x), axis = -1), axis = 2)
    # (B, 1, M)
    D_yy = tf.expand_dims(tf.math.reduce_sum(tf.multiply(y, y), axis = -1), axis = 1)
    # (B, N, M)
    D_xy = tf.matmul(x, tf.transpose(y, perm = (0, 2, 1)))
    return D_xx - 2*D_xy + D_yy

# sqrt[(x*x - 2xy + y*y)]
def distances(x, y):
    return tf.math.sqrt(tf.clip_by_value(squared_distances(x,y), 
                                         clip_value_min = 1e-8, 
                                         clip_value_max =  int(1e10)))   

# - ||x-y|| = - sqrt[(x*x - 2xy + y*y)]
def energy_kernel(x, y):
    """ Implements kernel norms between sampled measures.
    
    .. math::
        \\text{Loss}(\\alpha,\\beta) 
            ~&=~ \\text{Loss}\\big( \sum_{i=1}^N \\alpha_i \,\delta_{x_i} \,,\, \sum_{j=1}^M \\beta_j \,\delta_{y_j} \\big) 
            ~=~ \\tfrac{1}{2} \|\\alpha-\\beta\|_k^2 \\\\
            &=~ \\tfrac{1}{2} \langle \\alpha-\\beta \,,\, k\star (\\alpha - \\beta) \\rangle \\\\
            &=~ \\tfrac{1}{2} \sum_{i=1}^N \sum_{j=1}^N  \\alpha_i \\alpha_j \cdot k(x_i,x_j) 
              + \\tfrac{1}{2} \sum_{i=1}^M \sum_{j=1}^M  \\beta_i \\beta_j \cdot k(y_i,y_j) \\\\
            &-~\sum_{i=1}^N \sum_{j=1}^M  \\alpha_i \\beta_j \cdot k(x_i,y_j)
    where:
    .. math::
        k(x,y)~=~\\begin{cases}
            \exp( -\|x-y\|^2/2\sigma^2) & \\text{if loss = ``gaussian''} \\\\
            \exp( -\|x-y\|/\sigma) & \\text{if loss = ``laplacian''} \\\\
            -\|x-y\| & \\text{if loss = ``energy''} \\\\
        \\end{cases}
    """
    return -distances(x, y)    

## helper function
def scal(a, f):
    B = a.shape[0]
    return tf.math.reduce_sum(tf.reshape(a, (B, -1)) * tf.reshape(f, (B, -1)), axis = 1)
  
# k(x,y)=0.5*sum_i sum_j a_i a_j k(x_i,x_j) - sum_i sum_j a_i b_j k(x_i,y_j) + 0.5*sum_i sum_j b_i b_j k(y_i, y_j)
def kernel_loss(x, y, a, b):
    B, N, _ = x.shape
    
    K_xx = energy_kernel(x,x) # (B,N,N)
    K_yy = energy_kernel(y,y) # (B,M,M)
    K_xy = energy_kernel(x,y) # (B,N,M)
    
    # (B,N)
    a_x = tf.squeeze(tf.matmul(K_xx, tf.expand_dims(a, axis = -1)))
    # (B,M)
    b_y = tf.squeeze(tf.matmul(K_yy, tf.expand_dims(b, axis = -1)))
    # (B,N)   
    b_x = tf.squeeze(tf.matmul(K_xy, tf.expand_dims(b, axis = -1)))
    
    K = 0.5*scal(a, a_x) + 0.5*scal(b, b_y) - scal(a, b_x)
    
    return tf.reduce_mean(K)

class energy_loss:
    """Class used to compute the MMD loss using an energy kernel"""
    def __init__(self):
        super(energy_loss, self).__init__()   
    
    def __call__(self, x, y, B, M, N, **kwargs):
        """
        Parameters
        ----------
        x : tensor, shape = (B,M)
            expert sample distribution
        y : tensor, shape = (B,N)
            model sample distribution
        B : int
            batch size
        M : int
            number of dimensions
        N : int
            number of dimensions
        """
        x = tf.expand_dims(tf.reshape(x, (B, M)), -1)
        y = tf.expand_dims(tf.reshape(y, (B, N)), -1)
        
        a = generate_weights(x, M)
        b = generate_weights(y, N)   
        
        return kernel_loss(x, y, a, b)

    
    
class gaussian_loss: 
    "Input has form (B, N, D)"
    def __init__(self):
        super(gaussian_loss, self).__init__()
        
    def __call__(self, x, y, B, N, M):
        x = tf.expand_dims(tf.reshape(x, (B, M)), -1)
        y = tf.expand_dims(tf.reshape(y, (B, N)), -1)
                 
        return(self.maximum_mean_discrepancy(x, y))

    # (x*x - 2xy + y*y)
    def squared_distances(self, x, y):
        # (B, N, 1)
        D_xx = tf.expand_dims(tf.math.reduce_sum(tf.multiply(x, x), axis = -1), axis = 2)
        # (B, 1, M)
        D_yy = tf.expand_dims(tf.math.reduce_sum(tf.multiply(y, y), axis = -1), axis = 1)
        # (B, N, M)
        D_xy = tf.matmul(x, tf.transpose(y, perm = (0, 2, 1)))
        return D_xx - 2*D_xy + D_yy
    
    def gaussian_kernel_matrix(self, x, y, sigmas):
        # b = 1/2*sigma
        beta = 1. / (2. * (tf.expand_dims(sigmas, axis=1)))
        dist = self.squared_distances(x,y)
        # 1/(2*sigma) * || x-y ||2
        s = tf.matmul(beta, tf.reshape(dist, (1,-1)))
        # exp(- 1/(2*sigma) * || x-y ||2 )
        return tf.reshape(tf.reduce_sum(tf.exp(-s), axis=0), tf.shape(dist))
    
    def mmd_kernel(self, x, y, kernel=gaussian_kernel_matrix):
        # || x-y ||2 = xTx -2*xTy + yTy
        loss = tf.reduce_mean(input_tensor = kernel(x, x), axis=(1,2))
        loss += tf.reduce_mean(input_tensor = kernel(y, y), axis=(1,2))
        loss -= 2 * tf.reduce_mean(input_tensor = kernel(x, y), axis=(1,2))
        return loss
      
    def maximum_mean_discrepancy(self,x, y, weight=1., minimum=0., **args):
        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(self.gaussian_kernel_matrix, sigmas=sigmas)
        loss_value = self.mmd_kernel(x, y, kernel=gaussian_kernel)
        loss_value = tf.maximum(minimum, loss_value) * weight
        return tf.reduce_mean(loss_value)