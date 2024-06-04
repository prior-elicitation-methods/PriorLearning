import tensorflow as tf
from functions.helper_functions import save_as_pkl

class MmdEnergy(tf.Module): 
    def __call__(self, loss_component_expert, loss_component_training, **kwargs):
        """
        Computes the maximum-mean discrepancy between two samples.

        Parameters
        ----------
        loss_component_expert : tf.Tensor
            loss component based on expert input.
        loss_component_training : tf.Tensor
            loss component based on model simulations.
        **kwargs : optional additional keyword arguments
            optional: additional keyword arguments.

        Returns
        -------
        mmd_loss : float
            maximum mean discrepancy loss using an energy kernel
            
        """
        # get batch size
        B = loss_component_expert.shape[0]
        # dimensionality for expert input and model simulations
        dim_expert = loss_component_expert.shape[-1]
        dim_training = loss_component_training.shape[-1]

        # expert samples and model simulation samples
        x = tf.expand_dims(tf.reshape(loss_component_expert, (B, dim_expert)), -1)
        y = tf.expand_dims(tf.reshape(loss_component_training, (B, dim_training)), -1)
        
        a = self.generate_weights(x, dim_expert)
        b = self.generate_weights(y, dim_training)   
        
        return self.kernel_loss(x, y, a, b, dim_expert)

    def generate_weights(self, loss_component, dim):
        B, dim, _ = loss_component.shape
        weights = tf.divide(tf.ones(shape = (B, dim), dtype = loss_component.dtype), dim) 
        return weights

    # (x*x - 2xy + y*y)
    def squared_distances(self, loss_component_expert, loss_component_training):
        # (B, N, 1)
        distance_expert = tf.expand_dims(tf.math.reduce_sum(tf.multiply(loss_component_expert, loss_component_expert), axis = -1), axis = 2)
        # (B, 1, M)
        distance_training = tf.expand_dims(tf.math.reduce_sum(tf.multiply(loss_component_training, loss_component_training), axis = -1), axis = 1)
        # (B, N, M)
        distance_expert_training = tf.matmul(loss_component_expert, tf.transpose(loss_component_training, perm = (0, 2, 1)))
        # compute sq. distance
        squared_distance = distance_expert - 2*distance_expert_training + distance_training
        return squared_distance

    # -sqrt[(x*x - 2xy + y*y)]
    def distances(self, loss_component_expert, loss_component_training):
        distance = tf.math.sqrt(
            tf.clip_by_value(
                self.squared_distances(loss_component_expert,loss_component_training), 
                clip_value_min = 1e-8, clip_value_max =  int(1e10)
                )
            )   
        # energy distance as negative distance
        energy_distance = - distance
        return energy_distance

    ## helper function
    def scal(self, a, f):
        B = a.shape[0]
        return tf.math.reduce_sum(tf.reshape(a, (B, -1)) * tf.reshape(f, (B, -1)), axis = 1)

    # k(x,y)=0.5*sum_i sum_j a_i a_j k(x_i,x_j) - sum_i sum_j a_i b_j k(x_i,y_j) + 0.5*sum_i sum_j b_i b_j k(y_i, y_j)
    def kernel_loss(self, loss_component_expert, loss_component_training, a, b, dim_expert):        
        K_expert = self.distances(loss_component_expert,loss_component_expert) # (B,N,N)
        K_training = self.distances(loss_component_training,loss_component_training) # (B,M,M)
        K_expert_training = self.distances(loss_component_expert,loss_component_training) # (B,N,M)
        
        # (B,N)
        a_x = tf.squeeze(tf.matmul(K_expert, tf.expand_dims(a, axis = -1)))
        # (B,M)
        b_y = tf.squeeze(tf.matmul(K_training, tf.expand_dims(b, axis = -1)))
        # (B,N)   
        b_x = tf.squeeze(tf.matmul(K_expert_training, tf.expand_dims(b, axis = -1)))
        
        loss1 = 0.5 * self.scal(a, a_x) + 0.5 * self.scal(b, b_y) - self.scal(a, b_x)
        save_as_pkl(loss1, "elicit/simulations/results/data/deep_prior/pois_test/loss1.pkl")
        
        loss2 = 0.5 * a_x + 0.5 * b_y - b_x
        save_as_pkl(loss2, "elicit/simulations/results/data/deep_prior/pois_test/loss2.pkl")
        
        # scale loss
       # scale_loss = tf.divide(loss, tf.sqrt(tf.cast(dim_expert, tf.float32)))
        
        # average over batches
        mean_loss =  tf.reduce_mean(loss1)
        return mean_loss



    
MMD_energy = MmdEnergy()

# class MmdGaussian(tf.Module):  
#     def __init__(self):
#         self.MMD_BANDWIDTH_LIST = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]

#     def __call__(self, loss_component_expert, loss_component_training, **kwargs):
#         return self.maximum_mean_discrepancy(loss_component_training,
#                                              loss_component_expert, 
#                                               kernel="gaussian", 
#                                               mmd_weight=1.0, minimum=0.0)    

#     def gaussian_kernel_matrix(self, x, y, sigmas=None):
#         """Computes a Gaussian radial basis functions (RBFs) between the samples of x and y.

#         We create a sum of multiple Gaussian kernels each having a width :math:`\sigma_i`.

#         Parameters
#         ----------
#         x       :  tf.Tensor of shape (num_draws_x, num_features)
#             Comprises `num_draws_x` Random draws from the "source" distribution `P`.
#         y       :  tf.Tensor of shape (num_draws_y, num_features)
#             Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
#         sigmas  : list(float), optional, default: None
#             List which denotes the widths of each of the gaussians in the kernel.
#             If `sigmas is None`, a default range will be used, contained in ``bayesflow.default_settings.MMD_BANDWIDTH_LIST``

#         Returns
#         -------
#         kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
#             The kernel matrix between pairs from `x` and `y`.
#         """

#         if sigmas is None:
#             sigmas = self.MMD_BANDWIDTH_LIST
#         norm = lambda v: tf.reduce_sum(tf.square(v), 1)
#         beta = 1.0 / (2.0 * (tf.expand_dims(sigmas, 1)))
#         dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
#         s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
#         kernel = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
#         return kernel

#     def mmd_kernel(self, x, y, kernel):
#         """Computes the estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.

#         Maximum Mean Discrepancy (MMD) is a distance-measure between random draws from
#         the distributions `x ~ P` and `y ~ Q`.

#         Parameters
#         ----------
#         x      : tf.Tensor of shape (N, num_features)
#             An array of `N` random draws from the "source" distribution `x ~ P`.
#         y      : tf.Tensor of shape (M, num_features)
#             An array of `M` random draws from the "target" distribution `y ~ Q`.
#         kernel : callable
#             A function which computes the distance between pairs of samples.

#         Returns
#         -------
#         loss   : tf.Tensor of shape (,)
#             The statistically biased squared maximum mean discrepancy (MMD) value.
#         """

#         loss = tf.reduce_mean(kernel(x, x))
#         loss += tf.reduce_mean(kernel(y, y))
#         loss -= 2 * tf.reduce_mean(kernel(x, y))
#         return loss

#     def maximum_mean_discrepancy(self, source_samples, target_samples, 
#                                  kernel="gaussian", 
#                                  mmd_weight=1.0, minimum=0.0):
#         """Computes the MMD given a particular choice of kernel.

#         For details, consult Gretton et al. (2012):
#         https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

#         Parameters
#         ----------
#         source_samples : tf.Tensor of shape (N, num_features)
#             An array of `N` random draws from the "source" distribution.
#         target_samples : tf.Tensor of shape  (M, num_features)
#             An array of `M` random draws from the "target" distribution.
#         kernel         : str in ('gaussian', 'inverse_multiquadratic'), optional, default: 'gaussian'
#             The kernel to use for computing the distance between pairs of random draws.
#         mmd_weight     : float, optional, default: 1.0
#             The weight of the MMD value.
#         minimum        : float, optional, default: 0.0
#             The lower bound of the MMD value.

#         Returns
#         -------
#         loss_value : tf.Tensor
#             A scalar Maximum Mean Discrepancy, shape (,)
#         """

#         # Determine kernel, fall back to Gaussian if unknown string passed
#         if kernel == "gaussian":
#             kernel_fun = self.gaussian_kernel_matrix
            
#         assert kernel == "gaussian" , "only gaussian kernel is implemented"
#         # elif kernel == "inverse_multiquadratic":
#         #     kernel_fun = inverse_multiquadratic_kernel_matrix
#         # else:
#         #     kernel_fun = gaussian_kernel_matrix

#         # Compute and return MMD value
#         loss_value = self.mmd_kernel(source_samples, target_samples, kernel=kernel_fun)
#         loss_value = mmd_weight * tf.maximum(minimum, loss_value)
#         return loss_value
    
# MMD_gaussian = MmdGaussian()