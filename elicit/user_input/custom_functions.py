import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd

tfd = tfp.distributions
tfb = tfp.bijectors

#%% custom distribution families

class Normal_log():
    def __init__(self):
        self.name = "Normal_log_scale"
        self.parameters = ["loc","log_scale"]
    def __call__(self, loc, scale):
        """
        Instantiation of normal distribution with sigma being learned on the 
        log-scale.

        Parameters
        ----------
        loc : int
            location parameter of normal distribution.
        scale : int
            scale parameter of normal distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            normal distribution with sigma being on the log scale.

        """
        return tfd.Normal(loc, tf.exp(scale))

class Normal_log_log():
    def __init__(self):
        self.name = "Normal_log"
        self.parameters = ["log_loc","log_scale"]
    def __call__(self, loc, scale):
        """
        Instantiation of normal distribution with both mu and sigma being 
        learned on the log-scale.

        Parameters
        ----------
        loc : int
            location parameter of normal distribution on the original scale.
        scale : int
            scale parameter of normal distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            normal distribution with mu and sigma being on the log scale.

        """
        return tfd.Normal(tf.exp(loc), tf.exp(scale))

class TruncNormal_log():
    def __init__(self, loc, low, high):
        self.name = "TruncatedNormal_log"
        self.parameters = ["log_scale"]
        self.low = low
        self.high = high
        self.loc = loc
    def __call__(self, scale):
        """
        Instantiation of truncated normal distribution with loc=0. and scale 
        being learned on the log scale. 

        Parameters
        ----------
        scale : int
            scale parameter of truncated normal on the original scale.

        Returns
        -------
        tfp.distribution object
            truncated normal distribution with mu=0. and sigma being learned
            on the log scale.

        """
        return tfd.TruncatedNormal(self.loc, tf.exp(scale), 
                                   low=self.low, high=self.high)

class Gamma_log():
    def __init__(self):
        self.name = "Gamma_log"
        self.parameters = ["log_concentration","log_rate"]
    def __call__(self, concentration, rate):
        """
        Instantiation of gamma distribution with both concentration and rate 
        being learned on the log scale. 

        Parameters
        ----------
        concentration : int
            concentration parameter of gamma distribution on the original scale.
        rate : int
            rate parameter of gamma distribution on the original scale.

        Returns
        -------
        tfp.distribution object
            gamma distribution with both concentration and rate being learned
            on the log scale.

        """
        return tfd.Gamma(concentration=tf.exp(concentration), 
                         rate=tf.exp(rate))


class _InvSoftplus(tfb.Bijector):
    def __init__(self, validate_args=False, name='inv_softplus'):
      super(_InvSoftplus, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)

    def _forward(self, x):
      return tfp.math.softplus_inverse(x)

class _Gamma_inv_softplus():
    def __init__(self):
        self.name = "Gamma_inv_softplus"
        self.parameters = ["concentration_softplus","rate_softplus"]
        
    def __call__(self, concentration, rate):
        transformed_dist = tfd.TransformedDistribution(
            distribution=tfd.Gamma(concentration, rate),
            bijector = _InvSoftplus())
        return transformed_dist

#%% custom target quantities
def custom_R2(ypred, epred):
    """
    Defines R2 such that it is guaranteed to lie within zero and one.
    https://avehtari.github.io/bayes_R2/bayes_R2.html#2_Functions_for_Bayesian_R-squared_for_stan_glm_models

    Parameters
    ----------
    ypred : tf.Tensor
        simulated prior predictions from generative model.
    epred : tf.Tensor
        simulated linear predictor from generative model.

    Returns
    -------
    r2 : tf.Tensor
        computed R2 value.

    """
    # variance of linear predictor 
    var_epred = tf.math.reduce_variance(epred, -1) 
    # variance of difference between ypred and epred
    var_diff = tf.math.reduce_variance(tf.subtract(ypred, epred), -1)
    # variance of linear predictor divided by total variance
    r2 = var_epred/(var_epred + var_diff)
    return r2


def custom_group_means(ypred, design_matrix, factor_indices):
    """
    Computes group means from prior predictions with N observations.

    Parameters
    ----------
    ypred : tf.Tensor
        prior predictions as simulated from the generative model.
    design_matrix : tf.Tensor
        design matrix.
    factor_indices : list of integers
        indices referring to factors in design matrix. First columns has index = 0.

    Returns
    -------
    group_means : tf.Tensor
        group means computed from the model predictions.

    """
    # exclude cont. predictor
    dmatrix_fct = tf.gather(design_matrix, factor_indices, axis = 1)
    # create contrast matrix
    cmatrix = tf.cast(pd.DataFrame(dmatrix_fct).drop_duplicates(), tf.float32)
    # compute group means (shape = B,rep,N_obs,N_gr)
    group_means = tf.stack([tf.reduce_mean(tf.boolean_mask(ypred,
                              tf.math.reduce_all(cmatrix[i,:] == dmatrix_fct, axis = 1),
                              axis = 2), -1) for i in range(cmatrix.shape[0])], -1)
    return group_means


def custom_mu0_sd(ypred, selected_days, R2day0, from_simulated_truth = ["R2day0"]):
    """
    Computes the standard deviation of the linear predictor as the squared
    product of R2 with the variance of ypred.

    Parameters
    ----------
    ypred : tf.Tensor
        model predictions as generated from the generative model
    selected_days : list of integers
        indices of days for which the expert has to indicate prior predictions
    R2day0 : tf.Tensor
        R2 for day 0 as predicted by the expert or as simulated from a pre-
        defined ground truth.
    from_simulated_truth : list of strings, optional
        indicates that the argument "R2day0" should be used from the expert data / 
        simulated ground truth (and not searched for in the model simulations). 

    Returns
    -------
    sdmu : tf.Tensor
        standard deviation of linear predictor for day 0.

    """
    day = selected_days[0]
    len_days = len(selected_days)
    sdmu = tf.sqrt(tf.multiply(R2day0, tf.math.reduce_variance(
                                   ypred[:,:,day::len_days], 
                                   axis=-1)))
    return sdmu

def custom_mu9_sd(ypred, selected_days, R2day9, from_simulated_truth = ["R2day9"]):
    """
    Computes the standard deviation of the linear predictor as the squared
    product of R2 with the variance of ypred.

    Parameters
    ----------
    ypred : tf.Tensor
        model predictions as generated from the generative model
    selected_days : list of integers
        indices of days for which the expert has to indicate prior predictions
    R2day9 : tf.Tensor
        R2 for day 9 as predicted by the expert or as simulated from a pre-
        defined ground truth.
    from_simulated_truth : list of strings, optional
        indicates that the argument "R2day9" should be used from the expert data / 
        simulated ground truth (and not searched for in the model simulations). 

    Returns
    -------
    sdmu : tf.Tensor
        standard deviation of linear predictor for day 9.

    """
    day = selected_days[-1]
    len_days = len(selected_days)
    sdmu = tf.sqrt(tf.multiply(R2day9, tf.math.reduce_variance(
                                   ypred[:,:,day::len_days], 
                                   axis=-1)))
    return sdmu