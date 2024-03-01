import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf

bfn = bf.networks
tfd = tfp.distributions

from MakeMyPrior.global_config_params import _global_variables


class Normal_unconstrained:
    """Tensorflow_probability Normal distribution with exponentiated scale"""

    def __init__(self):
        self.keys = ["loc", "scale"]

    def __call__(self, loc, scale):
        """

        Parameters
        ----------
        loc : float
            The mean of the distribution.
        scale : float
            The standard deviation of the distribution on the (unconstrained)
            log scale

        Returns
        -------
        tensorflow_probability.distributions.distribution
            tfd.Normal distribution with mean: 'loc' and standard deviation:
                exp scale

        """
        return tfd.Normal(loc=loc, scale=tf.exp(scale))


class TruncatedNormal_unconstrained:
    def __init__(self, loc, low, high):
        """Tensorflow_probability TruncatedNormal distribution with exponentiated
        scale

        Parameters
        ----------
        loc : float
            The mean of the distribution.
        low : float
            Lower bound of the distribution's support. (Must be lower than
            value of argument 'high')
        high : float
            Upper bound of the distribution's support. (Must be higher than
            value of argument 'low')

        """
        self.low = low
        self.high = high
        self.loc = loc
        self.keys = ["scale"]

    def __call__(self, scale):
        """Tensorflow_probability TruncatedNormal distribution with exponentiated
        scale

        Parameters
        ----------
        scale : float
            The standard deviation of the normal distribution on the
            (unconstrained) log scale.

        Returns
        -------
        tensorflow_probability.distributions.distribution
            tfd.TruncatedNormal distribution with mean: 'loc' and standard
            deviation: exp scale

        """
        return tfd.TruncatedNormal(
            loc=self.loc, scale=tf.exp(scale), low=self.low, high=self.high
        )


class Exponential_unconstrained:
    def __init__(self, rep):
        """Tensorflow_probability Exponential distribution reparameterized
        as Gamma distribution and with exponentiated rate

        Assume s ~ Exponential(exp(rate)), then the mean of N replicated s follows
        a Gamma distribution:  1/N sum_n s_n ~ Gamma(N, N*exp(rate))

        This reparameterization achieved more robust results in simulations

        Parameters
        ----------
        rep : integer
            The number of simulated model parameters.

        """
        self.rep = rep
        self.keys = ["rate"]

    def __call__(self, rate):
        """Tensorflow_probability TruncatedNormal distribution with exponentiated
        scale

        Parameters
        ----------
        rate : float
            Rate parameter of Exponential distribution on the (unconstrained)
            log scale.

        Returns
        -------
        tensorflow_probability.distributions.distribution
            tfd.Gamma distribution concentration: 'rep' and rate: rep*'exp rate'

        """
        return tfd.Gamma(concentration=self.rep, rate=tf.exp(rate) * self.rep)


def create_cinn(num_params, num_coupling_layers):
    """Construct an invertible network using the bayesflow package

    Parameters
    ----------
    num_params : integer
        Number of model parameters.
    num_coupling_layers : integer
        Number of coupling layers.

    Returns
    -------
    invertible_net : bayesflow.networks.InvertibleNetwork
        Definition of the architecture of the invertible network.

    """
    invertible_net = bfn.InvertibleNetwork(
        num_params=num_params,
        num_coupling_layers=num_coupling_layers,
        coupling_design=_global_variables["coupling_design"],
        coupling_settings={
            "dense_args": dict(
                units=_global_variables["units"],
                activation=_global_variables["activation"],
            )
        },
        permutation=_global_variables["permutation"],
    )
    return invertible_net


def group_obs(samples, dmatrix_fct, cmatrix):
    """Function that assigns the values of the outcome variable y to groups
    given a design matrix consisting only of factors and a contrast matrix

    Parameters
    ----------
    samples : tf.tensor of shape (B, rep, N_obs)
        Samples of the response variable.
    dmatrix_fct : tf.tensor of shape (N_obs, num_factors+1)
        Design matrix consisting only of factors and intercept.
    cmatrix : tf.tensor of shape (num_groups, num_factors+1)
        contrast matrix given the respective contrast coding.

    Returns
    -------
    grouped_samples : tf.tensor of shape (B, rep, N_gr, num_groups) (where N_gr
                      is the number of observations within each group)
        Tensor with observations from the response variable assigned to
        respective group.

    """
    gr_samples = []
    # loop over all groups
    for i in range(cmatrix.shape[0]):
        # check for each observation whether it is in the respective group or not
        mask = tf.reduce_all(tf.cast(cmatrix, tf.float32)[i, :] == dmatrix_fct, 1)

        assert tf.math.reduce_any(
            mask
        ), f"No observation for group {i}. Note: First group is counted as group 0."
        # select all observations which are in the respective group
        gr_samples.append(tf.boolean_mask(tensor=samples, mask=mask, axis=2))
    # stack all observations together such that final tensor shape is
    # (B, rep, N_gr, num_groups)
    grouped_samples = tf.stack(gr_samples, -1)

    return grouped_samples


def scale_predictor(X_col, method):
    """Function that scales the predictor variable

    Parameters
    ----------
    X_col : tf.tensor of shape (1, N_obs)
        Continuous predictor variable from the design matrix.
    method : string; either 'standardize' or 'normalize'
        Scaling method applied to the predictor variable.

    Returns
    -------
    X_col_std : tf.tensor of shape (1, N_obs)
        scaled predictor variable.

    """
    # compute mean and standard dev. of predictor variable
    emp_mean = tf.reduce_mean(X_col)
    emp_sd = tf.math.reduce_std(X_col)

    if method == "standardize":
        # z = (x-mean)/sd
        X_col_std = (X_col - emp_mean) / emp_sd
    if method == "normalize":
        # z = x/sd
        X_col_std = X_col / emp_sd

    return X_col_std


def scale_loss_component(method, loss_comp_exp, loss_comp_sim):
    """Function for scaling the expert and model elicited statistic of
    a particular loss component

    Parameters
    ----------
    method : string; either 'scale-by-std', 'scale-to-unity', or 'unscaled'
        Scaling method applied to the elicited statistics of the loss component.
    loss_comp_exp : tf.tensor of shape (B, stats)
        Elicited statistics from the expert.
    loss_comp_sim : tf.tensor of shape (B, stats)
        Elicited statistics as simulated by the model.

    Returns
    -------
    loss_comp_exp_std : tf.tensor of shape (B, stats)
        if not 'unscaled': scaled elicited statistics from the expert otherwise
        unmodified elicited statistics
    loss_comp_sim_std : tf.tensor of shape (B, stats)
        if not 'unscaled': scaled elicited statistics from the model otherwise
        unmodified elicited statistics

    """
    if method == "unscaled":
        return loss_comp_exp, loss_comp_sim

    if method == "scale-by-std":
        emp_std_sim = tf.expand_dims(tf.math.reduce_std(loss_comp_sim, axis=-1), -1)

        loss_comp_exp_std = tf.divide(loss_comp_exp, emp_std_sim)
        loss_comp_sim_std = tf.divide(loss_comp_sim, emp_std_sim)
        
        return loss_comp_exp_std, loss_comp_sim_std

    if method == "scale-to-unity":
        emp_min_sim = tf.expand_dims(tf.math.reduce_min(loss_comp_sim, axis=-1), -1)
        emp_max_sim = tf.expand_dims(tf.math.reduce_max(loss_comp_sim, axis=-1), -1)

        loss_comp_exp_std = tf.math.divide(
            loss_comp_exp - emp_min_sim, emp_max_sim - emp_min_sim
        )
        loss_comp_sim_std = tf.math.divide(
            loss_comp_sim - emp_min_sim, emp_max_sim - emp_min_sim
        )
        
        return loss_comp_exp_std, loss_comp_sim_std

    
