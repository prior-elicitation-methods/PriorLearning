import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_probability as tfp
import bayesflow as bf
import time
import numpy as np

bfn = bf.networks
tfd = tfp.distributions

from MakeMyPrior.global_config_params import _global_variables
from prettytable import PrettyTable

def function_time(func):
    def inner(**kwargs):
        func_name = func.__name__
        start = time.time()
        func(**kwargs)
        end = time.time()
        print(f"{func_name}" + f": {(end-start)*60:.3f}ms") 
        return func(**kwargs)
    return inner


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

        # assert tf.math.reduce_any(
        #     mask
        # ), f"No observation for group {i}. Note: First group is counted as group 0."
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

    
class print_res:  
    def __call__(self, method, res, precision, 
                 true_values=None, names = None, start = None, num_vars = None, 
                 **kwargs):
        
        assert method == "hyperparameter_learning" or method == "normalizing_flow", "Method must be either 'hyperparameter_learning' or 'normalizing_flow'." 
        
        if method == "normalizing_flow":
            assert names is not None, "Method 'normalizing_flow' needs a string-vector with names of model parameters."
        if method == "hyperparameter_learning":
            assert num_vars is not None, "For method 'hyperparameter_learning' the argument 'num_vars' (number of hyperparameters) is missing."
            assert start is not None, "For method 'hyperparameter_learning' the argument 'start' (number of values that should be averaged for final value) is missing."
        
        
        if method == "hyperparameter_learning":
            avg_res = self.avg_value(res, num_vars)
            tab, means = self.create_table(num_vars, avg_res, start, precision, true_values)
            return tab, means
        
        if method == "normalizing_flow":
            tab = self.table_flow(res, names, precision, **kwargs)
            return tab
    
    def avg_value(self, res, num_vars):
        final_epochs = len(res[0])
        vars_dict = dict()
        for i in range(num_vars):
            vars_name = res[1][i][:-2] 
            
            if vars_name[0:4] == "rate" or vars_name[0:5] == "scale":
                vars_values = [tf.exp(res[0][e][i]) for e in range(final_epochs)]
            else: 
                vars_values = [res[0][e][i] for e in range(final_epochs)]
            vars_name = res[1][i][:-2] 
            vars_dict[vars_name] = vars_values
        return vars_dict
    
    def table_flow(self, res, names, precision, true_mu, true_sigma, true_nu):
        x = PrettyTable(field_names = ["var_name", "pred mean", "pred std"])
        for i in range(len(names)):
            x.add_row([names[i], np.round(np.mean(res[:,:,i]), precision), 
                       np.round(np.std(res[:,:,i]), precision)])
        
        if true_mu is not None:
            sig_true = tfd.Exponential(true_nu).sample(1000)
            sig_m = np.round(np.mean(sig_true), precision)
            sig_sd = np.round(np.std(sig_true), precision)
            
            x.add_column("true mean", true_mu+[sig_m])
            x.add_column("err mean", abs(np.round(tf.subtract(true_mu+[sig_m],
                                                           np.mean(res, (0,1))).numpy(), 
                                           precision)))
            x.add_column("true std", true_sigma+[sig_sd])
            x.add_column("err std", abs(np.round(tf.subtract(true_sigma+[sig_sd],
                                                           np.std(res, (0,1))).numpy(), 
                                           precision)))
        return x
            
    def create_table(self, num_vars, avg_res, start, precision, true_values):
        x = PrettyTable(field_names = ["var_name", "pred mean", "pred std"])
        means = []
        for i in range(num_vars):
            m, s = self.mean_std(avg_res, list(avg_res.keys())[i], start,  
                                 precision)
            means.append(m)
            x.add_row([list(avg_res.keys())[i], m, s])
        
        if true_values is not None:
            x.add_column("true", true_values)
            x.add_column("err", abs(np.round(tf.subtract(true_values,means).numpy(), 
                                           precision)))
            
        return x, means
    
    def mean_std(self, avg_res, key, start, precision):
        end = 1
        
        val_mean = lambda key, start, end: np.round(np.mean(avg_res[key][-start:-end]),precision)
        val_std = lambda key, start, end: np.round(np.std(avg_res[key][-start:-end]),precision)
        
        return val_mean(key, start, end), val_std(key, start, end)

print_restab = print_res()

def plot_priors_flow(res, true_mu, true_sigma, true_nu):
    _, axes = plt.subplots(2,4, constrained_layout = True, figsize = (6,3))
    [sns.kdeplot(res[0,:,i], ax = axes[0,i], color = "orange", 
                lw = 3) for i in range(4)]
    [sns.kdeplot(tfd.Normal(true_mu[i], true_sigma[i]).sample(1000), 
                linestyle = "dashed", color = "black", ax = axes[0,i]) for i in range(4)]
    [sns.kdeplot(res[0,:,i], ax = axes[1,j], color = "orange", 
                lw = 3) for j,i in enumerate(range(4,7))]
    [sns.kdeplot(tfd.Normal(true_mu[i], true_sigma[i]).sample(1000), 
                linestyle = "dashed", color = "black", ax = axes[1,j]) for j,i in enumerate(range(4,6))]
    sns.kdeplot(tfd.Exponential(true_nu).sample(1000), 
                linestyle = "dashed", color = "black", ax = axes[1,2])
    axes[1,3].set_axis_off()
    [axes[0,i].set_ylabel(None) for i in range(1,4)]
    [axes[1,i].set_ylabel(None) for i in range(1,4)]
    [axes[0,i].set_title(fr"$\beta_{{{i}}}$") for i in range(4)]
    [axes[1,j].set_title(fr"$\beta_{{{i}}}$") for j,i in enumerate(range(4,7))]
    axes[1,2].set_title(r"$\sigma$")
    axes[1,3].add_patch(plt.Rectangle((0,0.8), 0.1, 0.07, color = "orange"))
    axes[1,3].text(0.2, 0.8, 'learned prior', fontsize="small")
    axes[1,3].add_patch(plt.Rectangle((0,0.6), 0.1, 0.07, color = "black"))
    axes[1,3].text(0.2, 0.6, 'true prior', fontsize="small")
    
def plot_priors_hyp(avg_res, true_values):
    _, axes = plt.subplots(2,4, constrained_layout = True, figsize = (8,4))
    [sns.kdeplot(tfd.Normal(avg_res[i], avg_res[i+1]).sample(1000), 
                 ax = axes[0,s], lw=3) for s,i in enumerate([0,2,4,6])]
    [sns.kdeplot(tfd.Normal(true_values["mu"][s], true_values["sigma"][s]).sample(1000), 
                 ax = axes[0,s], color = "black", linestyle = "dashed") for s in range(4)]
    [sns.kdeplot(tfd.Normal(avg_res[i], avg_res[i+1]).sample(1000), 
                 ax = axes[1,s], lw=3) for s,i in enumerate([8,10])]
    [sns.kdeplot(tfd.Normal(true_values["mu"][s], true_values["sigma"][s]).sample(1000), 
                 ax = axes[1,i], color = "black", linestyle = "dashed") for i,s in enumerate(range(4,6))]
    sns.kdeplot(tfd.Exponential(avg_res[-1]).sample(1000), lw = 3, ax = axes[1,2])
    sns.kdeplot(tfd.Exponential(true_values["nu"]).sample(1000), ax = axes[1,2], 
                color = "black", linestyle = "dashed")
    [axes[0,i].set_xlabel(rf"$\beta_{i}$") for i in range(4)]
    [axes[1,j].set_xlabel(rf"$\beta_{i}$") for j,i in enumerate(range(4,6))]
    axes[1,2].set_xlabel(r"$s$")
    axes[1,3].set_axis_off()
    
def plot_expert_preds(expert_res_list):
    d = expert_res_list[list(expert_res_list.keys())[0]]
    d2 = expert_res_list[list(expert_res_list.keys())[1]]
    d3 = expert_res_list[list(expert_res_list.keys())[2]]
    
    _, axs = plt.subplots(1,3, constrained_layout = True, figsize = (8,3))
    [sns.scatterplot(y=k, x=d3[j][0,:], ax = axs[1], color = c, label = l) for c,l,k,j in zip(["yellowgreen", "lawngreen","seagreen"],["deep","standard","shallow"], np.linspace(0,1,3),range(3))]
    [sns.scatterplot(y=k, x=d2[j][0,:], ax = axs[0], color = c) for c,k,j in zip(["cornflowerblue"]*3+["lightsalmon"]*3,np.linspace(0,1,6),range(6))]
    axs[0].set_xlabel("mean PTJ")
    axs[0].set_yticklabels(["r","new: deep","new: standard","new: shallow","rep: deep","rep: standard","rep: shallow"])
    axs[1].set_yticklabels([])
    sns.histplot(d[0,:], bins = 10, ax = axs[2])
    axs[2].set_xlim(0,1)
    axs[1].set_xlim(0,0.3)
    axs[2].set_title("R2")
    axs[1].set_xlabel(r"$\Delta$ PTJ")
    axs[0].set_title("Group means")
    axs[1].set_title("Truth effect")
    axs[1].legend(fontsize = "x-small", loc = "upper right")
    plt.show()

def group_stats(ypred, num_groups):
    x = PrettyTable(field_names = ["group", "mean", "std"])
    [x.add_row([f"gr_{i}", 
               np.round(tf.reduce_mean(ypred[:,:,i::num_groups]),2),
               np.round(tf.math.reduce_std(
                   tf.reduce_mean(ypred[:,:,i::num_groups],-1)),2)
               ]) for i in range(num_groups)]
    return x
