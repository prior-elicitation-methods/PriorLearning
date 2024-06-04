import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf

tfd = tfp.distributions
bfn = bf.networks

from functions.helper_functions import save_as_pkl

def priors(global_dict, ground_truth=False):
    """
    Creates a class instantiation which initializes and samples from prior 
    distributions

    Parameters
    ----------
    global_dict : dict
        dictionary including all user-input settings..
    ground_truth : bool
        whether simulation is based on a pre-specified ground truth 
        (for method validation). The default is False.

    Returns
    -------
    prior_model : class instance
        class instance for initializing and sampling from prior distributions.

    """
    # initalize generator model
    class Priors(tf.Module):
        def __init__(self, ground_truth, global_dict):
            self.global_dict = global_dict
            self.ground_truth = ground_truth
            # initialize hyperparameter for learning (if true hyperparameter
            # are given, no initialization is needed)
            if not self.ground_truth:
                self.init_priors = intialize_priors(self.global_dict)
            else:
                self.init_priors = None

        def __call__(self):
            prior_samples = sample_from_priors(self.init_priors, 
                                               self.ground_truth, 
                                               self.global_dict)
            return prior_samples
    # create a class instance
    prior_model = Priors(ground_truth, global_dict)
    return prior_model

def intialize_priors(global_dict):
    """
    Initialize prior distributions.

    Parameters
    ----------
    global_dict : dict
        dictionary including all user-input settings..

    Returns
    -------
    init_prior : dict
        returns initialized prior distributions ready for sampling.

    """
    # number parameters
    no_param = len(global_dict["model_params"]["name"])

    if global_dict["method"] == "parametric_prior":
        # list for saving initialize hyperparameter values
        init_hyperparam_list = []
        # loop over model parameter and initialize each hyperparameter 
        # given the user specified initialization
        for model_param in range(no_param):
            get_hyp_dict = global_dict["model_params"]["hyperparams_dict"]
            initialized_hyperparam = dict()
            for name, init_val in zip(get_hyp_dict[model_param].keys(), 
                                      get_hyp_dict[model_param].values()):  
                # check whether initial value is a distributions
                # TODO currently we silently assume that we have either a value or a tfd.distribution object
                try:
                    init_val.reparameterization_type
                except:
                    initial_value = init_val
                else:
                    initial_value = init_val.sample()
                
                # if value was initialized on the log level apply exp. transf.
                if name.startswith("log_"):
                    name = name.removeprefix("log_")
                    
                # initialize hyperparameter
                initialized_hyperparam[f"{name}"] = tf.Variable(
                    initial_value = initial_value,
                    trainable = True,
                    name = f"{name}",
                )
            init_hyperparam_list.append(initialized_hyperparam)    
        # save initialized priors
        init_prior = init_hyperparam_list
        # save file in object
        path = global_dict["output_path"]["data"]+'/init_prior.pkl'
        save_as_pkl(init_prior, path)

    if global_dict["method"] == "deep_prior":
        # for more information see BayesFlow documentation
        # https://bayesflow.org/api/bayesflow.inference_networks.html
            
        invertible_neural_network = bfn.InvertibleNetwork(
            num_params = no_param,
            **global_dict["model_params"]['normalizing_flow_specs']
        )
        # save initialized priors
        init_prior = invertible_neural_network
        
    return init_prior

def sample_from_priors(initialized_priors, ground_truth, global_dict):
    """
    Samples from initialized prior distributions.

    Parameters
    ----------
    initialized_priors : dict
        initialized prior distributions ready for sampling.
    ground_truth : bool
        whether simulations are based on ground truth (then sampling is performed
                                                       from true distribution).
    global_dict : dict
        dictionary including all user-input settings..

    Returns
    -------
    prior_samples : dict
        Samples from prior distributions.

    """
    # extract variables from dict
    rep = global_dict["rep"]
    B = global_dict["B"]
    method = global_dict["method"]
    scale_prior_samples = global_dict["model_params"]["scaling_value"]
    # number parameters
    no_param = len(global_dict["model_params"]["name"])

    if ground_truth:
        priors = []
        # TODO: check what happens if you specify one multivariate distribution: How does the code has to be changed?
        for prior in list(global_dict["expert_input"]['simulator_specs'].values()):
            # sample from the prior distribution 
            priors.append(prior.sample((1, rep)))
        prior_samples = tf.stack(priors, axis=-1)

    if method == "parametric_prior" and not ground_truth:
        initialized_hyperparameters = initialized_priors
        priors = []
        for model_param in range(no_param):
            # get the prior distribution family as specified by the user
            prior_family = global_dict["model_params"]["family"][model_param]
            # save hyperparameter values in list
            hyperparams = [initialized_hyperparameters[model_param][key] for key 
                           in initialized_hyperparameters[model_param].keys()]
            # sample from the prior distribution 
            priors.append(prior_family(*hyperparams).sample((B, rep)))
        # stack all prior distributions into one tf.Tensor of
        # shape (B, rep, num_parameters)
        prior_samples = tf.stack(priors, axis=-1)

    if method == "deep_prior" and not ground_truth:
        # initialize base distribution
        base_distribution = tfd.Normal(loc=tf.zeros(no_param), 
                                       scale=tf.ones(no_param))
        # sample from base distribution
        u = base_distribution.sample((B, rep))
        # apply transformation function to samples from base distr.
        prior_samples, _ = initialized_priors(u, condition = None, inverse = False)

        # create results dictionary
        prior_samples_dict = {
            "prior_samples": prior_samples
        }
    
        ## scaling of prior distributions, default is no scaling (value = 1.)
        # scale prior samples
        prior_samples = tf.stack([prior_samples[:,:,i]*scaling_factor for i, 
                                  scaling_factor in enumerate(scale_prior_samples)], -1)
        # save results in dict
        prior_samples_dict["scaled_prior_samples"] = prior_samples
        # save results in path
        saving_path = global_dict["output_path"]["data"]
        path_prior_samples_dict = saving_path+'/prior_samples_dict.pkl'
        save_as_pkl(prior_samples_dict, path_prior_samples_dict)
        
    # save results in path
    saving_path = global_dict["output_path"]["data"]
    if ground_truth:
        saving_path = saving_path+"/expert"
    path_prior_samples = saving_path+'/prior_samples.pkl'
    save_as_pkl(prior_samples, path_prior_samples)
    return prior_samples