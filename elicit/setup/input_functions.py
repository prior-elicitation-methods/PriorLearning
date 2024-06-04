import inspect
import tensorflow as tf
import warnings
from functions.loss_functions import MmdEnergy
from dags.elicitation_pipeline import prior_elicitation_dag
from setup.create_dictionaries import create_global_dict

def normalizing_flow_specs(
        num_coupling_layers: int = 7,
        coupling_design: str = "affine", 
        coupling_settings: dict = {
            "dropout": False,
            "dense_args": {
                "units": 128,
                "activation": "softplus",
                "kernel_regularizer": tf.keras.regularizers.l2(1e-4)
                },
            "num_dense": 2
            },
        permutation: str = "fixed",
        **kwargs
        ) -> dict:
    # for more information see BayesFlow documentation
    # https://bayesflow.org/api/bayesflow.inference_networks.html
    
    nf_specs_dict = {
        "num_coupling_layers": num_coupling_layers,
        "coupling_design": coupling_design,      
        "coupling_settings": coupling_settings,                                 
        "permutation": permutation
        }
    
    return nf_specs_dict

def param(name: str, 
          family: callable = None, 
          hyperparams_dict: dict = None,
          scaling_value: float = 1.) -> dict:
    """
    Creates a dictionary including user information about the model parameters.

    Parameters
    ----------
    name : str
        name of the model parameter
            
    family : callable, optional
        prior distribution family as tfp.distribution object
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                tfd.Normal      # or
                tfd.Gamma       # or
                tfd.Exponential # etc.
            
    hyperparams_dict : dict, optional
        dictionary including all hyperparameters of the pre-specified prior distribution family, with
        key referring to hyperparameter name and value to the initial value or a tfp.distribution when initializing from a distribution is desired.
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                {"mu0": tfd.Normal(0.,1.),
                 "sigma0": 1.0}
        
    scaling_value : float, optional
        After sampling from the prior distribution it is possible to scale the value before it is used
        as input for the generative model. This might be helpful when model parameter values are expected
        to be on very different scales, e.g., b0=250. vs. b1=2. In this case it might be reasonable to scale
        b0 by b0/100.

    Returns
    -------
    dict
        Dictionary including all user information about model parameters

    """
    
    # specifies name and value of prior hyperparameters
    if hyperparams_dict is not None:
        hyppar_dict = {key: value for key,value in zip(hyperparams_dict.keys(),
                                                       hyperparams_dict.values())}
    else:
        hyppar_dict = None
    
    # creates final dictionary with prior specifications
    parameter_dict = {
        "name": [name], 
        "family": [family], 
        "hyperparams_dict": [hyppar_dict],
        "scaling_value": [scaling_value]
        }
    
    return parameter_dict


def model(model: callable, 
          additional_model_args: dict, 
          discrete_likelihood: bool,
          softmax_gumble_specs: dict or None = {"temperature": 1.,
                                                "upper_threshold": None}
          ) -> dict:
    """
    Creates a dictionary including user information about the generative model.

    Parameters
    ----------
    model : callable
        class object that specifies the generative model. 
        The class object requires a specific input-output structure which is presented in the following 
        example. Additional input and output arguments can be added but the here presented input and 
        output arguments are necessary.
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                class GenerativeModel(tf.Module):
                    def __call__(self, prior_samples, **kwargs):
                        
                        # linear predictor
                        theta = some_func(prior_samples)
    
                        # use link function (here identity)
                        epred = some_link_fun(theta)
                        
                        # define likelihood (here example Normal)
                        likelihood = tfd.Normal(loc = epred, scale = 1.)
                        
                        # sample prior predictions
                        ypred = likelihood.sample()
                    
                    return dict("ypred": ypred,
                                "epred": epred,
                                "likelihood": likelihood,
                                "prior_samples": prior_samples)
            
        :Note:
            
            For discrete likelihoods "ypred" is not sampled within the generative model
            but rather a continuous approximation will be used (see softmax-gumble-specs). 
            Thus, for discrete likelihoods: "ypred"=None
        
    additional_model_args : dict, optional
        Required if in the generative model (class object) input arguments are provided besides the
        obligatory 'prior_samples' argument. The following example shows a situation in which a design matrix
        is provided additionally.
        Important is that the key name is equal to the input argument name.
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                {"design_matrix": # path-to-design-matrix}
            
    discrete_likelihood : bool
        True if the likelihood is a discrete random variable otherwise False
    softmax_gumble_specs : dict or None, optional
        Only required if the likelihood is a discrete random variable. 
        In this case the output argument "ypred" is computed as a continuous approximation using the
        softmax-gumble trick. If the softmax-gumble trick is used further hyperparameters need to be set as
        shown by the following example.
                                                      
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                {"temperature": 1.,
                 "upper_threshold": 31}

        * The `temperature` argument specifies how strong the continuous approx. should be. 
          Values tending towards zero lead to a categorical distribution.
        * The `upper_threshold` argument defines the upper truncation of a categorical distribution. 
          For double-bounded distribution as the Binomial distribution is simple the total_count value. 
          For lower-bounded distribution a reasonable threshold value needs to be chosen. Note that it is not
          reasonable to be too conservative and set the upper value too high, as higher values will lead to longer
          computation time.

    Returns
    -------
    dict
        Dictionary including all user information about the generative model.

    """
  
    if softmax_gumble_specs is not None:
        softmax_gumble_specs_default = {"temperature": 1.,
                                        "upper_threshold": None}
        
        softmax_gumble_specs_default.update(softmax_gumble_specs)
    
    model_dict = {
        "model_function": model,
        "additional_model_args": additional_model_args,
        "softmax_gumble_specs": softmax_gumble_specs_default
        }
    
    # provide a warning if the upper_threshold argument is not specified
    if discrete_likelihood:
        assert "upper_threshold" in softmax_gumble_specs, "The upper_threshold argument in the softmax-gumble specifications is None."
        if "temperature" not in softmax_gumble_specs:
            warnings.warn(f"The Softmax-Gumble method with default temperature: {softmax_gumble_specs_default['temperature']} is used.")
    # get model arguments
    get_model_args = set(inspect.getfullargspec(model())[0]).difference({"self","prior_samples"})
    # check that all model arguments have been passed as input in the model_args section
    difference_set = get_model_args.difference(set(additional_model_args.keys()))
    assert len(difference_set) == 0, f"model arguments and specified arguments are not identical. Check the following argument(s): {difference_set}"
    
    return model_dict

def target(name: str, 
           elicitation_method: str, 
           loss_components: str, 
           select_obs: list or None = None, 
           quantiles_specs: tuple or None = None, 
           moments_specs: tuple or None = None, 
           custom_target_function: dict or None = None,
           custom_elicitation_function: dict or None = None) -> dict:
    """
    Creates a dictionary including user information about the target quantities and
    corresponding elicitation techniques.

    Parameters
    ----------
    name : str
        name of the target quantity. If a specific output argument from the generative
        model shall be used as target quantity, then the exact name should be indicated here. 
        It is not allowed to have multiple target quantities with the same name.
    elicitation_method : str
        currently implemented are 'histogram' (histogram-based elicitation),
        'moments' (moment-based elicitation), and 'quantiles' (quantile-based elicitation).
    loss_components : str
        method indicating how the elicited statistics should enter the loss. Currently available
        are 'all', 'by-group', and 'by-stats'. The following example shall demonstrate the differences:
            
        :Example:
            
            Consider as target quantity group_means from a factor with 3 levels and as elicitation
            technique: quantile-based elicitation using 5 quantiles. Then the elicited statistic will
            have the shape (B,5,3).
            
            * 'all': would result in one loss component of shape (B,15) 
            * 'by-group': would result in three loss components of shape (B,5)
            * 'by-stats': would result in five loss components of shape (B,3)

        The decision for one of the methods should be guided by the dependency structure. 
        If I want to learn the distribution of a target quantity based on quantiles, 
        the quantiles should be together in one loss component, thus in this case 
        the method 'by-group' should be preferrable.
    select_obs : list or None, optional
        Only needed if "ypred" is used as target quantity but only specific design 
        points should be used. Then a list with integers referring to the index of
        each design point in the design matrix is required.
    quantiles_specs : tuple or None, optional
        Only needed if elicitation_method = 'quantiles'. Tuple indicating the percentages that should
        be elicited. 
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                (25, 50, 75)          # or 
                (5, 25, 50, 75, 95)
            
        It is not recommended to use the maximum or minimum value (i.e., 0 and 100)
        as it leads generally to worse learning results.
    moments_specs : tuple or None, optional
        Only needed if elicitation_method = 'moments'. Tuple of strings indicating which moments 
        should be elicited. Currently only "mean" and "sd" are implemented
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
                
                ("mean", "sd")    # or 
                ("mean", )
        
    custom_target_function : dict or None, optional
        It is possible to specify a custom function for deriving any desired target quantity. It can take
        output arguments from the generative model as input arguments as well as additional user-specified 
        arguments. The following example shows the basic structure.
            
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                {"function": callable_custom_function,
                 "additional_args": {"argument1": 3.}}
            
        * the `function` argument takes as value the callable custom function
        * if the custom function requires input arguments that are not part of the
          output arguments of the generative model (i.e., ypred, epred, likelihood, prior_samples)
          then these additional arguments have to be specified in the `additional_args` argument in
          form of a dictionary. The key should have the same name as the argument.
            
    custom_elicitation_function : dict or None, optional
        It is possible to specify a custom function to create an elicitation technique that deviates
        from the currently implemented techniques. It has the same structure as explained in "custom_target_function".
        This function has not yet been tested.

    Returns
    -------
    dict
        Dictionary including all user information about the target quantities and elicitation techniques.

    """
    def check_custom_func(custom_function, func_type):
        if custom_function is not None:
            # check whether custom_function has correct form if specified
            assert "function" in list(custom_function.keys()), f"custom_{func_type}_function must be a dictionary with required key 'function' and optional key 'additional_args'"
            default_custom_function = {
                "function": None,
                "additional_args": None
                }
            default_custom_function.update(custom_function)
            
            # check whether additional args have been specified 
            # if they are specified check for correct form
            try:
                default_custom_function["additional_args"]
            except: 
                pass
            else:
                if default_custom_function["additional_args"] is not None:
                    assert type(default_custom_function["additional_args"]) is dict, "additional_args must be a dictionary keys are the name of the argument and value the corresponding value of the argument to be passed"
            custom_function = default_custom_function    
        return custom_function
    
    check_custom_func(custom_target_function, func_type = "target")
    check_custom_func(custom_elicitation_function, func_type = "elicitation")
    # currently implemented loss-component combis
    assert loss_components in ["by-group", "all", "by-stats"], "Currently only available values for loss_components are 'all', 'by-stats' or 'by-group'."
    # currently implemented elicitation methods are histogram, quantiles, or moments
    assert elicitation_method in ["quantiles", "histogram", "moments"], f"The elicitation method {elicitation_method} is not implemented"
    # if elicitation method 'quantiles' is used, quantiles_specs has to be defined
    if elicitation_method == "quantiles":
        assert quantiles_specs is not None, "quantiles_specs has to be defined for elicitation method 'quantiles'"
    if elicitation_method == "moments":
        assert moments_specs is not None, "moments_specs has to be defined for elicitation method 'moments'"
    
    target_dict = {
        "name": [name],
        "elicitation_method": [elicitation_method],
        "select_obs": [select_obs],
        "quantiles_specs": [quantiles_specs],
        "moments_specs": [moments_specs],
        "custom_target_function": [custom_target_function],
        "custom_elicitation_function": [custom_elicitation_function],
        "loss_components": [loss_components]
        }
    
    return target_dict


def expert(data: str or None,
           simulate_data: bool,
           simulator_specs: dict or None) -> dict:
    """
    Creates a dictionary including user information about the expert data which should be
    used for training the simulation model. It is possible to provide expert
    information from a file or to simulate expert information based on a pre-defined ground truth
    which might be helpful for checking method performance and tuning the algorithm hyperparameter values.

    Parameters
    ----------
    data : str or None
        path to file in which expert data is saved. The expert data must have the same format
        as the model-simulated elicited statistics (more information will follow)
    simulate_data : bool
        if true, expert data is simulated based on a pre-defined ground truth. This setting
        might be usefull to check method validity and performance in general.
    simulator_specs : dict or None
        Only needed if simulate_data=True. If data should be simulated the true prior distribution
        for each model parameter needs to be specified. The following example shows a situation 
        in which two model parameters b0 and b1 with normal prior distribution family had been 
        specified in the `param()` object.
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                {"b0": tfd.Normal(250.4, 7.27),
                 "b1": tfd.Normal(30.26, 4.82)}
            
        The true hyperparameter values are consequently (250.4, 7.27, 30.26, 4.82).

    Returns
    -------
    dict
        Returns a dictionary including user information about expert data or simulated data given a 
        pre-defined ground truth.

    """
    # either expert data or expert simulator has to be specified
    assert not(data is None and simulate_data is False), "either a path to expert data has to be specified or simulation of expert data has to be true."
    # if expert simulator is true, simulator specifications have to be given
    assert simulate_data is True and simulator_specs is not None, "if expert simulation is true simulator specifications need to be provided"
    
    expert_dict = {
        "data": data,
        "simulate_data": simulate_data,
        "simulator_specs": simulator_specs
        } 
    return expert_dict

def loss(loss_function: str or callable = "mmd-energy", 
         loss_weighting: dict or None = {
             "method": "dwa",
             "method_specs": { "temperature": 1.6 }
             }
         ) -> dict:
    
    # check whether user-specified loss function is implemented
    if type(loss_function) is str:
        assert loss_function in ["mmd-energy"], "Currently only the following loss functions are implemented 'mmd-energy'."
        if loss_function == "mmd-energy":
            # call class and initialize it
            call_loss_function = MmdEnergy()
    else:
        call_loss_function = loss_function
    # check whether user-specified loss-weighting is implemented
    if loss_weighting is not None:
        assert loss_weighting["method"] in ["dwa"], "Currently only the following loss-weighting methods are implemented 'dwa'."

    loss_dict = {
        "loss_function": call_loss_function,
        "loss_weighting": loss_weighting
        }
    return loss_dict

def optimization(optimizer: callable = tf.keras.optimizers.Adam,
                 optimizer_specs: dict = {
                     "learning_rate": callable or float,
                     "clipnorm": 1.0
                     }
                 ) -> dict:
    """
    Creates a dictionary including user information about the optimizer.

    Parameters
    ----------
    optimizer : callable
        Optimizer used for batch-stochastic gradient descent in form of a tf.keras.optimizers object. 
        The default is tf.keras.optimizers.Adam.
    optimizer_specs : dict, optional
        Specifications of input arguments for the optimizer. This includes for example
        the initial learning rate which can be provided as float or in form of a learning rate schedule. 
        The default is shown in the following example.
        
        :Example:
            
            .. highlight:: python
            .. code-block:: python
            
                {"learning_rate": tf.keras.optimizers.schedules.CosineDecayRestarts(
                                  0.005, 100),                    
                 "clipnorm": 1.0}

    Returns
    -------
    dict
        dictionary including user information about the optimizer.

    """
    # TODO assert that keywords of optimizer are in optimizer_specs
    optim_dict = {
        "optimizer": optimizer,
        "optimizer_specs": optimizer_specs
        }
    
    return optim_dict

def prior_elicitation(
        method: str,
        sim_id: str,
        epochs: int,
        B: int,
        rep: int,
        seed: int,
        burnin: int,
        model_params: callable,
        expert_input: callable,
        generative_model: callable,
        target_quantities: callable,
        loss_function: callable,
        optimization_settings: callable,
        output_path: str = "results",
        print_info: bool = True,
        view_ep: int = 1
        ) -> dict:
    """
    wrapper around the optimization process, when called a global dictionary will be 
    created including all information about the user specifications and the 
    learning process starts.

    Parameters
    ----------
    method : str
        which method should be used, either 'deep_prior' or 'parametric_prior'.
    sim_id : str
        unique identification of simulation used to save result in a folder with
        the corresponding name.
    epochs : int
        number of epochs 
    B : int
        batch size (2^7 or 2^8 should be enough)
    rep : int
        number of samples from the prior distributions (200 to 300 should be enough)
    seed : int
        seed for reproducibility.
    burnin : int
        number of initializations that are tried out before learning starts. 
        The initialization setting leading to the smallest loss is used for running the learning algorithm.
        Method is only reasonable for 'parametric_prior' method.
    model_params : callable
        user information on model parameters using the `param()` object
    expert_input : callable
        user information on expert data or pre-defined ground truth using the `expert()` object
    generative_model : callable
        user information on the generative model using the ´model()´ object
    target_quantities : callable
        user information on the target quantities and elicitation techniques using the `target()` object
    loss_function : callable
        user information on the loss function using the `loss()` object
    optimization_settings : callable
        user information on the optimization method using the `optimization()` object
    output_path : str
        name of folder in which results should be saved
    print_info : bool
        whether user feedback about epoch, loss value, and average time per epoch should be provided during training
    view_ep : int
        if user feedback shall be provided after how many epochs shall information be provided?
        Default value is 1, thus feedback is provided after every epoch.

    Returns
    -------
    dict
        global dictionary including all user input.

    """
    
    # create global dict.
    global_dict = create_global_dict(
        method, sim_id, epochs, B, rep, seed, burnin, model_params, expert_input, 
        generative_model, target_quantities, loss_function, 
        optimization_settings, output_path, print_info, view_ep)

    # run workflow
    prior_elicitation_dag(global_dict)
