import numpy as np
import inspect
from functions.helper_functions import save_as_pkl


def create_dict(user_input: callable) -> dict:
    """
    Takes tuple of dictionaries as input and creates one dictionary which summaries
    all different dictionary inputs in form of lists.

    Parameters
    ----------
    user_input : callable
        tuple of dictionaries as specified by the used.

    Returns
    -------
    dict
        one dictionary that merges all input dictionaries using lists.

    """
    # get method name
    if len(inspect.getargspec(user_input)[0]) != 0:
        method = inspect.getargspec(user_input)[-1][0]
        assert method in ["learning", "ground_truth"] , "Value for argument 'method' in 'target_quantities' must be either 'learning' or 'ground_truth'"
    # call function
    param_tuple = user_input()
    # initialize output dictionary;  get keys from all input function
    all_keys = np.unique(np.concatenate(
        [list(param_tuple[i].keys()) for i in range(len(param_tuple))], 0))
    param_dict = {f"{key}": [] for key in all_keys}
    # loop over each element in tuple
    for p in range(len(param_tuple)):
        input_dict = param_tuple[p]

        if "num_coupling_layers" in list(input_dict.keys()):
            param_dict["normalizing_flow_specs"] = input_dict

        else:
            for key in list(param_dict.keys()):
                try:
                    input_dict[key]
                except:
                    param_dict[key] = param_dict[key] + [None]
                else:
                    param_dict[key] = param_dict[key] + input_dict[key]
    if len(inspect.getargspec(user_input)[0]) != 0:
        param_dict["method"] = [method]*len(param_tuple)
    return param_dict

# Create necessary config files


def create_global_dict(
        method, sim_id, epochs, B, rep, seed, burnin, model_params, expert_input,
        generative_model, target_quantities, loss_function,
        optimization_settings, output_path, print_info, view_ep) -> dict:
    """
    Creates the global dictionary including all input information from the user.

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
        user information on the generative model using the `model()` object
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

    global_dict = dict()

    global_dict["method"] = method
    global_dict["sim_id"] = sim_id
    global_dict["epochs"] = epochs
    global_dict["B"] = B
    global_dict["rep"] = rep
    global_dict["seed"] = seed
    global_dict["burnin"] = burnin
    global_dict["model_params"] = create_dict(model_params)
    global_dict["expert_input"] = expert_input()
    global_dict["generative_model"] = generative_model()
    if type(target_quantities) is tuple:
         global_dict["target_quantities"] = {f"{target_quantities[0]['method'][0]}": target_quantities[0],
                                             f"{target_quantities[1]['method'][0]}": target_quantities[1]}
    else:
        global_dict["target_quantities"] = create_dict(target_quantities)
    global_dict["loss_function"] = loss_function()
    global_dict["optimization_settings"] = optimization_settings()
    global_dict["output_path"] = {
        "data": f"elicit/simulations/results/data/{method}/{sim_id}",
        "plots": f"elicit/simulations/results/plots/{method}/{sim_id}",
        # "data": os.path.join(os.path.dirname(__name__), output_path, "data", method, sim_id),
        # "plots": os.path.join(os.path.dirname(__name__), output_path, "plots", method, sim_id)
    }
    global_dict["print_info"] = print_info
    global_dict["view_ep"] = view_ep

    # save global dict
    path = global_dict["output_path"]["data"] + '/global_dict.pkl'
    save_as_pkl(global_dict, path)

    return global_dict
