import tensorflow as tf
import tensorflow_probability as tfp
import bayesflow as bf

tfd = tfp.distributions
bfn = bf.networks

from functions.helper_functions import save_as_pkl


def compute_loss_components(elicited_statistics, global_dict, expert):
    """
    Computes the single loss components used for computing the discrepancy
    between the elicited statistics. This computation depends on the
    method as specified in the 'combine-loss' argument.

    Parameters
    ----------
    elicited_statistics : dict
        dictionary including the elicited statistics.
    global_dict : dict
        dictionary including all user-input settings.
    expert : bool
        if workflow is run to simulate a pre-specified ground truth; expert is
        set as 'True'. As consequence the files are saved in a special 'expert'
        folder.

    Returns
    -------
    loss_component_res : dict
        dictionary including all loss components which will be used to compute
        the discrepancy.

    """
    # check whether the simulated ground truth has a different dict of the 
    # target quantities. If yes select the correct target quantity dict
    try: 
        global_dict["target_quantities"]['name']
    except:
        sub_global_dict = global_dict["target_quantities"]["learning"]
    else:
        sub_global_dict = global_dict["target_quantities"]
    
    # extract names from elicited statistics
    name_elicits = list(elicited_statistics.keys())
    # prepare dictionary for storing results
    loss_component_res = dict()
    # initialize some helpers for keeping track of target quantity
    target_control = []
    i_target = 0
    eval_target = True
    # loop over elicited statistics
    for i, name in enumerate(name_elicits):
        # get name of target quantity
        target = name.split(sep = "_")[-1]
        if i != 0:
            # check whether elicited statistic correspond to same target quantity
            eval_target = target_control[-1] == target
        # append current target quantity
        target_control.append(target)
        # if target quantity changes go with index one up
        if not eval_target:
            i_target += 1
        # extract loss component 
        loss_component = elicited_statistics[name]
       
        if tf.rank(loss_component) == 1:
            assert sub_global_dict["loss_components"][i_target] == "all", f"the elicited statistic {name} has rank=1 and can therefore support only combine_loss = 'all'"
            # add a last axis for loss computation
            final_loss_component = tf.expand_dims(loss_component, axis = -1)
            # store result
            loss_component_res[f"{name}_loss"] = final_loss_component
        
        else:
            if sub_global_dict["loss_components"][i_target] == "all":
                assert tf.rank(loss_component) <= 3, f"the elicited statistic {name} has more than 3 dimensions; combine_loss = all is therefore not possible. Consider using combine_loss = 'by-group'"
                if tf.rank(loss_component) == 3:
                    loss_component_res[f"{name}_loss_{i_target}"] =  tf.reshape(loss_component, (loss_component.shape[0], loss_component.shape[1]*loss_component.shape[2]))
                if tf.rank(loss_component) <= 2:
                    loss_component_res[f"{name}_loss_{i_target}"] = loss_component
                
            
            if sub_global_dict["loss_components"][i_target] == "by-stats":
                assert sub_global_dict["elicitation_method"][i_target] == "quantiles", "loss combination method 'by-stats' is currently only possible for elicitation techniques: 'quantiles'."
                for j in range(loss_component.shape[1]):
                    if tf.rank(loss_component) == 2:
                        loss_component_res[f"{name}_loss_{j}"] = loss_component[:,j]
                    if tf.rank(loss_component) == 3:
                        loss_component_res[f"{name}_loss_{j}"] = loss_component[:,j,:]
                        
            if sub_global_dict["loss_components"][i_target] == "by-group":
                for j in range(loss_component.shape[-1]):
                    final_loss_component = loss_component[...,j]
                    if tf.rank(final_loss_component) == 1:
                        final_loss_component = tf.expand_dims(final_loss_component, axis = -1)

                    loss_component_res[f"{name}_loss_{j}"] = final_loss_component
                    
    # save file in object
    saving_path = global_dict["output_path"]["data"]
    if expert:
        saving_path = saving_path+"/expert"
    path = saving_path+'/loss_components.pkl'
    save_as_pkl(loss_component_res, path)
    # return results                
    return loss_component_res


def dynamic_weight_averaging(epoch, loss_per_component_current, 
                             loss_per_component_initial, 
                             task_balance_factor,
                             saving_path):
    """DWA determines the weights based on the learning speed of each component
    
    The Dynamic Weight Averaging (DWA) method proposed by 
    Liu, Johns, & Davison (2019) determines the weights based on the learning 
    speed of each component, aiming to achieve a more balanced learning process. 
    Specifically, the weight of a component exhibiting a slower learning speed 
    is increased, while it is decreased for faster learning components.
    
    Liu, S., Johns, E., & Davison, A. J. (2019). End-To-End Multi-Task Learning With Attention. In
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 1871–1880).
    doi: https://doi.org/10.1109/CVPR.2019.00197

    Parameters
    ----------
    epoch : int
        How often should the hyperparameter values be updated?
    loss_per_component_current : list of floats
        List of loss values per loss component for the current epoch.
    loss_per_component_initial : list of floats
        List of loss values per loss component for the initial epoch (epoch = 0).
    task_balance_factor : float
        temperature parameter that controls the softness of the loss weighting 
        in the softmax operator. Setting the temperature 𝑎 to a large value 
        results in the weights approaching unity.

    Returns
    -------
    total_loss : float
        Weighted sum of all loss components. Loss used for gradient computation.
    weight_loss_component : list of floats
        List of computed weight values per loss component for current epoch.

    """
    # get number of loss components
    num_loss_components = len(loss_per_component_current)

    # initialize weights
    if epoch < 2:
        rel_weight_descent = tf.ones(num_loss_components)
    # w_t (epoch-1) = L_t (epoch-1) / L_t (epoch_0)
    else:
        rel_weight_descent = tf.math.divide(loss_per_component_current, 
                                            loss_per_component_initial)
    
    # T*exp(w_t(epoch-1)/a)
    numerator = tf.math.multiply(
        tf.cast(num_loss_components, dtype = tf.float32), 
        tf.exp(tf.math.divide(rel_weight_descent, task_balance_factor)))

    # softmax operator
    weight_loss_component = tf.math.divide(numerator, 
                                           tf.math.reduce_sum(numerator))
    
    # total loss: L = sum_t lambda_t*L_t
    weighted_total_loss = tf.math.reduce_sum(tf.math.multiply(weight_loss_component, 
                                                              loss_per_component_current)) 
    # save file in object
    path = saving_path+'/weighted_total_loss.pkl'
    save_as_pkl(weighted_total_loss, path)
    return weighted_total_loss


def compute_discrepancy(loss_components_expert, loss_components_training,
                        global_dict):
    """
    Computes the discrepancy between all loss components using a specified
    discrepancy measure and returns a list with all loss values.

    Parameters
    ----------
    loss_components_expert : dict
        dictionary including all loss components derived from the expert-elicited
        statistics.
    loss_components_training : dict
        dictionary including all loss components derived from the model 
        simulations. (The names (keys) between loss_components_expert and loss_components_training must match)
    global_dict : dict
        dictionary including all user-input settings.

    Returns
    -------
    loss_per_component : list
        list of loss value for each loss component

    """
    # import loss function 
    loss_function = global_dict["loss_function"]["loss_function"]
    # create dictionary for storing results
    loss_per_component = []
    # compute discrepancy 
    for name in list(loss_components_expert.keys()):
        # broadcast expert loss to training-shape
        loss_comp_expert = tf.broadcast_to(loss_components_expert[name], 
                                           shape=loss_components_training[name].shape)
        # compute loss
        loss_per_component.append(loss_function(loss_comp_expert, 
                                                loss_components_training[name]))
    
    # save file in object
    saving_path = global_dict["output_path"]["data"]
    path = saving_path+'/loss_per_component.pkl'
    save_as_pkl(loss_per_component, path)
    return loss_per_component