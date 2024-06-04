import pandas as pd
import tensorflow as tf

from functions.helper_functions import save_as_pkl

def save_hyperparameters(generator, epoch, global_dict):
    saving_path = global_dict["output_path"]["data"]
    if epoch == 0:  
        # prepare list for saving hyperparameter values
        hyperparams = global_dict["model_params"]["hyperparams_dict"]
        hyp_list = dict()
        for d in hyperparams:
            hyp_list.update(d)
        # create a dict with empty list for each hyperparameter
        res_dict = dict()
        for key in hyp_list.keys():
            if key.startswith("log_"):
                key = key.removeprefix("log_")
            res_dict[f"{key}"] = [] 
    else:
        path_res_dict = saving_path+'/res_dict.pkl'
        res_dict = pd.read_pickle(rf"{path_res_dict}")
    hyperparams = generator.trainable_variables
    vars_values = [hyperparams[i].numpy().copy() for i in range(len(hyperparams))]
    vars_names = [hyperparams[i].name[:-2] for i in range(len(hyperparams))]
    for val, name in zip(vars_values, vars_names):
        res_dict[name].append(val)
    # save result dictionary
    path_res_dict = saving_path+'/res_dict.pkl'
    save_as_pkl(res_dict, path_res_dict)
    return res_dict


def marginal_prior_moments(prior_samples, epoch, global_dict):
    saving_path = global_dict["output_path"]["data"]
    if epoch == 0:
        res_dict = {"means": [], "stds": []}
    else:
        path_res_dict = saving_path+'/res_dict.pkl'
        res_dict = pd.read_pickle(rf"{path_res_dict}")

    means = tf.reduce_mean(prior_samples, (0,1))
    sds = tf.reduce_mean(tf.math.reduce_std(prior_samples, 1), 0)
    for val,name in zip([means,sds],["means", "stds"]):
        res_dict[name].append(val)
    # save result dictionary
    path_res_dict = saving_path+'/res_dict.pkl'
    save_as_pkl(res_dict, path_res_dict)
    return res_dict