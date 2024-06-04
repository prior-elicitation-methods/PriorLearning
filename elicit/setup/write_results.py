import pandas as pd
import tensorflow as tf

def create_output_summary(path_res, global_dict):
    """
    Creates a text summary of all user inputs and some information about the 
    learning process

    Parameters
    ----------
    path_res : str
        path to folder in which the results are saved.
    global_dict : dict
        global dictionary incl. all user specifications as created when call the
        prior_elicitation function.

    Returns
    -------
    txt file
        returns a text file summarizing all input information and some information
        about the learning process (e.g., wall time).

    """
    def summary_targets():
        try: 
            global_dict["target_quantities"]['name']
        except:
            sub_global_dict = global_dict["target_quantities"]["ground_truth"]
            df = pd.DataFrame()
            df["target quantities"] = sub_global_dict["name"]
            df["elicitation technique"] = sub_global_dict["elicitation_method"]
            df["combine-loss"] = sub_global_dict["loss_components"]
            
            sub_global_dict = global_dict["target_quantities"]["learning"]
            df2 = pd.DataFrame()
            df2["target quantities"] = sub_global_dict["name"]
            df2["elicitation technique"] = sub_global_dict["elicitation_method"]
            df2["combine-loss"] = sub_global_dict["loss_components"]
            return str(f"expert:\n===== \n{df}\nlearning:\n===== \n{df2}\n")
        else:
            sub_global_dict = global_dict["target_quantities"]
            df = pd.DataFrame()
            df["target quantities"] = sub_global_dict["name"]
            df["elicitation technique"] = sub_global_dict["elicitation_method"]
            df["combine-loss"] = sub_global_dict["loss_components"]
            return df
    
    loss_comp = pd.read_pickle(path_res+"/loss_components.pkl")
    
    df2 = pd.DataFrame()
    df2["loss components"] = list(loss_comp.keys())
    df2["shape"] = [list(loss_comp[key].shape) for key in list(loss_comp.keys())]
    
    time = tf.reduce_sum(pd.read_pickle(path_res+"/final_results.pkl")["time_epoch"])/60.
    min, sec = tuple(f"{time:.2f}".split("."))
    
    optimizer_dict = {}
    if global_dict['optimization_settings']['optimizer_specs']['learning_rate']._keras_api_names[0] == 'keras.optimizers.schedules.CosineDecayRestarts':
        optimizer_dict["lr_scheduler"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate']._keras_api_names[0]
        optimizer_dict["init_lr"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate'].initial_learning_rate
        optimizer_dict["decay_steps"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate'].first_decay_steps
    if global_dict['optimization_settings']['optimizer_specs']['learning_rate']._keras_api_names[0] == 'keras.optimizers.schedules.ExponentialDecay':
        optimizer_dict["lr_scheduler"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate']._keras_api_names[0]
        optimizer_dict["init_lr"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate'].initial_learning_rate
        optimizer_dict["decay_rate"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate'].decay_rate
        optimizer_dict["decay_steps"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate'].decay_steps
    else:
        optimizer_dict["lr_scheduler"]=global_dict['optimization_settings']['optimizer_specs']['learning_rate']
    
    if global_dict['method'] == "deep_prior":
        def method_settings():
            return str("\nNormalizing Flow"+
            "\n---------------- \n"+
            f"number coupling layers={global_dict['model_params']['normalizing_flow_specs']['num_coupling_layers']}\n"+
            f"coupling design={global_dict['model_params']['normalizing_flow_specs']['coupling_design']}\n"+
            f"units in dense layer={global_dict['model_params']['normalizing_flow_specs']['coupling_settings']['dense_args']['units']}\n"+
            f"activation function={global_dict['model_params']['normalizing_flow_specs']['coupling_settings']['dense_args']['activation']}\n"+
            f"number dense layers={global_dict['model_params']['normalizing_flow_specs']['coupling_settings']['num_dense']}\n"+
            f"permutation={global_dict['model_params']['normalizing_flow_specs']['permutation']}\n")
    else:
        def method_settings():
            family_list = [global_dict['model_params']['family'][i].name for i in range(len(global_dict['model_params']['family']))]
            family_dict = {f"{param}": family_list[i] for i, param in enumerate(global_dict['model_params']['name'])}
            init_dict = {}
            for i in range(len(global_dict['model_params']['family'])):
                for j,key in enumerate(global_dict['model_params']['hyperparams_dict'][i].keys()):
                    init_info = global_dict['model_params']['hyperparams_dict'][i][key].parameters.copy()
                    init_info.pop("validate_args", None) 
                    init_info.pop("allow_nan_stats", None)
                    
                    init_dict[list(global_dict['model_params']['hyperparams_dict'][i].keys())[j]] = init_info
           
            return str("\nParametric Prior"+"\n---------------- \n"+
                       f"distribution family={family_dict}\n"+
                       f"initialization={init_dict}\n")
    
    output_summary = str(
        "General summary"+
        "\n---------------- \n"+
        f"method={global_dict['method']}\n"+
        f"sim_id={global_dict['sim_id']}\n"+
        f"seed={global_dict['seed']}\n"+
        f"B={global_dict['B']}\n"+
        f"rep={global_dict['rep']}\n"+
        f"epochs={global_dict['epochs']}\n"+
        f"wall time={min}:{sec} (min:sec)\n"+
        f"optimizer={global_dict['optimization_settings']['optimizer']}\n"+
        f"learning rate={optimizer_dict}\n"+
        "\nModel info"+
        "\n---------------- \n"+
        f"model name={global_dict['generative_model']['model_function']}\n"+
        f"model parameters={global_dict['model_params']['name']}\n"+
        method_settings()+
        "\nTarget quantities and elicitation techniques"+
            "\n--------------------- \n"+
            f"\n{summary_targets()}\n"+
            "\nLoss components"+
            "\n--------------------- \n"+
            f"\n{df2}"
            )
    return output_summary

def write_results(path_res, global_dict):
    """
    saves the summary of the user inputs in a text file

    Parameters
    ----------
    path_res : str
        path to location where results are saved.
    global_dict : dict
        dictionary containing all user specifications.

    """
    f = open(path_res+"\overview.txt", "w")
    output_summary = create_output_summary(path_res,global_dict)
    f.write(output_summary)
    f.close()
    
def model_summary(path_res, global_dict):
    """
    Prints the summary output without saving it to a particular location.

    Parameters
    ----------
    path_res : str
        path to location where results are saved.
    global_dict : dict
        global dictionary containing all user specifications.

    """
    print(create_output_summary(path_res, global_dict))