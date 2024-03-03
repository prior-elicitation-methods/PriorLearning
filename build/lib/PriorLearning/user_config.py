# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:44:44 2023

@author: flobo
"""

def target_config(target, elicitation, combine_loss, 
                  custom_target_function = None, internal_loss = None,
                  **kwargs):  

    # get the user input
    user_input = locals()
    # extract additional arguments via the kwargs argument
    opt_args_dict = user_input["kwargs"]
    keys_args = list(opt_args_dict.keys())
    
    # create the config dictionary as copy of the user input
    # without the kwargs dict
    config_dict = user_input.copy()
    config_dict.pop("kwargs")
    
    # append the config_dict by the extracted additional arguments
    if len(keys_args) != 0:
        for i in range(len(keys_args)):
            config_dict[keys_args[i]] = opt_args_dict[keys_args[i]]
    
    # list all allowed values per argument 
    # if list is not finite use None and test whether the input type is correct
    arg_values = dict(
        target = [None, str],
        elicitation = ["histogram","quantiles","moments"],
        combine_loss = [None,"all","by-stats","by-group","by-target-type"],
        moments_specs = ["mean", "sd", "variance"],
        quantiles_specs = list(range(101))
     #   custom_target_name = [None, str],
    )
    
    # check that additionally required arguments, conditional on selected values
    # of previous arguments are specified correctly
    if config_dict["elicitation"] == "moments":
        if "moments_specs" not in list(config_dict.keys()): 
            raise NameError(f"The elicitation method 'moments' requires an additional argument named 'moments_specs' which is not specified. Currently supported are {arg_values['moments_specs']}.")
        if type(config_dict["moments_specs"]) != tuple:
            raise ValueError(f"Specification of moments has to be given in form of a tuple. Got {type(config_dict['moments_specs'])}")
            
    if config_dict["elicitation"] == "quantiles":
        if "quantiles_specs" not in list(config_dict.keys()): 
            raise NameError("The elicitation method 'quantiles' requires an additional argument named 'quantiles_specs' which is a tuple of queried quantiles, e.g., (25, 50, 75).")
        if type(config_dict["quantiles_specs"]) != tuple:
            raise ValueError(f"Specification of quantiles has to be given in form of a tuple. Got {type(config_dict['quantiles_specs'])}")
            
    if config_dict["target"] == "custom_target":
        if "custom_target_name" not in list(config_dict.keys()): 
            raise NameError("If custom target is provided, the additional argument 'custom_target_name' is required. It should be identical to the name of the custom target quantity as declared in the targets dictionary of the generative model.")
       
    # specification of first six arguments is obligatory
    # for n in list(config_dict.keys())[0:6]:
    #     # check that obligatory arguments are specified
    #     if config_dict[n] is None: 
    #         if arg_values[n][0] is None:
    #             arg_values[n] = "specified"
    #         raise ValueError(f"The argument {n} must be {arg_values[n]} but is None.")
    
    # check that values per arguments are of correct form
    for k,v in zip(config_dict.keys(),config_dict.values()):
        # if values are not already lists, create lists
        if ((type(v) != list) and (type(v) != tuple)):
            v = [v]
        # # raise an error if values are not among allowed values 
        # if arg_values[k][0] is None and type(v[0]) != arg_values[k][1]:
        #     raise TypeError(f"{v[0]} must be of type {arg_values[k][1]}, but is {type(v[0])}.")
        # if (arg_values["target"] != "custom_target" and 
        #     arg_values[k][0] is not None and
        #     not(set(v).issubset(set(arg_values[k])))):
        #     if k == "quantiles_specs":
        #         raise ValueError(f"The argument quantiles_specs takes a tuple of numbers within [0,100]. Got {v}.")
        #     else:
        #         raise ValueError(f"The argument {k} must be {arg_values[k]} but is {v}.")
         
        config_dict[k] = v
   
    for k,v in zip(keys_args, opt_args_dict.values()):
        config_dict[k] = [v]

    return config_dict


def target_input(*args):
    user_input = locals()
    components = user_input["args"]
    
    info_dict = dict()
    for i in range(len(components)):
        for k,v in zip(components[i].keys(),components[i].values()):
            if k in info_dict.keys():
                # tuple is saved as list otherwise it would be concatenated
                # if multiple quantiles are provided and it is not any more 
                # possible to differentiate which set of quantiles was given 
                # for which target quantity
                if type(components[i][k]) == tuple:
                    info_dict[k] = info_dict[k]+[components[i][k]]
                else:
                    info_dict[k] = info_dict[k]+components[i][k]
            else:
                if type(v) == tuple:
                    info_dict[k] = [v]
                else:
                    info_dict[k] = v
    
    return info_dict




