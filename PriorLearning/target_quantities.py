"""
Computation of target quantities from samples of the generative model.

"""
import tensorflow as tf
import tensorflow_probability as tfp
import inspect

tfd = tfp.distributions

from MakeMyPrior.helper_functions import group_obs


def method_grand_mean(ypred):
    return tf.reduce_mean(ypred, -1)


def method_group_means(ypred, dmatrix_fct, cmatrix, **kwargs):
    # reshape tf.Tensor from (B, rep, Nobs) to (B, rep, Ngr, num_gr)
    samples_grouped = group_obs(ypred, dmatrix_fct, cmatrix)
    # average over individual obs (i.e., Ngr)
    group_mean = tf.reduce_mean(samples_grouped, axis=2)
    # return group means
    return group_mean


def method_R2(ypred, epred):
    R2 = tf.math.divide(
        tf.math.reduce_variance(epred, -1),
        tf.math.reduce_variance(ypred, -1),
    )
    return R2

class TargetQuantities(tf.Module):
    def __init__(self):
        super(TargetQuantities).__init__()

    def __call__(self, simulations, target_info, **kwargs):
        ypred = simulations["ypred"]
        epred = simulations["epred"]
        
        user_input = locals()
        opt_args = user_input["kwargs"]
        
        #assert type(targets_list) == list, "targets_list must be a list. It is possible to pass [None], if no transformation is desired."
        
        target_dict = dict()
        for i, target in enumerate(target_info["target"]):
           
            if target_info["custom_target_function"][i] is not None:
                
                get_function_args = str(inspect.signature(target_info["custom_target_function"][i])).removeprefix("(").removesuffix(")").split(",")
                function_args = [get_function_args[i].removeprefix(" ") for i in range(len(get_function_args))]
                if "**kwargs" in function_args: 
                    function_args = set(function_args).difference(set(["**kwargs"]))
         
                simulation_subset = dict()
                for argument in function_args:
                    simulation_subset[argument] = simulations[argument]
                
                target_dict[target] = target_info["custom_target_function"][i](**simulation_subset)
           
            if target == "y_obs" and target_info["custom_target_function"][i] is None:
                target_dict[target] = ypred
            
            if target == "R2" and target_info["custom_target_function"][i] is None:
                target_dict[target] = method_R2(ypred, epred)
            
            if target == "group_means" and target_info["custom_target_function"][i] is None:
                
                if not set(["dmatrix_fct", "cmatrix"]).issubset(set(opt_args.keys())):
                    raise ValueError("The elicitation method 'group_means' requires the additional arguments dmatrix_fct (i.e., design matrix incl. only factors) and cmatrix (i.e., contrast matrix).")
                
                target_dict[target] = method_group_means(ypred, **kwargs)

            if target == "grand_mean" and target_info["custom_target_function"][i] is None:
                target_dict[target] = method_grand_mean(ypred)
                
            if (not(target in set(["y_obs", "R2", "group_means", "grand_mean"])) and
                target_info["custom_target_function"][i] is None):
                target_dict[target] = simulations[target]
             
        return target_dict
    
    


