import tensorflow_probability as tfp
import tensorflow as tf
import pandas as pd
import sys

tfd = tfp.distributions 

from setup.input_functions import param, model, target, loss, expert, optimization, prior_elicitation
from user_input.custom_functions import Normal_log

def run_simulation(seed, upper_thres):
    #%% Model parameters
    normal_log = Normal_log()
    
    def model_params():  
        return (
            param(name = "b0", 
                  family = normal_log, 
                  hyperparams_dict = {"mu0": tfd.Uniform(1.,2.5), 
                                      "log_sigma0": tfd.Uniform(-2.,-5.)}
                  ),
            param(name = "b1", 
                  family = normal_log, 
                  hyperparams_dict = {"mu1": tfd.Uniform(0.,0.5), 
                                      "log_sigma1": tfd.Uniform(-2.,-5.)}
                  ),
            param(name = "b2", 
                  family = normal_log, 
                  hyperparams_dict = {"mu2": tfd.Uniform(-1.,-1.5), 
                                      "log_sigma2": tfd.Uniform(-2.,-5.)}
                  ),
            param(name = "b3", 
                  family = normal_log, 
                  hyperparams_dict = {"mu3": tfd.Uniform(-0.5,-1.), 
                                      "log_sigma3": tfd.Uniform(-2.,-5.)}
                  )
            )
    
    #%% Expert input
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(2.91, 0.07),
                          "b1": tfd.Normal(0.23, 0.05),
                          "b2": tfd.Normal(-1.51, 0.135),
                          "b3": tfd.Normal(-0.61, 0.105)
                          })
    
    #%% Generative model
    from user_input.generative_models import GenerativePoissonModel
    # from user_input.design_matrices import load_design_matrix_equality
    # design_matrix = load_design_matrix_equality("standardize", selected_obs = [0, 13, 14, 35, 37, 48])

    design_matrix = pd.read_pickle("elicit/simulations/data/design_matrix_pois.pkl")
    
    def generative_model():
        return model(GenerativePoissonModel,
                     additional_model_args = {
                         "total_count": 80, 
                         "design_matrix": design_matrix},
                     discrete_likelihood = True,
                     softmax_gumble_specs = {"temperature": 1.,
                                             "upper_threshold": upper_thres}
                    )
    
    #%% Target quantities and elicited statistics
    from user_input.custom_functions import custom_group_means
    
    def target_quantities():
        return (
            target(name = "ypred",
                   elicitation_method = "histogram",
                   loss_components = "by-group"
                   ),
            target(name = "group_means",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
                    custom_target_function = {
                        "function": custom_group_means,
                        "additional_args": {"design_matrix": design_matrix,
                                            "factor_indices": [0,2,3]
                                            }
                        },
                    loss_components = "by-group"
                    )
            )
    #%% regularizations
    from user_input.custom_functions import custom_target_dist

    def regularizers():
        return None
    #%% Loss function
    def loss_function():
        return loss(loss_function = "mmd-energy",
                    loss_weighting = None
                    )
    
    #%% Training settings
    def optimization_settings():
        return optimization(
                        optimizer = tf.keras.optimizers.Adam,
                        optimizer_specs = {
                            "learning_rate": tf.keras.optimizers.schedules.CosineDecayRestarts(
                                0.01, 50),
                            "clipnorm": 1.0
                            }
                        )
    ##% global method function
    prior_elicitation(
        method = "parametric_prior",
        sim_id = f"pois_{seed}_{upper_thres}",
        B = 128,
        rep = 300,
        seed = seed,
        burnin = 10,
        epochs = 700,
        output_path = "results",
        model_params = model_params,
        expert_input = expert_input,
        generative_model = generative_model,
        target_quantities = target_quantities,
        regularizers = regularizers,
        loss_function = loss_function,
        optimization_settings = optimization_settings,
        log_info = 0,
        print_info=True,
        view_ep = 1
        )

if __name__ == "__main__":
    seed = int(sys.argv[1])
    upper_thres = int(sys.argv[2])
    run_simulation(seed, upper_thres)


# from sim_scripts.plot_learned_prior import learned_prior_pois
# from sim_scripts.plot_diagnostics import diagnostics_pois
# from validation.plot_pois_res import plot_results_overview_param

# path = "elicit/simulations/case_studies/sim_results/"
# file = "pois_34765558"
# selected_obs = [0, 13, 14, 35, 37, 48]
# true_values = [2.91, 0.07, 0.23, 0.05, -1.51, 0.135, -0.61, 0.105]

# # plot diagnostics
# diagnostics_pois(path, file, save_fig = True)

# # plot learned prior distributions
# learned_prior_pois(path, file, selected_obs, true_values, 
#                     last_vals = 30, save_fig = True)

# # plot overview of simulation results (for appendix)
# plot_results_overview_param(path, file, "Poisson model - parametric prior")
