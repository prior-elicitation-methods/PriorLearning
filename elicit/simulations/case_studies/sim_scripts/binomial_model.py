import tensorflow_probability as tfp
import tensorflow as tf
import sys
import pandas as pd

tfd = tfp.distributions 

from setup.input_functions import param, model, target, loss, expert, optimization, prior_elicitation
from user_input.custom_functions import Normal_log

def run_simulation(seed):
    #%% Model parameters
    normal_log = Normal_log()
    
    def model_params():  
        return (
            param(name = "b0", 
                  family = normal_log, 
                  hyperparams_dict = {"mu0": tfd.Normal(0.,1.), 
                                      "log_sigma0": tfd.Uniform(-2.,-3.)}
                  ),
            param(name = "b1", 
                  family = normal_log, 
                  hyperparams_dict = {"mu1": tfd.Normal(0.,1.), 
                                      "log_sigma1": tfd.Uniform(-2.,-3.)}
                  )
            )
    
    #%% Expert input
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(-0.51, 0.06),
                          "b1": tfd.Normal(0.26, 0.04)
                          })
    
    #%% Generative model
    from user_input.generative_models import GenerativeBinomialModel
    # from user_input.design_matrices import load_design_matrix_haberman
    # design_matrix = load_design_matrix_haberman("standardize", 
    #                                            [0, 5, 10, 15, 20, 25, 30])
    design_matrix = pd.read_pickle("elicit/simulations/data/design_matrix_binom.pkl")

    def generative_model():
        return model(GenerativeBinomialModel,
                     additional_model_args = {
                         "total_count": 31, 
                         "design_matrix": design_matrix},
                     discrete_likelihood = True,
                     softmax_gumble_specs = {"temperature": 1.,
                                             "upper_threshold": 31}
                    )
    
    #%% Target quantities and elicited statistics
    def target_quantities():
        return (
            target(name = "ypred",
                   elicitation_method = "quantiles",
                   quantiles_specs = (25, 50, 75),
                   loss_components = "by-group"
                   ),
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
    
    #%% global method function
    prior_elicitation(
        method = "parametric_prior",
        sim_id = f"binom_{seed}",
        B = 128,
        rep = 300,
        seed = seed,
        burnin = 10,
        epochs = 1000,
        output_path = "results",
        model_params = model_params,
        expert_input = expert_input,
        generative_model = generative_model,
        target_quantities = target_quantities,
        regularizers = regularizers,
        loss_function = loss_function,
        optimization_settings = optimization_settings,
        log_info = 0,
        print_info = True,
        view_ep = 1
        )
    
if __name__ == "__main__":
    seed = int(sys.argv[1])
    
    run_simulation(seed)


from sim_scripts.plotting.plot_learned_prior import learned_prior_binom
from sim_scripts.plotting.plot_diagnostics import diagnostics_binom
from sim_scripts.plotting.plot_res_overview import res_overview_binom

path = "elicit/simulations/case_studies/sim_results/"
file = "binom_34764831"
selected_obs = [0, 5, 10, 15, 20, 25, 30]
true_values = [-0.51, 0.06, 0.26, 0.04]

# plot diagnostics
diagnostics_binom(path, file, save_fig = True)

# plot learned prior distributions
learned_prior_binom(path, file, selected_obs, true_values, 
                    last_vals = 30, save_fig = True)

# plot overview of simulation results (for appendix)
res_overview_binom(path, file, selected_obs, "Binomial model - parametric prior")
