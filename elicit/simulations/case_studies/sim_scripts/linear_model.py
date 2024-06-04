import tensorflow_probability as tfp
import tensorflow as tf
import sys
tfd = tfp.distributions 

from setup.input_functions import param, model, target, loss, expert, optimization, prior_elicitation

def run_simulation(seed):
    #%% Model parameters
    from user_input.custom_functions import Normal_log
    from user_input.custom_functions import Gamma_log
    
    # initialize truncated normal with boundaries 
    normal_log = Normal_log()
    gamma_log = Gamma_log()
    
    def model_params():  
        return (
            param(name = "b0", 
                  family = normal_log, 
                  hyperparams_dict = {"mu0": tfd.Normal(0.,0.1), 
                                      "log_sigma0": tfd.Uniform(-2.,-4.)}
                  ),
            param(name = "b1", 
                  family = normal_log, 
                  hyperparams_dict = {"mu1": tfd.Normal(0.,0.1), 
                                      "log_sigma1": tfd.Uniform(-2.,-4.)}
                  ),
            param(name = "b2", 
                  family = normal_log, 
                  hyperparams_dict = {"mu2": tfd.Normal(0.,0.1), 
                                      "log_sigma2": tfd.Uniform(-2.,-4.)}
                  ),
            param(name = "b3", 
                  family = normal_log, 
                  hyperparams_dict = {"mu3": tfd.Normal(0.,0.1), 
                                      "log_sigma3": tfd.Uniform(-2.,-4.)}
                  ),
            param(name = "b4", 
                  family = normal_log, 
                  hyperparams_dict = {"mu4": tfd.Normal(0.,0.1), 
                                      "log_sigma4": tfd.Uniform(-2.,-4.)}
                  ),
            param(name = "b5", 
                  family = normal_log, 
                  hyperparams_dict = {"mu5": tfd.Normal(0.,0.1), 
                                      "log_sigma5": tfd.Uniform(-2.,-4.)}
                  ),
            param(name = "sigma", 
                  family = gamma_log, 
                  hyperparams_dict = {"log_concentration": tfd.Normal(3.,0.1), 
                                      "log_rate": tfd.Normal(5.,0.1)}
                  )
            )
    
    #%% Expert input
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(0.12, 0.02),
                          "b1": tfd.Normal(0.15, 0.02),
                          "b2": tfd.Normal(-0.02, 0.06),
                          "b3": tfd.Normal(-0.03, 0.06),
                          "b4": tfd.Normal(-0.02, 0.03),
                          "b5": tfd.Normal(-0.04, 0.03),
                          "sigma": tfd.Gamma(20., 200.)
                          })
    
    #%% Generative model
    from user_input.generative_models import GenerativeNormalModel_param
    from user_input.design_matrices import load_design_matrix_truth
    
    design_matrix = load_design_matrix_truth(n_group=60)
    
    def generative_model():
        return model(GenerativeNormalModel_param,
                     additional_model_args = {
                         "design_matrix": design_matrix},
                     discrete_likelihood = False
                    )
    
    #%% Target quantities and elicited statistics
    def target_quantities():
        return (
            target(name = "marginal_ReP",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
                    loss_components = "by-group"
                    ),
            target(name = "marginal_EnC",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
                    loss_components = "by-group"
                    ),
            target(name = "mean_effects",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
                    loss_components = "by-group"
                    ),
            target(name = "R2",
                    elicitation_method = "histogram",
                    loss_components = "all"
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
                                0.001, 50),
                            "clipnorm": 1.0
                            }
                        )
    
    ##% global method function
    prior_elicitation(
        method = "parametric_prior",
        sim_id = f"norm_{seed}",
        B = 128,
        rep = 300,
        seed = seed,
        epochs = 1000,
        burnin = 10,
        output_path = "results",
        model_params = model_params,
        expert_input = expert_input,
        generative_model = generative_model,
        target_quantities = target_quantities,
        regularizers = regularizers,
        loss_function = loss_function,
        optimization_settings = optimization_settings,
        log_info = 0,
        view_ep = 1,
        print_info=True
        )

if __name__ == "__main__":
    seed = int(sys.argv[1])
    run_simulation(seed)


from sim_scripts.plot_learned_prior import learned_prior_linear, elicited_statistics_normal
from sim_scripts.plot_diagnostics import diagnostics_linear
from validation.plot_norm_res import plot_results_overview_param

path = "elicit/simulations/case_studies/sim_results/"
file = "norm_34765771"
true_values = [0.12, 0.02, 0.15, 0.02, -0.02, 0.06, -0.03, 0.06,
                -0.02, 0.03, -0.04, 0.03, 20., 200.]

# plot diagnostics
diagnostics_linear(path, file, save_fig = True)

# plot elicited statistics according to ground truth
elicited_statistics_normal(path, file, save_fig = True)

# plot learned prior distributions
learned_prior_linear(path, file, true_values, 
                      last_vals = 30, save_fig = True)

# plot overview of simulation results (for appendix)
plot_results_overview_param(path, file, "Normal model - parametric prior")
