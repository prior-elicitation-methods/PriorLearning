import tensorflow_probability as tfp
import tensorflow as tf
import sys

tfd = tfp.distributions 

from functions.loss_functions import MMD_energy
from setup.input_functions import param, model, target, loss, expert, optimization, prior_elicitation

def run_simulation(seed):
    #%% Model parameters
    from user_input.custom_functions import Normal_log_log
    from user_input.custom_functions import Gamma_log
    from user_input.custom_functions import TruncNormal_log
    
    # initialize truncated normal with boundaries 
    normal_log = Normal_log_log()
    normal_trunc_log = TruncNormal_log(0., 0., 500.)
    gamma_log = Gamma_log()
    
    def model_params():  
        return (
            param(name = "b0",
                  family = normal_log, 
                  hyperparams_dict = {"log_mu0": tfd.Normal(5.,2.), 
                                      "log_sigma0": tfd.Normal(2.,0.5)}),
            param(name = "b1",
                  family = normal_log, 
                  hyperparams_dict = {"log_mu1": tfd.Normal(3.,2.), 
                                      "log_sigma1": tfd.Normal(1.5,0.5)}),
            param(name = "tau0",
                  family = normal_trunc_log, 
                  hyperparams_dict = {"log_omega0": tfd.Normal(3.,0.5)}),
            param(name = "tau1",
                  family = normal_trunc_log, 
                  hyperparams_dict = {"log_omega1": tfd.Normal(3.,0.5)}),
            param(name = "sigma",
                  family = gamma_log, 
                  hyperparams_dict = {"log_concentration": tfd.Normal(5.,0.5), 
                                      "log_rate": tfd.Normal(2.,0.5)})
            )
    #%% Expert input
    def expert_input():
        return expert(data = None,
                      simulate_data = True,
                      simulator_specs = {
                          "b0": tfd.Normal(250.4, 7.27),
                          "b1": tfd.Normal(30.26, 4.82),
                          "tau0": tfd.TruncatedNormal(0., 33., low=0., high=500),
                          "tau1": tfd.TruncatedNormal(0., 23., low=0., high=500),
                          "sigma": tfd.Gamma(200., 8.)
                          })
    #%% Generative model
    from user_input.generative_models import GenerativeMultilevelModel
    from user_input.design_matrices import load_design_matrix_sleep
    
    design_matrix = load_design_matrix_sleep("divide_by_std", N_days = 10, 
                                             N_subj = 200, 
                                             selected_days = [0,2,5,6,9])
    
    def generative_model():
        return model(GenerativeMultilevelModel,
                     additional_model_args = {
                         "design_matrix": design_matrix,
                         "selected_days": [0,2,5,6,9],
                         "alpha_lkj": 1.,
                         "N_subj": 200,
                         "N_days": 5
                         },
                     discrete_likelihood = False
                    )
    
    #%% Target quantities and elicited statistics
    from user_input.custom_functions import custom_mu0_sd, custom_mu9_sd
    from functions.user_interface.create_dictionaries import create_dict
    
    @create_dict
    def target_quantities1(method = "ground_truth"):
        return (
            target(name = "sigma",
                    elicitation_method = "moments",
                    moments_specs=("mean","sd"),
                    loss_components = "all"
                    ),
            target(name = "mu0sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all"
                    ),
            target(name = "mu9sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all"
                    ),
            target(name = "meanperday",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
                    loss_components = "by-group"
                    )
            )
    
    @create_dict
    def target_quantities2(method = "learning"):
        return (
            target(name = "sigma",
                    elicitation_method = "moments",
                    moments_specs=("mean","sd"),
                    loss_components = "all"
                    ),
            target(name = "mu0sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all",
                    custom_target_function={
                                        "function": custom_mu0_sd,
                                        "additional_args": {"selected_days": [0,2,5,6,9]}
                                        }
                    ),
            target(name = "mu9sdcomp",
                    elicitation_method = "histogram",
                    loss_components = "all",
                    custom_target_function={
                                        "function": custom_mu9_sd,
                                        "additional_args": {"selected_days": [0,2,5,6,9]}
                                        }
                    ),
            target(name = "meanperday",
                    elicitation_method = "quantiles",
                    quantiles_specs = (25, 50, 75),
                    loss_components = "by-group"
                    )
            )
    
    #%% regularizations
    from user_input.custom_functions import custom_target_dist
    
    def regularizers():
        return None  
    
    #%% Loss function
    def loss_function():
        return loss(loss_function = MMD_energy,
                    loss_weighting = None
                    )
    
    #%% Training settings
    def optimization_settings():
        return optimization(
                        optimizer = tf.keras.optimizers.Adam,
                        optimizer_specs = {
                            "learning_rate": tf.keras.optimizers.schedules.CosineDecayRestarts(
                                0.005, 100),
                            "clipnorm": 1.0
                            }
                        )
    
    ##% global method function
    prior_elicitation(
        method = "parametric_prior",
        sim_id = f"mlm_{seed}",
        B = 128,
        rep = 200,
        seed = seed,
        epochs = 700,
        output_path = "results",
        burnin = 10,
        model_params = model_params,
        expert_input = expert_input,
        generative_model = generative_model,
        target_quantities = (target_quantities1,target_quantities2),
        regularizers = regularizers,
        loss_function = loss_function,
        optimization_settings = optimization_settings,
        log_info = 0,
        view_ep = 1,
        print_info = True
        )

if __name__ == "__main__":
    seed = int(sys.argv[1])
    run_simulation(seed)

from sim_scripts.plot_learned_prior import learned_prior_multilevel
from sim_scripts.plot_diagnostics import diagnostics_multilevel
from validation.plot_mlm_res import plot_results_overview_param

path = "elicit/simulations/case_studies/sim_results/"
file = "mlm_34765522"
true_values = [250.4, 7.27, 30.26, 4.82, 33., 23., 200., 8.]
selected_obs = [0,2,5,6,9]

# plot diagnostics
diagnostics_multilevel(path, file, save_fig = True)

# plot learned prior distributions
learned_prior_multilevel(path, file, selected_obs, true_values, last_vals = 30, 
                          save_fig = True)

# plot overview of simulation results (for appendix)
plot_results_overview_param(path, file, "Multilevel model - parametric prior")
