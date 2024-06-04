import os
#os.chdir('/home/flob/prior_elicitation')
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
import seaborn as sns

########### Convergence diagnostics ###########
## % loss
def plot_loss(global_dict, save_fig = False):
    res_dict = pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")

    loss = tf.stack(res_dict["loss"], 0)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    plt.plot(range(global_dict["epochs"]), loss, lw = 2)
    plt.title("loss function", ha = "left", x = 0)
    axs.set_ylabel("loss")
    axs.set_xlabel("epochs")
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/loss.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

## % gradients
def plot_gradients(global_dict, save_fig = False):
    res_dict = pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")

    gradients = tf.stack(res_dict["gradients"], 0)
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    [sns.scatterplot(x=range(global_dict["epochs"]), y=gradients[:,i], 
                    marker="+", label = fr"$\lambda_{{{i}}}$") for i in range(gradients.shape[-1])]
    plt.legend(labelspacing = 0.2, columnspacing = 1, ncols = 2, handletextpad = 0.3, fontsize = "small")
    axs.set_xlabel("epochs")
    axs.set_ylabel("gradient")
    plt.title("gradients", ha = "left", x = 0)
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/gradients.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()
  
def plot_convergence(true_hyperparams, names_hyperparams, global_dict, file_name,
                     pos_only = False, save_fig = False):    
    # prepare hyperparameter values
    res_dict = pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")
    selected_hyppar = {key: res_dict["hyperparameter"][key] for key in names_hyperparams}  
    learned_hyperparams = tf.stack(list(selected_hyppar.values()), -1)
    
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    if pos_only:
        learned_hyperparams = tf.exp(learned_hyperparams)
        
    # learned values
    [axs.plot(range(learned_hyperparams.shape[0]), learned_hyperparams[:,i], lw = 2, 
              label = names_hyperparams[i]) for i in range(learned_hyperparams.shape[1])]
    # expert
    [axs.axhline(true_hyperparams[i], linestyle = "dashed", color = "black", 
                 lw = 1) for i in range(learned_hyperparams.shape[1])]
    # legend and axes
    axs.legend(labelspacing = 0.2, columnspacing = 1, ncols = 2, handletextpad = 0.3, 
                fontsize = "small", handlelength = 0.5)
    axs.set_xlabel("epochs") 
    axs.set_ylabel(r"$\lambda$")
    plt.suptitle("convergence of hyperparameters")
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+f'/{file_name}.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

def plot_convergence_deep(true_hyperparams, moment, global_dict, file_name,
                          save_fig = False):    
    # prepare hyperparameter values
    res_dict = pd.read_pickle(global_dict["output_path"]["data"]+"/final_results.pkl")
    selected_hyppar = res_dict["hyperparameter"][moment] 
    learned_hyperparams = tf.stack(selected_hyppar, 0)
    
    _, axs = plt.subplots(1,1, constrained_layout = True, figsize = (4,3))
    # learned values
    [axs.plot(range(learned_hyperparams.shape[0]), learned_hyperparams[:,i], lw = 2, 
              label = f"{moment}{i}") for i in range(learned_hyperparams.shape[1])]
    # expert
    [axs.axhline(true_hyperparams[i], linestyle = "dashed", color = "black", 
                 lw = 1) for i in range(learned_hyperparams.shape[1])]
    # legend and axes
    axs.legend(labelspacing = 0.2, columnspacing = 1, ncols = 2, handletextpad = 0.3, 
                fontsize = "small", handlelength = 0.5)
    axs.set_xlabel("epochs") 
    axs.set_ylabel(r"$\lambda$")
    plt.suptitle("convergence of hyperparameters")
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+f'/{file_name}.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()


########### Learned prior distributions: Marginals ###########
def plot_marginal_priors(global_dict, sims = 100, save_fig = False):
    truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
    learned = pd.read_pickle(global_dict["output_path"]["data"]+"/prior_samples.pkl")
    
    name_params = global_dict["model_params"]["name"]
    num_params = len(name_params)
    
    _, axs = plt.subplots(1,num_params, constrained_layout = True, 
                          figsize = (int(num_params*2),3))
    for b in range(sims):
        [sns.kdeplot(learned[b,:,i], lw = 2, alpha = 0.2, color = "orange", 
                     ax = axs[i]) for i in range(truth.shape[-1])]
    [sns.kdeplot(truth[0,:,i], lw = 2, color = "black", linestyle = "dashed", 
                 ax = axs[i]) for i in range(truth.shape[-1])]
    axs[0].set_xlabel(r"model parameters $\beta$")
    axs[0].set_ylabel("density")
    [axs[i].set_title(name_params[i]) for i in range(num_params)]
    plt.suptitle("learned prior distributions")
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/marginal_prior.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

########### Learned prior distributions: Joint ###########

def plot_joint_prior(global_dict, save_fig = False):
    truth = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/prior_samples.pkl")
    learned = pd.read_pickle(global_dict["output_path"]["data"]+"/prior_samples.pkl")

    num_params = len(global_dict["model_params"]["name"])
    # prepare data as pandas data frame
    def prep_data(dat, model):
        prepare_data = pd.DataFrame(dat, columns = [fr"$\theta_{{{i}}}$" for i in range(num_params)])
        prepare_data.insert(2, "model", [model]*global_dict["rep"])
        return prepare_data
    frames = [prep_data(truth[0,:,:], "expert"),
              prep_data(learned[0,:,:], "training")]
    df = pd.concat(frames)
    # plot data 
    g = sns.pairplot(df, hue="model", plot_kws=dict(marker="+", linewidth=1), 
                    height=2, aspect = 1.)
    g.map_lower(sns.kdeplot)
    labels = g._legend_data.keys()
    sns.move_legend(g, "upper center",bbox_to_anchor=(0.45, 1.1), 
                    labels=labels, ncol=num_params, title=None, frameon=True)
    if save_fig:
        path_to_file = global_dict["output_path"]["plots"]+'/joint_prior.png'
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        plt.savefig(path_to_file)
    else:
        plt.show()

########### Elicited statistics ###########
def plot_elicited_statistics(global_dict, sims = 100, selected_obs = None,
                             save_fig = False):
    learned_elicits = pd.read_pickle(global_dict["output_path"]["data"]+"/elicited_statistics.pkl")
    true_elicits = pd.read_pickle(global_dict["output_path"]["data"]+"/expert/elicited_statistics.pkl")

    keys_elicit = list(learned_elicits.keys())
    methods = [keys_elicit[i].split(sep="_")[0] for i in range(len(keys_elicit))]
    for i, (key, meth) in enumerate(zip(keys_elicit, methods)):
        training_data = learned_elicits[key]
        expert_data = true_elicits[key]

        if meth == "histogram":
            if tf.rank(tf.squeeze(training_data)) == 1:
                _, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3,3))
                sns.histplot(training_data, bins = 20, color = "orange", stat="density",
                            ax = ax, edgecolor = None) 
                ax.axvline(x = expert_data, color = "black", linestyle="dashed")
                plt.suptitle("elicited statistics - histogram", size = "medium")
             
            else:
                if training_data.shape[-1] == global_dict["rep"]:
                    _, ax = plt.subplots(1,1, constrained_layout = True, figsize = (3,3))
                    [sns.histplot(training_data[b,...], bins = 20, color = "orange", stat="density",
                                alpha = 0.2, ax = ax, edgecolor = None) for b in range(sims)]
                    sns.kdeplot(expert_data[0,...], ax = ax, color = "black")
                    plt.suptitle("elicited statistics - histogram")
                
                if (training_data.shape[-1] != global_dict["rep"]) and tf.rank(training_data) == 3:
                    groups = training_data.shape[-1]
                    _, ax = plt.subplots(1,groups, constrained_layout = True, figsize = (int(groups*2),3))
                    
                    for gr in range(groups):
                        [sns.histplot(training_data[b,...,gr], bins = 20, color = "orange", stat="density",
                                    alpha = 0.2, ax = ax[gr], edgecolor = None) for b in range(sims)]
                        sns.kdeplot(expert_data[0,...,gr], ax = ax[gr], color = "black")
                    [ax[i].set_title(f"y_pred {i}") for i in range(groups)]
                    plt.suptitle("elicited statistics - histogram")
                
                if (training_data.shape[-1] != global_dict["rep"]) and tf.rank(training_data) == 2:
                     groups = training_data.shape[-1]
                     _, ax = plt.subplots(1,groups, constrained_layout = True, figsize = (int(groups*2),3))
                     
                     for gr in range(groups):
                         sns.histplot(training_data[:,gr], bins = 20, 
                                      color = "orange", stat="density", 
                                      ax = ax[gr], edgecolor = None) 
                         ax[gr].axvline(expert_data[0,gr], color = "black", 
                                        linestyle = "dashed", lw = 3)
                     [ax[i].set_title(f"y_pred {i}") for i in range(groups)]
                     plt.suptitle("elicited statistics - histogram")
                     
            if save_fig:
                path_to_file = global_dict["output_path"]["plots"]+f'/elicited_statistics_hist{i}.png'
                os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
                plt.savefig(path_to_file)
            else:
                plt.show()
        
        if meth == "quantiles":
            groups = training_data.shape[-1]
            _, axs = plt.subplots(1,groups, constrained_layout=True, figsize = (int(groups*2),2))
            [axs[x].axline((training_data[0,0,x],
                            training_data[0,0,x]), 
                            slope = 1, color = "black", linestyle = "dashed") for x in range(groups)]
            for x in range(groups):
                [sns.scatterplot(
                    x = training_data[b,:,x],  
                    y = expert_data[0,:,x],
                    ax = axs[x],
                    color = "orange", alpha = 0.2,
                    s=50
                    ) for b in range(sims)]
            [axs[x].set_yticklabels([]) for x in range(groups)]
            axs[0].set_xlabel("training data")
            axs[0].set_ylabel("expert data")
            #[axs[i].set_title(fr"$y_{{n,{obs}}}$") for i, obs in enumerate(selected_obs)]
            plt.suptitle("elicited statistics - quantile-based")
            if save_fig:
                path_to_file = global_dict["output_path"]["plots"]+f'/elicited_statistics_quant{i}.png'
                os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
                plt.savefig(path_to_file)
            else:
                plt.show()