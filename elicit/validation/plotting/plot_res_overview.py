import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from functions.user_interface.write_results import write_results 

def res_overview_binom(path, file, selected_obs, title):
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/prior_samples.pkl")
    model_samples = pd.read_pickle(path+file+"/prior_samples.pkl")
    
    model_elicit = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicit = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    
    expert_loc = tf.reduce_mean(expert_res, (0,1))
    expert_scale = tf.reduce_mean(tf.math.reduce_std(expert_res, 1), 0)
    
    hyperparams = tf.stack([final_res["hyperparameter"][key] for key in final_res["hyperparameter"].keys()], 0)
    hyperparam_loc = tf.gather(hyperparams, [0,2], axis = 0)   
    hyperparam_scale = tf.exp(tf.gather(hyperparams, [1,3], axis = 0))   
    
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    
    c_exp = "#03396c"
    c_mod = "#45b5fd"
    
    fig = plt.figure(layout = "constrained", figsize = (6,6))
    subfigs = fig.subfigures(3,1, height_ratios = [1,0.8,0.8])
    
    fig0 = subfigs[0].subplots(2,2, sharex = True)
    fig1 = subfigs[1].subplots(2,2)
    fig2 = subfigs[2].subplots(1,6)
    
    fig0[0,0].plot(range(len(total_loss)), total_loss)
    [fig0[0,1].plot(range(len(total_loss)), component_loss[i,:]) for i in range(component_loss.shape[0])]
    [fig0[1,0].axhline(expert_loc[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,1].axhline(expert_scale[i], linestyle = "dashed", color = "black") for i in range(expert_loc.shape[0])]
    [fig0[1,0].plot(range(len(total_loss)), hyperparam_loc[i,:], lw =3) for i in range(hyperparam_loc.shape[0])]
    [fig0[1,1].plot(range(len(total_loss)), hyperparam_scale[i,:], lw =3, label = fr"$\lambda_{i}$") for i in range(hyperparam_scale.shape[0])]
    fig0[1,1].legend(loc="upper right", handlelength=0.4, fontsize="small", 
                     ncol=2, frameon=False)
    
    [fig0[0,i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    [fig0[1,i].set_title(t) for i,t in enumerate([r"hyperparameter: $\mu_k$", r"hyperparameter: $\sigma_k$"])] 
    
    [fig0[1,i].set_xlabel("epochs", fontsize = "small") for i in range(2)]
    [fig0[0,i].set_yscale("log") for i in range(2)]
    fig0[1,1].set_yscale("log") 
    subfigs[0].suptitle(r"$\textbf{(a)}$"+" Diagnostics and convergence", 
                        ha = "left", x = 0.)
    
    for i in range(2):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,1,1,2], [1,2,3,2,3,3]):
        sns.scatterplot(x=model_samples[0,:,0], y = model_samples[0,:,1], color = c_mod,
                        marker="x", ax = fig1[0,1])
        sns.scatterplot(x=expert_res[0,:,0], y = expert_res[0,:,1], marker="+", 
                        color = c_exp, ax = fig1[0,1])
        sns.kdeplot(x=model_samples[0,:,1], y = model_samples[0,:,0], color = c_mod,
                        ax = fig1[1,0])
        sns.kdeplot(x=expert_res[0,:,1], y = expert_res[0,:,0],
                        color = c_exp, ax = fig1[1,0])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(2),range(2))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in zip([0,1], [1,1])]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(2),range(2))]
    [fig1[i,0].set_ylabel(l) for i,l in enumerate([r"$\beta_0$", r"$\beta_1$"])]
    [fig1[i,0].set_yticklabels([]) for i in range(2)]
    subfigs[1].suptitle(r"$\textbf{(b)}$"+" Joint prior", ha = "left", x = 0.)
    
    subfigs[2].suptitle(r"$\textbf{(c)}$"+" Elicited statistics", ha = "left",
                        x = 0.)
    
    [fig2[x].axline((model_elicit["quantiles_ypred"][0,0,x],
                    model_elicit["quantiles_ypred"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(6)]
    for x in range(6):
        [sns.scatterplot(
            x = model_elicit["quantiles_ypred"][b,:,x],  
            y = expert_elicit["quantiles_ypred"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(fr"$y_{{{selected_obs[x]}}}$")
    fig2[0].set_xlabel("learned data", fontsize = "small")
    fig2[0].set_ylabel("true data", fontsize = "small")

    fig.suptitle(title)
    plot_path = path.replace("sim_results", "plots")
    plt.savefig(plot_path+file+".png", dpi = 300)

    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(path+file,global_dict)

def res_overview_multileve(path, true_values, file, title):
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    expert_res = pd.read_pickle(path+file+"/expert/model_simulations.pkl")["prior_samples"]
    model_samples = pd.read_pickle(path+file+"/model_simulations.pkl")["prior_samples"]
    
    expert_sim = pd.read_pickle(path+file+"/expert/model_simulations.pkl")
    model_sim = pd.read_pickle(path+file+"/model_simulations.pkl")
    
    model_elicit = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicit = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    
    expert_loc = tf.reduce_mean(expert_res, (0,1))
    expert_scale = tf.reduce_mean(tf.math.reduce_std(expert_res, 1), 0)
    
    total_loss = tf.stack(final_res["loss"], -1)
    component_loss = tf.stack(final_res["loss_component"], -1)
    hyperparams =  tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys()],0)
 
    c_exp = "#03396c"
    c_mod = "#45b5fd"
    
    fig = plt.figure(layout = "constrained", figsize = (8,12))
    subfigs = fig.subfigures(5,1, height_ratios = [0.7,0.7,1,0.5,0.5])
    
    fig0 = subfigs[0].subplots(1,2, sharex = True)
    fig01 = subfigs[1].subplots(1,3, sharex = True)
    fig1 = subfigs[2].subplots(5,5)
    fig2 = subfigs[3].subplots(1,5)
    fig3 = subfigs[4].subplots(1,3)
    
    fig0[0].plot(range(len(total_loss)), total_loss)
    [fig0[1].plot(range(len(total_loss)), component_loss[i,:], label = f"loss {i}") for i in range(component_loss.shape[0])]
    [fig01[0].axhline(true_values[i], linestyle = "dashed", color = "black") for i in [0,7]]
    [fig01[1].axhline(true_values[i], linestyle = "dashed", color = "black") for i in [2,4,5]]
    [fig01[2].axhline(true_values[i], linestyle = "dashed", color = "black") for i in [1,3,6]]
    [fig01[0].plot(range(len(total_loss)), hyperparams[i,:], lw =3, label = list(final_res["hyperparameter"].keys())[i]) for i in [0,7]]
    [fig01[1].plot(range(len(total_loss)), hyperparams[i,:], lw =3, label = list(final_res["hyperparameter"].keys())[i]) for i in [2,4,5]]
    [fig01[2].plot(range(len(total_loss)), hyperparams[i,:], lw =3, label = list(final_res["hyperparameter"].keys())[i]) for i in [1,3,6]]
    [fig01[i].legend(loc="center right", handlelength=0.4, fontsize="small", ncol=1,
                    columnspacing=0.5, frameon = False) for i in range(2)]
    fig0[1].legend(loc="upper left", handlelength=0.4, fontsize="small", ncol=1,
                    columnspacing=0.5, frameon = False) 
    
    [fig0[i].set_title(t) for i,t in enumerate(["total loss", "loss per component"])]
    
    [fig0[i].set_xlabel("epochs") for i in range(2)]
    [fig01[i].set_xlabel("epochs") for i in range(3)]
    [fig0[i].set_yscale("log") for i in range(2)]
    [fig01[i].set_yscale("log") for i in [1,2]] 
    
    for i in range(5):
        [sns.kdeplot(model_samples[b,:,i], color = c_mod, alpha = 0.2, ax = fig1[i,i]) for b in range(100)]
        sns.kdeplot(expert_res[0,:,i], color = c_exp, linestyle = "dashed", ax = fig1[i,i])
    for i,j in zip([0,0,0,0,1,1,1,2,2,3], [1,2,3,4,2,3,4,3,4,4]):
        sns.scatterplot(x=model_samples[0,:,i], y = model_samples[0,:,j], color = c_mod,
                        marker="x", ax = fig1[i,j])
        sns.scatterplot(x=expert_res[0,:,i], y = expert_res[0,:,j], marker="+", 
                        color = c_exp, ax = fig1[i,j])
        sns.kdeplot(x=model_samples[0,:,j], y = model_samples[0,:,i], color = c_mod,
                        ax = fig1[j,i])
        sns.kdeplot(x=expert_res[0,:,j], y = expert_res[0,:,i],
                        color = c_exp, ax = fig1[j,i])
    [fig1[i,j].get_xaxis().set_visible(False) for i,j in product(range(5),range(5))]
    [fig1[i,j].get_yaxis().set_visible(False) for i,j in product(range(5),range(5))]
    [fig1[i,j].spines[['right', 'top']].set_visible(False) for i,j in product(range(5),range(5))]
    subfigs[1].suptitle("Marginal priors")
    subfigs[2].suptitle("Joint prior")
    
    subfigs[3].suptitle("elicited statistics")
    [fig2[x].axline((model_elicit["quantiles_meanperday"][0,0,x],
                    model_elicit["quantiles_meanperday"][0,0,x]), 
                    slope = 1, color = "black", linestyle = "dashed") for x in range(5)]
    for x in range(5):
        [sns.scatterplot(
            x = model_elicit["quantiles_meanperday"][b,:,x],  
            y = expert_elicit["quantiles_meanperday"][0,:,x],
            ax = fig2[x],
            color = c_mod, alpha = 0.2,
            s=50
            ) for b in range(100)]
        fig2[x].set_yticklabels([])
        fig2[x].set_title(f"day {x}")
    fig2[0].set_xlabel("training data")
    fig2[0].set_ylabel("expert data")
    
    for i, elicit in enumerate(["R2day0","R2day9"]):
        [sns.histplot(model_sim[elicit][b,...], bins = 20, 
                      color = c_mod, stat="density",
                      alpha = 0.2, ax = fig3[i], edgecolor = None) for b in range(100)]
        # if i == 2:
        sns.kdeplot(expert_sim[elicit][0,...], ax = fig3[i], 
                    color = c_exp, lw = 3)
        #else:
        # fig3[i].axvline(expert_elicit[elicit][0,...], color = c_exp, 
        #                 lw = 3, linestyle="dashed")
        fig3[i].get_yaxis().set_visible(False) 
    [fig3[i].set_title(title) for i, title in enumerate(["mu0 sd", "mu9 sd", "sigma"])]
    
    fig.suptitle(title)
    plt.savefig(path+file+".png")
    # write results table if not done already:
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    write_results(global_dict)