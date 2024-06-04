import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

def diagnostics_pois(path, file, save_fig):

    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    
    gradients = tf.stack([final_res["gradients"][i] for i in range(len(final_res["gradients"]))], -1)
    total_loss = tf.stack([final_res["loss"][i] for i in range(len(final_res["loss"]))], -1)
    individual_losses = tf.stack([final_res["loss_component"][i] for i in range(len(final_res["loss_component"]))], -1)
    # learned hyperparameters
    learned_mus = tf.stack([final_res["hyperparameter"][key] for key in final_res["hyperparameter"].keys() if key.startswith("mu")], -1)
    learned_sigmas = tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys() if key.startswith("sigma")], -1)
    
    x_rge = np.arange(0,global_dict["epochs"],1)
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    _, axs = plt.subplots(2,3, constrained_layout = True, sharex = True, 
                          figsize = (5,2.5), gridspec_kw = {"hspace": 0.1})
    
    axs[0,0].plot(x_rge, total_loss)
    axs[0,0].set_title(r"$\textbf{(a)}$"+" Total loss", loc = "left")
    
    [sns.scatterplot(x = x_rge, y = gradients[i,:], ax = axs[0,1], 
                     marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2,4,6]]
    axs[0,1].set_title(r"$\textbf{(b)}$"+r" Gradients: $\mu_k$", loc = "left")
    axs[0,1].set_ylim(-10,10)
    
    [sns.scatterplot(x = x_rge, y = gradients[i+1,:], ax = axs[0,2], marker = "x", 
                     s = 5, color = "black", alpha =.6) for i in [0,2,4,6]]
    axs[0,2].set_title(r"$\sigma_k$")
    
    [axs[1,0].plot(x_rge, individual_losses[i,:]) for i in range(individual_losses.shape[0])]
    axs[1,0].set_title("Individual losses", loc = "left")
    
    [axs[1,1].plot(x_rge, learned_mus[:,i], color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\mu_{i}$") for i in range(4)]
    axs[1,1].set_title(r"$\textbf{(c)}$"+" Convergence:", loc = "left")
    axs[1,1].set_ylim(-2.,4.5)
    
    [axs[1,2].plot(x_rge, learned_sigmas[:,i], color = col_betas[i], lw = 3, alpha = 0.6,
                  label = rf"$\sigma_{i}$") for i in range(4)]
    axs[1,2].set_ylim(0.,0.3)
    
    [axs[1,i].legend(handlelength = 0.3, labelspacing = 0.1, loc = l, frameon = False, fontsize = "x-small", ncol = 4, 
                    columnspacing = 0.2) for i,l in zip([1,2], [(0.05,0.8), (0.05,0.8)])]
    [axs[1,i].set_xlabel("epochs", size = "x-small") for i in range(3)]
    [axs[1,i].tick_params(axis='x', labelsize=8) for i in range(3)]
    [axs[0,i].tick_params(axis='y', labelsize=8) for i in range(3)]
    [axs[1,i].tick_params(axis='y', labelsize=8) for i in range(3)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/pois_diagnostics.png', dpi = 300)
    else:
        plt.show()


def diagnostics_binom(path, file, save_fig):

    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    
    gradients = tf.stack([final_res["gradients"][i] for i in range(len(final_res["gradients"]))], -1)
    total_loss = tf.stack([final_res["loss"][i] for i in range(len(final_res["loss"]))], -1)
    individual_losses = tf.stack([final_res["loss_component"][i] for i in range(len(final_res["loss_component"]))], -1)
    # learned hyperparameters
    learned_mus = tf.stack([final_res["hyperparameter"][key] for key in final_res["hyperparameter"].keys() if key.startswith("mu")], -1)
    learned_sigmas = tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys() if key.startswith("sigma")], -1)

    x_rge = np.arange(0, global_dict["epochs"],1)
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    _, axs = plt.subplots(2,3, constrained_layout = True, sharex = True, 
                          figsize = (5,2.5), gridspec_kw = {"hspace": 0.1})
    
    axs[0,0].plot(x_rge, total_loss)
    axs[0,0].set_title(r"$\textbf{(a)}$"+" Total loss", loc = "left")
    
    [sns.scatterplot(x = x_rge, y = gradients[i,:], ax = axs[0,1], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2]]
    axs[0,1].set_title(r"$\textbf{(b)}$"+r" Gradients: $\mu_k$", loc = "left")
    axs[0,1].set_ylim(-10,10)
    
    [sns.scatterplot(x = x_rge, y = gradients[i+1,:], ax = axs[0,2], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2]]
    axs[0,2].set_title(r"$\sigma_k$")
    
    [axs[1,0].plot(x_rge, individual_losses[i,:]) for i in range(individual_losses.shape[0])]
    axs[1,0].set_title("Individual losses", loc = "left")
    
    for i in range(2):
        axs[1,1].plot(x_rge, learned_mus[:,i], color = col_betas[i], lw = 3, alpha = 0.6,
                      label = rf"$\mu_{i}$")
        axs[1,2].plot(x_rge, learned_sigmas[:,i], color = col_betas[i], lw = 3, alpha = 0.6,
                      label = rf"$\sigma_{i}$") 
    axs[1,1].set_title(r"$\textbf{(c)}$"+r" Convergence:", loc = "left")
    [axs[1,i].legend(handlelength = 0.6, labelspacing = 0.2, loc = l, frameon = False, fontsize = "small", ncol = 2, 
                        columnspacing = 1.) for i,l in zip([1,2], ["center right", "upper right"])]
    [axs[1,i].set_xlabel("epochs", size = "x-small") for i in range(3)]
    [axs[1,i].tick_params(axis='x', labelsize=8) for i in range(3)]
    [axs[0,i].tick_params(axis='y', labelsize=8) for i in range(3)]
    [axs[1,i].tick_params(axis='y', labelsize=8) for i in range(3)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/binom_diagnostics.png', dpi = 300)
    else:
        plt.show()
        
def diagnostics_linear(path, file, save_fig):
    
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    col_nu = ["#a6b7c6", "#594d5c"]
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    
    gradients = tf.stack([final_res["gradients"][i] for i in range(len(final_res["gradients"]))], -1)
    total_loss = tf.stack([final_res["loss"][i] for i in range(len(final_res["loss"]))], -1)
    individual_losses = tf.stack([final_res["loss_component"][i] for i in range(len(final_res["loss_component"]))], -1)
    # learned hyperparameters
    learned_mus = tf.stack([final_res["hyperparameter"][key] for key in final_res["hyperparameter"].keys() if key.startswith("mu")], -1)
    learned_sigmas = tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys() if key.startswith("sigma")], -1)
    learned_nu = tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys() if key.startswith("concentration") or key.startswith("rate")], -1)
    x_rge = np.arange(0, global_dict["epochs"],1)
    
    
    _, axs = plt.subplots(2,4, constrained_layout = True, sharex = True, 
                          figsize = (5.,2.5))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    axs[0,0].plot(x_rge, total_loss)
    axs[0,0].set_title(r"$\textbf{(a)}$"+" Total loss", loc = "left", size = "medium")
    
    [sns.scatterplot(x = x_rge, y = gradients[i,:], ax = axs[0,1], marker = "x", s = 5, color = "black", alpha =.6) for i in [12,13]]
    axs[0,1].set_title(r"$\textbf{(b)}$"+r" Gradients: $\nu$", loc = "left", size = "medium")
    
    [sns.scatterplot(x = x_rge, y = gradients[i,:], ax = axs[0,2], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2,4,6,8,10]]
    axs[0,2].set_title(r"$\mu_k$")
    
    [sns.scatterplot(x = x_rge, y = gradients[i+1,:], ax = axs[0,3], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2,4,6,8,10]]
    axs[0,3].set_title(r"$\sigma_k$")
    
    [axs[0,i].set_ylim(-0.4,0.4) for i in [1,3]]
    
    [axs[1,0].plot(x_rge, individual_losses[i,:]) for i in range(individual_losses.shape[0])]
    axs[1,0].set_title("Individual losses", size = "medium")
    
    [axs[1,1].plot(x_rge, learned_nu[:,i], label = l, 
                   color = col_nu[i], lw = 3, alpha = 0.6) for i,l in enumerate([r"$\alpha$",r"$\beta$"])]
    axs[1,1].set_title(r"$\textbf{(c)}$"+r" Convergence:", loc = "left", 
                       size = "medium")
    
    for i in range(6):
        axs[1,2].plot(x_rge, learned_mus[:,i], color = col_betas[i], lw = 3, 
                      alpha = 0.6, label = rf"$\mu_{i}$") 
        axs[1,3].plot(x_rge, learned_sigmas[:,i], color = col_betas[i], 
                      lw = 3, alpha = 0.6, label = rf"$\sigma_{i}$") 
    axs[1,3].set_ylim(0, 0.2)
    [axs[1,i].legend(handlelength = 0.3, labelspacing = 0.1, loc = (0.25,0.5), 
                     frameon = False, fontsize = "x-small", ncol = 2, 
                     columnspacing = 0.3) for i in [2,3]]
    axs[1,1].legend(handlelength = 0.3, labelspacing = 0.1, loc = (0.25,0.5), 
                    frameon = False, fontsize = "x-small", ncol = 2, 
                    columnspacing = 0.3)
    axs[1,2].set_ylim(-0.25, 0.7)
    [axs[1,i].set_xlabel("epochs", size = "small") for i in range(4)]
    [axs[1,i].tick_params(axis='x', labelsize=7) for i in range(4)]
    [axs[1,i].tick_params(axis='y', labelsize=7) for i in range(4)]
    [axs[0,i].tick_params(axis='y', labelsize=7) for i in range(4)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/linear_diagnostics.png', dpi = 300)
    else:
        plt.show()  
   
def diagnostics_multilevel(path, file, save_fig):    

    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    epochs = global_dict["epochs"]
    col_nu = ["#a6b7c6", "#594d5c"]
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    
    gradients = tf.stack([final_res["gradients"][i] for i in range(len(final_res["gradients"]))], -1)
    total_loss = tf.stack([final_res["loss"][i] for i in range(len(final_res["loss"]))], -1)
    individual_losses = tf.stack([final_res["loss_component"][i] for i in range(len(final_res["loss_component"]))], -1)
    # learned hyperparameters
    learned_mus = tf.stack([final_res["hyperparameter"][key] for key in final_res["hyperparameter"].keys() if key.startswith("mu")], -1)
    learned_sigmas = tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys() if key.startswith("sigma")], -1)
    learned_omegas = tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys() if key.startswith("omega")], -1)
    learned_nu = tf.stack([tf.exp(final_res["hyperparameter"][key]) for key in final_res["hyperparameter"].keys() if key.startswith("concentration") or key.startswith("rate")], -1)

    
    fig = plt.figure(layout='constrained', figsize=(5., 4.))
    figs = fig.subfigures(2,1, height_ratios = (1,2)) 
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    axs0 = figs[0].subplots(1,2)
    axs1 = figs[1].subplots(2,4, sharex = True)
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Helvetica")
    
    axs0[0].plot(range(epochs), total_loss, lw = 3)
    axs0[0].set_title(r"$\textbf{(a)}$"+" Total loss", loc = "left", x = -0.11)
    [axs0[1].plot(range(epochs), individual_losses[i,:], lw = 3) for i in range(individual_losses.shape[0])]
    axs0[1].set_title("Individual losses", loc = "left")
    [axs0[i].set_xlabel("epochs", size = "x-small") for i in range(2)]
    
    [sns.scatterplot(x = range(epochs), y = gradients[i,:], ax = axs1[0,0], marker = "x", s = 5, color = "black", alpha =.6) for i in [0,2]]
    axs1[0,0].set_title(r"$\textbf{(b)}$"+r" Gradients: $\mu_k$", loc = "left", x = -0.3)
    [sns.scatterplot(x = range(epochs), y = gradients[i,:], ax = axs1[0,1], marker = "x", s = 5, color = "black", alpha =.6) for i in [1,3]]
    axs1[0,1].set_title(r"$\sigma_k$", loc = "center")
    [sns.scatterplot(x = range(epochs), y = gradients[i,:], ax = axs1[0,2], marker = "x", s = 5, color = "black", alpha =.6) for i in [4,5]]
    axs1[0,2].set_title(r"$\omega_k$", loc = "center")
    [sns.scatterplot(x = range(epochs), y = gradients[i,:], ax = axs1[0,3], marker = "x", s = 5, color = "black", alpha =.6) for i in [6,7]]
    axs1[0,3].set_title(r"$\alpha, \beta$", loc = "center")
    axs1[0,0].set_ylim(-500,500)
    l = [r"$\alpha$", r"$\beta$"]
    for i in range(2):
        axs1[1,0].plot(range(epochs), learned_mus[:,i], color = col_betas[i], lw = 3, alpha = 0.6,
                          label = rf"$\mu_{i}$") 
        axs1[1,1].plot(range(epochs), learned_sigmas[:,i], color = col_betas[i+2], lw = 3, alpha = 0.6,
                          label = rf"$\sigma_{i}$") 
        axs1[1,2].plot(range(epochs), learned_omegas[:,i], color = col_betas[i+4], lw = 3, alpha = 0.6,
                          label = rf"$\omega_{i}$") 
        axs1[1,3].plot(range(epochs), learned_nu[:,i], color = col_nu[i], lw = 3, alpha = 0.6,
                      label = l[i]) 
    [axs1[1,i].set_ylim(0,up) for i,up in zip([1,2], [30, 50])]
    axs1[1,0].set_title("\n"+r"$\textbf{(c)}$"+" Convergence: ", loc = "left", x = -0.3)
    [axs1[1,i].legend(handlelength = 0.3, labelspacing = 0.05, loc = l, frameon = False, fontsize = "small", 
                columnspacing = 0.2, ncol=2, handletextpad = 0.3) for i,l in enumerate([(0.3,0.6)]+[(0.3,0.7)]*2+[(0.3,0.4)])]
    [axs1[1,i].set_xlabel("epochs", size = "x-small") for i in range(4)]

    [axs1[1,i].tick_params(axis='x', labelsize=8) for i in range(4)]
    [axs1[0,i].tick_params(axis='y', labelsize=8) for i in range(4)]
    [axs1[1,i].tick_params(axis='y', labelsize=8) for i in range(4)]
    [axs0[i].tick_params(axis='x', labelsize=8) for i in range(2)]
    [axs0[i].tick_params(axis='y', labelsize=8) for i in range(2)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/multilevel_diagnostics.png', dpi = 300)
    else:
        plt.show()