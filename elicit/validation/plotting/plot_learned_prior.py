import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def learned_prior_pois(path, file, selected_obs, true_values, 
                            last_vals = 30, save_fig = True):
  
    # load required results
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    epochs = global_dict["epochs"]
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    elicited_statistics = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicited_statistics = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    learned_hyperparameters = tf.stack([final_res["hyperparameter"][key] 
                                        for key in final_res["hyperparameter"].keys()], -1)
    
    # define colors for plotting
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    q_cols_total = ["#49c1db","#5c84c5","#822556"]
    
    # preprocess learned hyperparameters (apply log transformation)
    learned_locs = tf.gather(learned_hyperparameters, [0,2,4,6], axis = 1)
    learned_scales = tf.gather(tf.exp(learned_hyperparameters), [1,3,5,7], axis = 1)
    
    # compute final learned hyperparameter by averaging over the last "last vals".
    avg_res = tf.reduce_mean(learned_hyperparameters[-last_vals:,:], 0)
        
    # compute error between learned and true hyperparameter values
    errors_mus = tf.stack([tf.subtract(learned_locs[i,:], tf.gather(true_values, [0,2,4,6],0))
                                for i in range(epochs)], -1)
    errors_sigmas = tf.stack([tf.subtract(learned_scales[i,:], tf.gather(true_values, [1,3,5,7],0))
                                for i in range(epochs)], -1)
    
    # get elicited statistics from expert and learned ones
    prior_pred_exp = expert_elicited_statistics['quantiles_group_means'][0,:,:]
    prior_pred_mod = elicited_statistics['quantiles_group_means']
    prior_pred_exp_hist = expert_elicited_statistics['histogram_ypred'][0,:,:]
    prior_pred_mod_hist = elicited_statistics['histogram_ypred']
    
    
    fig = plt.figure(layout='constrained', figsize=(5., 4.5))
    figs = fig.subfigures(2,1, height_ratios = (1.4,1)) 
    
    axs0 = figs[0].subplots(2,5)
    axs1 = figs[1].subplots(1,3, width_ratios = (1,1,3))
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Helvetica")
    
    # plot elicited statistics
    [axs0[0,i].axline(xy1 = (prior_pred_mod[0,0,i], prior_pred_exp[0,i]), 
                      xy2 = (prior_pred_mod[0,-1,i], prior_pred_exp[-1,i]), 
                      color = "black", lw = 2, linestyle = "dashed", 
                      zorder = 1) for i in range(3)]
    for b in range(100):
        [sns.scatterplot(x = prior_pred_mod[b,:,i], y = prior_pred_exp[:,i], 
                         color = q_cols_total, ax = axs0[0,i], zorder = 2,
                         linewidth=0, alpha = 0.3) 
         for i in range(3)]
    [axs0[0,i].tick_params(left = False, right = False , labelleft = False , 
                         labelbottom = False, bottom = False) for i in range(4)]
    axs0[0,4].tick_params(left = False, right = False , labelleft = False) 
    [axs0[0,i].set_title(f"{t}", size = "x-small") for i,t in 
     zip(range(3),["Democrats","Swing","Republican"])]
    axs0[0,3].remove()
    axs0[0,1].set_xlabel("model-based quantiles ", size = "x-small")
    axs0[0,0].set_ylabel("true quantiles \n", size = "x-small")
    
    sns.histplot(prior_pred_exp_hist[:,0], bins = 30, ax = axs0[0,4])
    sns.histplot(prior_pred_mod_hist[0,:,0], bins = 30, ax = axs0[0,4], alpha = 0.6)
    [sns.histplot(prior_pred_exp_hist[:,j], bins = 30, ax = axs0[1,i]) 
     for i,j in enumerate(range(1,6))]
    [sns.histplot(prior_pred_mod_hist[0,:,j], bins = 30, ax = axs0[1,i], 
                  alpha = 0.6) for i,j in enumerate(range(1,6))]
    [axs0[1,i].tick_params(left = False, right = False , labelleft = False) 
     for i in range(5)]
    [axs0[1,i].set_ylabel(" ") for i in range(5)]
    [axs0[1,i].set_title(fr"$y_{{{selected_obs[i+1]}}}$", size = "x-small") 
      for i in range(5)]
    axs0[0,4].set_title(fr"$y_{{{selected_obs[0]}}}$", size = "x-small") 
    axs0[1,2].set_xlabel("\# LGBTQ+ antidiscrimination laws ", size = "x-small")
    axs0[1,0].set_ylabel("counts \n", size = "x-small")
    figs[0].suptitle(r"$\textbf{(a)}$"+" Prior predictions: Model-based vs. true quantiles", ha = "left", x = 0.06, size = "small")

    # plot learned prior distributions
    x_rge = np.arange(-3, 4, 0.01)
    [axs1[2].plot(x_rge, tfd.Normal(true_values[i],true_values[i+1]).prob(x_rge), 
                  linestyle = "dotted", lw = 2, color = "black") for i in [0,2,4,6]]
    [axs1[2].plot(x_rge, tfd.Normal(avg_res[i], tf.exp(avg_res[i+1])).prob(x_rge), 
                  lw = 3, color = col_betas[j], alpha = 0.6, 
                  label = fr"$\beta_{j}\sim N$({avg_res[i]:.2f}, {tf.exp(avg_res[i+1]):.2f})") 
     for j,i in enumerate([0,2,4,6])]
    axs1[2].legend(handlelength = 0.15, labelspacing = 0.05, loc = (0.0, 0.45), 
                   frameon = False, fontsize = "x-small")
    axs1[2].set_title(r"$\textbf{(c)}$"+" Learned priors $p(\\theta \mid \hat\lambda)$"+"\n", 
                      loc = "left", size = "small")
    axs1[2].set_xlabel(r"model parameters $\beta_k$", size = "x-small")
    
    # plot error between true und learned hyperparameters
    x_rge = np.arange(0, epochs)
    [axs1[i].axhline(0, color = "black", lw = 2, linestyle = "dashed") for i in range(2)]
    for i in range(4):
        axs1[0].plot(x_rge, errors_mus[i], color = col_betas[i], lw = 3, alpha = 0.6,
                 label = rf"$\mu_{i}$") 
        axs1[1].plot(x_rge, errors_sigmas[i], color = col_betas[i], lw = 3, alpha = 0.6,
                 label = rf"$\sigma_{i}$")
    [axs1[i].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.3, 0.7), 
                    ncol = 2, frameon = False, fontsize = "x-small",
                    columnspacing = .5) for i in [0,1]]
    [axs1[i].set_xlabel("epochs", size = "x-small") for i in range(2)]
    axs1[2].set_xlim((-4.5, 3.5))
    axs1[0].set_title(r"$\textbf{(b)}$"+r" Error: $\hat\lambda-\lambda^*$"+"\n",
                      ha = "left", x = 0., size = "small")
    axs1[0].set_ylim(-0.5, 0.5) 
    axs1[1].set_ylim(-0.25, 0.25)
    [axs1[i].tick_params(axis='x', labelsize=8) for i in range(3)]
    [axs1[i].tick_params(axis='y', labelsize=8) for i in range(3)]

    for j in [0,1]:
        [axs0[j,i].tick_params(axis='y', labelsize=8) for i in range(5)]
        [axs0[j,i].tick_params(axis='x', labelsize=8) for i in range(5)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/pois_priors.png', dpi = 300)
    else:
        plt.show()


def learned_prior_binom(path, file, selected_obs, true_values, 
                            last_vals = 30, save_fig = True):
    # load required results
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    epochs = global_dict["epochs"]
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    elicited_statistics = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    expert_elicited_statistics = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    learned_hyperparameters = tf.stack([final_res["hyperparameter"][key] 
                                        for key in final_res["hyperparameter"].keys()], -1)
    
    # define colors for plotting
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    q_cols_total = ["#49c1db","#5c84c5","#822556"]
    
    # preprocess learned hyperparameters (apply log transformation)
    learned_locs = tf.gather(learned_hyperparameters, [0,2], axis = 1)
    learned_scales = tf.gather(tf.exp(learned_hyperparameters), [1,3], axis = 1)
    
    # compute final learned hyperparameter by averaging over the last "last vals".
    avg_res = tf.reduce_mean(learned_hyperparameters[-last_vals:,:], 0)
        
    # compute error between learned and true hyperparameter values
    errors_mus = tf.stack([tf.subtract(learned_locs[i,:], tf.gather(true_values, [0,2],0))
                                for i in range(epochs)], -1)
    errors_sigmas = tf.stack([tf.subtract(learned_scales[i,:], tf.gather(true_values, [1,3],0))
                                for i in range(epochs)], -1)
    
    # get elicited statistics from expert and learned ones
    prior_pred_exp = expert_elicited_statistics['quantiles_ypred'][0,:,:]
    prior_pred_mod = elicited_statistics['quantiles_ypred']
    
    
    fig = plt.figure(layout='constrained', figsize=(5., 3.))
    figs = fig.subfigures(2,1, height_ratios = (1,1.5))
    
    axs0 = figs[0].subplots(1,7, gridspec_kw = {"hspace": 0.1})
    axs1 = figs[1].subplots(1,3, width_ratios = (1,1,2))
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Helvetica")
    
    [axs0[i].axline(xy1 = (prior_pred_mod[0,0,i], prior_pred_exp[0,i]), 
                    xy2 = (prior_pred_mod[0,-1,i], prior_pred_exp[-1,i]), color = "black", lw = 2, 
                    linestyle = "dashed", zorder = 1) for i in range(7)]
    for b in range(100):
        [sns.scatterplot(x = prior_pred_mod[b,:,i], y = prior_pred_exp[:,i], 
                         color = q_cols_total, ax = axs0[i], zorder = 2, linewidth = 0,
                         alpha = 0.3) for i in range(7)]
    [axs0[i].tick_params(left = False, right = False , labelleft = False , 
                     labelbottom = False, bottom = False) for i in range(7)]
    [axs0[i].set_title(fr"$x_{{{selected_obs[i]}}}$", size = "x-small") for i in range(7)]
    axs0[3].set_xlabel("model-based quantiles ", size = "x-small")
    axs0[0].set_ylabel("true quantiles \n", size = "x-small")
    figs[0].suptitle(r"$\textbf{(a)}$"+" Prior predictions: Model-based vs. true quantiles", ha = "left", x = 0.06, size = "small")
    
    x_rge = np.arange(-1.5, 0.9, 0.01)
    [axs1[2].plot(x_rge, tfd.Normal(true_values[i],true_values[i+1]).prob(x_rge), linestyle = "dotted", lw = 2, color = "black") for i in [0,2]]
    [axs1[2].plot(x_rge, tfd.Normal(avg_res[i],tf.exp(avg_res[i+1])).prob(x_rge), lw = 3, color = col_betas[j], 
              alpha = 0.6, label = fr"$\beta_{j} \sim N$({avg_res[i]:.2f}, {tf.exp(avg_res[i+1]):.2f})") for j,i in enumerate([0,2])]
    axs1[2].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.02, 0.7), frameon = False, fontsize = "x-small")
    axs1[2].set_title(r"$\textbf{(c)}$"+" Learned priors $p(\\theta \mid \hat\lambda)$"+"\n", loc = "left", size = "small")
    axs1[2].set_xlabel(r"model parameters $\beta_k$", size = "x-small")
    
    x_rge = np.arange(0, epochs)
    [axs1[0].plot(x_rge, errors_mus[i], color = col_betas[i], lw = 3, alpha = 0.6,
             label = rf"$\mu_{i}$") for i in range(2)]
    for i in range(2):
        axs1[i].axhline(0, color = "black", lw = 2, linestyle = "dashed") 
        axs1[1].plot(x_rge, errors_sigmas[i], color = col_betas[i], lw = 3, alpha = 0.6,
                 label = rf"$\sigma_{i}$")
        axs1[i].legend(handlelength = 0.3, labelspacing = 0.2, 
                       frameon = False, fontsize = "x-small", columnspacing = 1.,
                       ncol = 2, loc = (0.3, 0.7))
        axs1[i].set_xlabel("epochs", size ="x-small") 
        axs1[i].set_ylim(-0.15, 0.1)
    axs1[0].set_title(r"$\textbf{(b)}$"+r" Error: $\hat\lambda-\lambda^*$"+"\n", 
                      ha = "left", x = -0.1, size = "small")

    [axs1[i].tick_params(axis='x', labelsize=8) for i in range(3)]
    [axs1[i].tick_params(axis='y', labelsize=8) for i in range(3)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/binom_priors.png', dpi = 300)
    else:
        plt.show()
        
def elicited_statistics_normal(path, file, save_fig):
    expert_elicited_statistics = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    q_exp = expert_elicited_statistics["quantiles_marginal_EnC"]
    q_exp2 = expert_elicited_statistics["quantiles_marginal_ReP"]
    q_exp3 = expert_elicited_statistics["quantiles_mean_effects"]
    R2_exp = expert_elicited_statistics["histogram_R2"]
    
    df = pl.DataFrame( ) 
    df = df.with_columns(
        q = pl.Series([0.25, 0.5, 0.75]),
        deep = pl.Series(q_exp[0,:,0].numpy()),
        standard = pl.Series(q_exp[0,:,1].numpy()),
        shallow = pl.Series(q_exp[0,:, 2].numpy())
    )
    df = df.melt(id_vars = "q", value_vars = ["deep", "standard", "shallow"], variable_name = "group")
    df_pd = df.to_pandas()
    
    df2 = pl.DataFrame( ) 
    df2 = df2.with_columns(
        q = pl.Series([0.25, 0.5, 0.75]),
        repeated = pl.Series(q_exp2[0,:,0].numpy()),
        new = pl.Series(q_exp2[0,:,1].numpy())
    )
    df2 = df2.melt(id_vars = "q", value_vars = ["repeated", "new"], variable_name = "group")
    df_pd2 = df2.to_pandas()
    
    df3 = pl.DataFrame( ) 
    df3 = df3.with_columns(
        q = pl.Series([0.25, 0.5, 0.75]),
        deep = pl.Series(q_exp3[0,:,0].numpy()),
        standard = pl.Series(q_exp3[0,:,1].numpy()),
        shallow = pl.Series(q_exp3[0,:,2].numpy())
    )
    df3 = df3.melt(id_vars = "q", value_vars = ["deep", "standard", "shallow"], variable_name = "group")
    df_pd3 = df3.to_pandas()
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    fig, axs = plt.subplots(1,4, layout='constrained', figsize=(6., 2))
    sns.histplot(R2_exp[0,:], ax = axs[0], stat = "proportion", color = "#44546A")
    [axs[1].hlines(y=i, xmin = q_exp[0,0,i], xmax = q_exp[0,2,i], color = "black", zorder=0, lw = 1) for i in range(3)]
    [axs[2].hlines(y=i, xmin = q_exp2[0,0,i], xmax = q_exp2[0,2,i], color = "black", zorder=0, lw = 1) for i in range(2)]
    [axs[3].hlines(y=i, xmin = q_exp3[0,0,i], xmax = q_exp3[0,2,i], color = "black", zorder=0, lw = 1) for i in range(3)]
    [sns.scatterplot(y = i, x = q_exp[0,:,i], color = c, ax = axs[1], zorder = 1) for i,c in zip(range(3), ["#020024", "#44546A", "#00d4ff"])]
    [sns.scatterplot(y = i, x = q_exp2[0,:,i], color = c, ax = axs[2], zorder = 1) for i,c in zip(range(2), ["#020024", "#00d4ff"])]
    [sns.scatterplot(y = i, x = q_exp3[0,:,i], color = c, ax = axs[3], zorder = 1) for i,c in zip(range(3), ["#020024", "#44546A", "#00d4ff"])]
    axs[0].set_xlim(0,1)
    axs[0].set_ylabel("prop.", size = "x-small")
    axs[1].set_yticks([0,1,2], ["dep", "std", "shw"])
    axs[2].set_yticks([0,1], ["rep", "new"])
    axs[3].set_yticks([0, 1, 2], ["dep", "std", "shw"])
    [plt.setp(axs[i].get_yticklabels(), rotation=45, ha="right",fontsize=10, 
         rotation_mode="anchor") for i in [1,2,3]]
    axs[0].set_xlabel(r"${R^2}^{(s)}$", size = "x-small")
    [axs[i].set_title(l, size = "x-small", loc = "left") for i,l in zip([1,2,3], ["Encoding depth", "Repetition", r"$\Delta$PTJ (Rep-New)"])]
    [axs[i].set_xlabel("$Q_p^G$", size = "x-small") for i in [1,2,3]]
    [axs[i].set_ylabel(" ") for i in [1,2,3]]
    [axs[i].set_title(t, pad = 10., size = "small") for i,t in zip([0,2], [r"$\textbf{(a)}$"+" Histogram-based \n elicitation", r"$\textbf{(b)}$"+" Quantile-based elicitation \n"])]
    [axs[i].tick_params(axis='x', labelsize=8) for i in range(4)]
    [axs[i].tick_params(axis='y', labelsize=8) for i in range(4)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/linear_elicited_stats.png', dpi = 300)
    else:
        plt.show()         
        
def learned_prior_linear(path, file, true_values, last_vals = 30, save_fig = True):
    # load required results
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    epochs = global_dict["epochs"]
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    learned_hyperparameters = tf.stack([final_res["hyperparameter"][key] 
                                        for key in final_res["hyperparameter"].keys()], -1)
    
    # define colors for plotting
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    col_nu = ["#a6b7c6", "#594d5c"]
    
    # preprocess learned hyperparameters (apply log transformation)
    learned_locs = tf.gather(learned_hyperparameters, [0,2,4,6,8,10], axis = 1)
    learned_scales = tf.gather(tf.exp(learned_hyperparameters), [1,3,5,7,9,11], axis = 1)
    learned_nu = tf.gather(tf.exp(learned_hyperparameters), [12,13], axis = 1)
    
    # compute final learned hyperparameter by averaging over the last "last vals".
    avg_res = tf.reduce_mean(learned_hyperparameters[-last_vals:,:], 0)
        
    # compute error between learned and true hyperparameter values
    errors_mus = tf.stack([tf.subtract(learned_locs[i,:], tf.gather(true_values, [0,2,4,6,8,10],0))
                                for i in range(epochs)], -1)
    errors_sigmas = tf.stack([tf.subtract(learned_scales[i,:], tf.gather(true_values, [1,3,5,7,9,11],0))
                                for i in range(epochs)], -1)
    errors_nu = tf.stack([tf.subtract(learned_nu[i,:], tf.gather(true_values, [12,13],0))
                                for i in range(epochs)], -1)
    

    fig = plt.figure(layout='constrained', figsize=(6, 3.5))
    figs = fig.subfigures(2,1)
    
    axs0 = figs[0].subplots(1,2, gridspec_kw = {"width_ratios": (2,1)})
    axs1 = figs[1].subplots(1,3, sharex = True)
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Helvetica")
    
    x_rge = np.arange(-0.4, 0.3, 0.01)
    [axs0[0].plot(x_rge, tfd.Normal(true_values[i],true_values[i+1]).prob(x_rge), linestyle = "dotted", lw = 2, color = "black") for i in [0,2,4,6,8,10]]
    [axs0[0].plot(x_rge, tfd.Normal(avg_res[i], tf.exp(avg_res[i+1])).prob(x_rge), lw = 3, color = col_betas[j], 
                  alpha = 0.6, label = fr"$\beta_{j} \sim N$({avg_res[i]:.2f}, {tf.exp(avg_res[i+1]):.2f})") for j,i in enumerate([0,2,4,6,8,10])]
    axs0[0].legend(handlelength = 0.6, labelspacing = 0.2, loc = (0.02,0.15), frameon = False, fontsize = "x-small")
    axs0[0].set_xlabel(r"model parameters $\beta_k$", size = "x-small")
    
    x_rge = np.arange(0, 0.5, 0.01)
    axs0[1].plot(x_rge, tfd.Gamma(true_values[12],true_values[13]).prob(x_rge), linestyle = "dotted", lw = 2, color = "black")
    axs0[1].plot(x_rge, tfd.Gamma(tf.exp(avg_res[12]), tf.exp(avg_res[13])).prob(x_rge), lw = 3, color = col_nu[0], 
                  alpha = 0.6, label = fr"$s \sim Gam$({tf.exp(avg_res[12]):.0f},{tf.exp(avg_res[13]):.0f})")
    axs0[1].legend(handlelength = 0.6, labelspacing = 0.2, loc = "upper right", frameon = False, fontsize = "x-small")
    axs0[1].set_xlabel(r"random noise $s$", size = "x-small")
    
    x_rge = np.arange(0, epochs)
    [axs1[i].axhline(0, color = "black", lw = 2, linestyle = "dashed") for i in range(3)]
    [axs1[0].plot(x_rge, errors_nu[i,:], color = col_nu[i], lw = 3, alpha = 0.6,
                  label = l) for i,l in enumerate([r"$\alpha$",r"$\beta$"])]
    
    for i in range(6):
        axs1[1].plot(x_rge, errors_mus[i,:], color = col_betas[i], lw = 3, alpha = 0.6,
                     label = rf"$\mu_{i}$") 
        axs1[2].plot(x_rge, errors_sigmas[i,:], color = col_betas[i], lw = 3, alpha = 0.6,
                     label = rf"$\sigma_{i}$")
    [axs1[i].legend(handlelength = 0.6, labelspacing = 0.2, loc = l, 
                    frameon = False, fontsize = "x-small", ncol = 3, 
                    columnspacing = 1.) for i,l in zip(range(3), [(0.5,0.35), (0.1,0.55), (0.1,0.05)])]
    [axs1[i].set_title(t, fontsize = "medium") for i,t in zip(range(3), [r"$\alpha,\beta$", r"$\mu_k$", r"$\sigma_k$"])]
    axs1[1].set_xlabel("epochs", size = "x-small")
    axs1[2].set_ylim(-0.1, 0.1)
    figs[1].suptitle(r"$\textbf{(b)}$"+r" Error: $\hat\lambda-\lambda^*$", 
                     x = 0.05, ha = "left", size = "small")
    figs[0].suptitle(r"$\textbf{(a)}$"+" Learned priors $p(\\theta \mid \hat\lambda)$", 
                     x = 0.05, ha = "left", size = "small")

    [axs1[i].tick_params(axis='x', labelsize=8) for i in range(3)]
    [axs1[i].tick_params(axis='y', labelsize=8) for i in range(3)]
    [axs0[i].tick_params(axis='x', labelsize=8) for i in range(2)]
    [axs0[i].tick_params(axis='y', labelsize=8) for i in range(2)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/linear_priors.png', dpi = 300)
    else:
        plt.show()
        
def learned_prior_multilevel(path, file, selected_obs, true_values, last_vals = 30, 
                      save_fig = True): 
    # load required results
    global_dict = pd.read_pickle(path+file+"/global_dict.pkl")
    epochs = global_dict["epochs"]
    final_res = pd.read_pickle(path+file+"/final_results.pkl")
    learned_hyperparameters = tf.stack([tf.exp(final_res["hyperparameter"][key]) 
                                        for key in final_res["hyperparameter"].keys()], -1)
    
    # define colors for plotting
    col_betas = ["#2acaea", "#0a75ad", "#ffd700", "#e86e4d", "#00ffaa", "#135553"]
    col_nu = ["#a6b7c6", "#594d5c"]
    q_cols_total = ["#49c1db","#5c84c5","#822556"]
    
    # preprocess learned hyperparameters (apply log transformation)
    learned_locs = tf.gather(learned_hyperparameters, [0,2], axis = 1)
    learned_scales = tf.gather(learned_hyperparameters, [1,3], axis = 1)
    learned_omegas = tf.gather(learned_hyperparameters, [4,5], axis = 1)
    learned_nu = tf.gather(learned_hyperparameters, [6,7], axis = 1)
    
    # compute final learned hyperparameter by averaging over the last "last vals".
    avg_res = tf.reduce_mean(learned_hyperparameters[-last_vals:,:], 0)
        
    # compute error between learned and true hyperparameter values
    errors_mus = tf.stack([tf.subtract(learned_locs[i,:], tf.gather(true_values, [0,2],0))
                                for i in range(epochs)], -1)
    errors_sigmas = tf.stack([tf.subtract(learned_scales[i,:], tf.gather(true_values, [1,3],0))
                                for i in range(epochs)], -1)
    errors_omegas = tf.stack([tf.subtract(learned_omegas[i,:], tf.gather(true_values, [4,5],0))
                                for i in range(epochs)], -1)
    errors_nu = tf.stack([tf.subtract(learned_nu[i,:], tf.gather(true_values, [6,7],0))
                                for i in range(epochs)], -1)
    
    expert_elicited_statistics = pd.read_pickle(path+file+"/expert/elicited_statistics.pkl")
    elicited_statistics = pd.read_pickle(path+file+"/elicited_statistics.pkl")
    
    expert_model_simulations = pd.read_pickle(path+file+"/expert/model_simulations.pkl")
    model_simulations = pd.read_pickle(path+file+"/model_simulations.pkl")
    
    prior_pred_exp = expert_elicited_statistics['quantiles_meanperday'][0,:,:]
    prior_pred_mod = elicited_statistics['quantiles_meanperday']
    prior_pred_exp_hist = expert_model_simulations['R2day0'][0,:]
    prior_pred_mod_hist = model_simulations['R2day0'][0,:]
    prior_pred_exp_hist2 = expert_model_simulations['R2day9'][0,:]
    prior_pred_mod_hist2 = model_simulations['R2day9'][0,:]
    prior_pred_exp_mom = [expert_elicited_statistics["moments.mean_sigma"][0],
                          expert_elicited_statistics["moments.sd_sigma"][0]]
    prior_pred_mod_mom = [elicited_statistics["moments.mean_sigma"][0],
                          elicited_statistics["moments.sd_sigma"][0]]
    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    
    fig = plt.figure(layout='constrained', figsize=(5., 6.))
    figs = fig.subfigures(2,1, height_ratios = (1,1.5)) 
    
    axs0 = figs[0].subplots(2,5)
    axs1 = figs[1].subplots(3,2)
    
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params, font = "Helvetica") 
    
    [axs0[0,i].axline(xy1 = (prior_pred_mod[0,0,i], prior_pred_exp[0,i]), 
                      xy2 = (prior_pred_mod[0,-1,i], prior_pred_exp[-1,i]), color = "black", lw = 2, 
                     linestyle = "dashed", zorder = 1) for i in range(5)]
    for b in range(100):
        [sns.scatterplot(x = prior_pred_mod[b,:,i], y = prior_pred_exp[:,i], 
                         color = q_cols_total, ax = axs0[0,i], zorder = 2, 
                         linewidth = 0, alpha = 0.3) for i in range(5)]
    [axs0[0,i].tick_params(left = False, right = False , labelleft = False , 
                         labelbottom = False, bottom = False) for i in range(5)]
    axs0[0,4].tick_params(left = False, right = False , labelleft = False) 
    [axs0[0,i].set_title(fr"$y_{{{selected_obs[i]}}}$", size = "x-small") for i in range(5)]
    axs0[0,2].set_xlabel("model-based quantiles", size = "x-small")
    axs0[0,0].set_ylabel("true quantiles", size = "x-small")
    
    sns.histplot(prior_pred_exp_hist, bins = 30, ax = axs0[1,0], stat = "density")
    sns.histplot(prior_pred_mod_hist, bins = 30, ax = axs0[1,0], alpha = 0.6, stat = "density")
    sns.histplot(prior_pred_exp_hist2, bins = 30, ax = axs0[1,1], stat = "density")
    sns.histplot(prior_pred_mod_hist2, bins = 30, ax = axs0[1,1], alpha = 0.6, stat = "density")
    [axs0[1,i].set_xlim(0.4,1) for i in range(2)]
    [axs0[1,i].set_title(fr"$R^2$ (day {d})", size = "x-small") for i,d in enumerate([0,9])]
    [axs0[1,i].tick_params(left = False, right = False , labelleft = False) for i in range(2)]
    [axs0[1,i].set_ylabel(l, size = "x-small") for i,l in enumerate(["density \n", ""])]
    [axs0[1,i].tick_params(axis='x', labelsize=8) for i in range(4)]
    
    [axs0[1,i].axis('off') for i in [2,3]]
    axs0[1,4].remove()
    figs[0].suptitle(r"$\textbf{(a)}$"+" Prior predictions: Model-based vs. expert-elicited quantiles", ha = "left", x = 0.06, size = "small")
    axs0[1,2].text(0.,0.9, r"mean for s:", fontsize = "xx-small")
    axs0[1,2].text(0.,0.5, fr"$m_{{true}}(s)= ${prior_pred_exp_mom[0]:.2f}", fontsize = "xx-small")
    axs0[1,2].text(0.,0.1, fr"$m_{{sim}}(s)= ${prior_pred_mod_mom[0]:.2f}", fontsize = "xx-small")
    axs0[1,3].text(0.,0.9, r"sd for s:", fontsize = "xx-small")
    axs0[1,3].text(0.,0.5, fr"$sd_{{true}}(s)= ${prior_pred_exp_mom[1]:.2f}", fontsize = "xx-small")
    axs0[1,3].text(0.,0.1, fr"$sd_{{sim}}(s)= ${prior_pred_mod_mom[1]:.2f}", fontsize = "xx-small")
    
    x_rge = np.arange(0, epochs)
    axs1[0,0].axhline(0, color = "black", lw = 2, linestyle = "dashed") 
    for i in range(2):
        mus = [r"$\mu_0$",r"$\mu_1$"]
        sigs = [r"$\sigma_0$",r"$\sigma_1$"]
        oms = [r"$\omega_0$",r"$\omega_1$"]
        nus = [r"$\alpha$", r"$\beta$"]
        axs1[0,0].plot(x_rge, errors_mus[i], color = col_betas[i], lw = 3, alpha = 0.6,
                     label = mus[i])
        axs1[0,0].plot(x_rge, errors_sigmas[i], color = col_betas[i+2], lw = 3, alpha = 0.6,
                     label = sigs[i])
        axs1[1,0].plot(x_rge, errors_omegas[i], color = col_betas[i+4], lw = 3, alpha = 0.6,
                     label = oms[i])
        axs1[2,0].plot(x_rge, errors_nu[i], color = col_nu[i], lw = 3, alpha = 0.6,
                     label = nus[i])
    axs1[0,0].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.7, 0.55), 
                     ncol = 2, frameon = False, fontsize = "small", 
                     columnspacing = .5, handletextpad=0.4)
    axs1[0,0].set_ylim(-30,40)
    axs1[1,0].axhline(0, color = "black", lw = 2, linestyle = "dashed") 
    axs1[1,0].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.7, 0.3), 
                     ncol = 2, frameon = False, fontsize = "small", 
                     columnspacing = .5, handletextpad=0.4)
    axs1[2,0].axhline(0, color = "black", lw = 2, linestyle = "dashed")
    axs1[2,0].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.7, 0.4), 
                     ncol = 4, frameon = False, fontsize = "small", 
                     columnspacing = .5, handletextpad=0.4)
    axs1[0,0].set_title(r"$\textbf{(b)}$"+" Error between true and learned \n hyperparameter", ha = "left", x = 0., size = "small")
    axs1[1,0].set_xlabel("epochs", size = "x-small")
    
    x_rge = np.arange(0, 300, 0.01)
    [axs1[0,1].plot(x_rge, tfd.Normal(true_values[i],true_values[i+1]).prob(x_rge), 
                    linestyle = "dotted", lw = 2, color = "black") for i in [0,2]]
    [axs1[0,1].plot(x_rge, tfd.Normal(avg_res[i],avg_res[i+1]).prob(x_rge), 
                    lw = 3, color = col_betas[j], alpha = 0.6, 
                    label = fr"$\beta_{j} \sim N$({avg_res[i]:.1f}, {avg_res[i+1]:.1f})") 
     for j,i in enumerate([0,2])]
    axs1[0,1].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.2, 0.35), 
                     frameon = False, fontsize = "x-small", handletextpad=0.4)
    axs1[0,1].set_title(r"$\textbf{(c)}$"+" Learned prior distributions \n", 
                        loc = "left", size = "small")
    axs1[0,1].set_xlabel(r"model parameters $\beta_k$", size = "x-small")
    [axs1[0,i].tick_params(axis='x', labelsize=8) for i in range(2)]
    
    x_rge = np.arange(0, 125, 0.01)
    [axs1[1,1].plot(x_rge, tfd.TruncatedNormal(0,true_values[i], 
                                               low = 0, high = 500).prob(x_rge), 
                    linestyle = "dotted", lw = 2, color = "black") for i in [4,5]]
    [axs1[1,1].plot(x_rge, tfd.TruncatedNormal(0,avg_res[i], low = 0, 
                                               high = 500).prob(x_rge), 
                    lw = 3, color = col_betas[j], alpha = 0.6, 
                    label = fr"$\omega_{j} \sim N_+$(0, {avg_res[i]:.1f})") 
     for j,i in enumerate([4,5])]
    axs1[1,1].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.4, 0.3), 
                     frameon = False, fontsize = "x-small", ncol = 1, 
                     handletextpad=0.4)
    
    x_rge = np.arange(10, 50, 0.01)
    axs1[2,1].plot(x_rge, tfd.Gamma(true_values[-2], true_values[-1]).prob(x_rge), 
                   linestyle = "dotted", lw = 2, color = "black") 
    axs1[2,1].plot(x_rge, tfd.Gamma(avg_res[-2], avg_res[-1]).prob(x_rge), lw = 3, 
                   color = col_nu[0],  alpha = 0.6, 
                   label = fr"$\nu \sim Gam$({avg_res[-2]:.0f}, {avg_res[-1]:.0f})")
    axs1[2,1].legend(handlelength = 0.3, labelspacing = 0.2, loc = (0.5, 0.4), 
                     frameon = False, fontsize = "x-small", ncol = 1, 
                     handletextpad=0.4)
    axs1[1,1].set_xlabel(r"model parameters $\tau_k, s$", size = "x-small")
    [axs1[1,i].tick_params(axis='x', labelsize=8) for i in range(2)]
    [axs1[i,0].tick_params(axis='y', labelsize=8) for i in range(2)]
    [axs1[i,1].tick_params(axis='y', labelsize=8) for i in range(2)]
    
    if save_fig:
        plt.savefig('elicit/simulations/case_studies/plots/multilevel_priors.png', dpi = 300)
    else:
        plt.show()