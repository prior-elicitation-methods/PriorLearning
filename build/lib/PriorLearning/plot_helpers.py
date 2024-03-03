# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:46:55 2023

@author: flobo
"""


import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# loss function
def plot_loss(user_config, res_dict):
    plt.plot(tf.range(0., user_config["epochs"], 1), res_dict["loss_info"]["loss_total"])
    plt.title("Loss")
    plt.xlabel("epochs")

# convergence (for hyperparameter_learning)
def plot_convergence(user_config, res_dict):
    
    locs_rate = tf.stack([res_dict["hyperparam_info"]["hyperparameter"][i][0::2] for i in range(user_config["epochs"])], -1)
    locs_rate_label = res_dict["hyperparam_info"]["varnames"][-13:][0::2]
    rate = locs_rate[6,:]
    rate_label = locs_rate_label[6]
    locs = locs_rate[:6,:] 
    locs_label = locs_rate_label[:6]
    scales = tf.stack([tf.exp(res_dict["hyperparam_info"]["hyperparameter"][i][1::2]) for i in range(user_config["epochs"])], -1)
    scales_label = res_dict["hyperparam_info"]["varnames"][-13:][1::2]
    
    parameter_types = 3
    fig, axs = plt.subplots(1, parameter_types, constrained_layout = True, 
                          figsize = (6,2))
    
    [axs[0].plot(tf.range(0., user_config["epochs"], 1), locs[i], 
                 label=locs_label[i]) for i in range(len(locs))]
    [axs[1].plot(tf.range(0., user_config["epochs"], 1), scales[i], 
                 label = scales_label[i]) for i in range(len(scales))]
    axs[2].plot(tf.range(0., user_config["epochs"], 1), rate,
                label = rate_label)
    [axs[i].legend(handlelength=0.3, fontsize = "small") for i in range(3)]
    fig.suptitle("Convergence")

# final prior distributions

def final_hyperparameters(res_dict):
    mean_res = tf.reduce_mean(
        tf.stack(res_dict["hyperparam_info"]["hyperparameter"][-30:], -1), 
                   -1)
    mus = mean_res[0::2][:6]
    sigmas = tf.exp(mean_res[1::2])
    rate = tf.exp(mean_res[-1])
    
    return mus, sigmas, rate

def plot_priors(res_dict, ideal_expert_dict):
    
    final_mus, final_sigmas, final_rate = final_hyperparameters(res_dict)
    
    fig, axs = plt.subplots(1, 2, constrained_layout = True, 
                          figsize=(6,2), width_ratios=[2,1])
    for key in list(ideal_expert_dict.keys())[:6]:
        sns.kdeplot(ideal_expert_dict[key]["true"].sample(10000), color = "black",
                    ax = axs[0])
    sns.kdeplot(ideal_expert_dict["sigma"]["true"].sample(10000), color = "black",
                ax = axs[1])
    
    for i in range(6):
        sns.histplot(tfd.Normal(final_mus[i], final_sigmas[i]).sample(10000), bins = 30,
                     alpha = 0.5, ax = axs[0], stat = "density", 
                     label = f"N({final_mus[i]:.2f}, {final_sigmas[i]:.2f})")
    sns.histplot(tfd.Exponential(final_rate).sample(10000), bins = 30,
                 alpha = 0.5, ax = axs[1], stat = "density",
                 label = f"Exp({final_rate:.2f})")
    fig.suptitle("Learned prior distributions")
    [axs[i].legend(fontsize = "small", handlelength=0.5) for i in range(2)]

#%% binomial model

cb0 = dict(single="#25be99", mu="#a6f584", sigma="#00ad9a")
cb1 = dict(single="#00828c", mu="#009494", sigma="#2a4858")
cb2 = dict(single="#e5b000", mu="#f4f22e", sigma="#d79000")
cb3 = dict(single="#b35300", mu="#c77100", sigma="#851506")


def plot_binom_hp(user_config, res_dict, prior_pred_res):
    def final_hyperparameters(res_dict):
        mean_res = tf.reduce_mean(
            tf.stack(res_dict["hyperparam_info"]["hyperparameter"][-30:], -1), 
                       -1)
        mus = mean_res[0::2]
        sigmas = tf.exp(mean_res[1::2])
        return mus, sigmas
    
    hyp_epochs = tf.stack(res_dict["hyperparam_info"]["hyperparameter"], -1)
    hyp_labels = [r"$\mu_0$", r"$\sigma_0$", r"$\mu_1$", r"$\sigma_1$"]
    hyp_colors = [cb0["mu"], cb0["sigma"], cb1["mu"], cb1["sigma"]]
    mu, sigma = final_hyperparameters(res_dict)
    prior_labels = [fr"$\beta_0$ ~ N({mu[0]:.2f}, {sigma[0]:.2f})",
                    fr"$\beta_1$ ~ N({mu[1]:.2f}, {sigma[1]:.2f})"]
    prior_colors = [cb0["single"], cb1["single"]]
    
    _, axs = plt.subplots(2,2, constrained_layout=True, figsize = (6,4.5))
    
    axs[0,0].plot(tf.range(0,user_config["epochs"],1),
                  res_dict["loss_info"]["loss_total"], color = "black")
    for i in range(4):
        if i%2 != 0:
            axs[0,1].plot(tf.range(0,user_config["epochs"],1), 
                     tf.exp(hyp_epochs[i,:]), label = hyp_labels[i],
                     color = hyp_colors[i])
        else:
            axs[0,1].plot(tf.range(0,user_config["epochs"],1), 
                     hyp_epochs[i,:], label = hyp_labels[i],
                     color = hyp_colors[i])
    for j in range(2):
        sns.histplot(tf.reshape(res_dict["priors_info"]["priors_samples"][:,:,j],
                                (user_config["B"]*user_config["rep"])), 
                     bins = 50, stat = "density", ax = axs[1,0], 
                     label = prior_labels[j], color = prior_colors[j])
        sns.kdeplot(tf.reshape(prior_pred_res["priors"][:,:,j],
                                (user_config["rep"])), ax = axs[1,0], 
                    color = "black", linestyle = "dashed", linewidth = 2)
    axs[0,1].legend(handlelength=0.5, loc = "upper right", ncols = 2, 
                    fontsize = "small")
    axs[1,0].legend(handlelength=0.5, loc = "upper left", ncols = 1, 
                    fontsize = "small")
    [axs[0,i].set_xlabel("epochs") for i in range(2)]
    axs[1,0].set_xlabel(r"$\beta_k$")
    axs[1,1].remove()
    axs[0,0].set_title("Total loss", fontsize = "medium", ha = "left", x=0)
    axs[0,1].set_title("Hyperparameter convergence", fontsize = "medium", 
                       ha = "left", x=0)
    axs[1,0].set_title("Learned prior distributions", fontsize = "medium", 
                       ha = "left", x=0)

def plot_binom_nf(user_config, res_dict, prior_pred_res):
    def final_moments(res_dict):
        mean_res = tf.reduce_mean(res_dict["priors_info"]["priors_samples"], 
                                  axis=(0,1))
        sd_res = tf.reduce_mean(
            tf.math.reduce_std(res_dict["priors_info"]["priors_samples"],
                               axis=1), axis=0)
        mus = mean_res
        sigmas = sd_res
        return mus, sigmas
    
    mu, sigma = final_moments(res_dict)
    prior_labels = [fr"$\beta_0$ ~ N({mu[0]:.2f}, {sigma[0]:.2f})",
                    fr"$\beta_1$ ~ N({mu[1]:.2f}, {sigma[1]:.2f})"]
    prior_colors = [cb0["single"], cb1["single"]]
    
    _, axs = plt.subplots(1,2, constrained_layout=True, figsize = (6,2.5))
    
    axs[0].plot(tf.range(0,user_config["epochs"],1),
                  res_dict["loss_info"]["loss_total"], color = "black")
    for j in range(2):
        sns.histplot(tf.reshape(res_dict["priors_info"]["priors_samples"][:,:,j],
                                (user_config["B"]*user_config["rep"])), 
                     bins = 50, stat = "density", ax = axs[1], 
                     label = prior_labels[j], color = prior_colors[j])
        sns.kdeplot(tf.reshape(prior_pred_res["priors"][:,:,j],
                                (user_config["rep"])), ax = axs[1], 
                    color = "black",linestyle = "dashed", linewidth = 2)
    axs[1].legend(handlelength=0.5, loc = "upper left", ncols = 1, 
                    fontsize = "small")
    axs[0].set_xlabel("epochs") 
    axs[1].set_xlabel(r"$\beta_k$")
    axs[0].set_title("Total loss", fontsize = "medium", ha = "left", x=0)
    axs[1].set_title("Learned prior distributions", fontsize = "medium", 
                       ha = "left", x=0)

def plot_pois_hp(user_config, res_dict, prior_pred_res):
    def final_hyperparameters(res_dict):
        mean_res = tf.reduce_mean(
            tf.stack(res_dict["hyperparam_info"]["hyperparameter"][-30:], -1), 
                       -1)
        mus = mean_res[0::2]
        sigmas = tf.exp(mean_res[1::2])
        return mus, sigmas
    
    hyp_epochs = tf.stack(res_dict["hyperparam_info"]["hyperparameter"], -1)
    hyp_labels = ["$\mu_0$", "$\sigma_0$", "$\mu_1$", "$\sigma_1$",
                  "$\mu_2$", "$\sigma_2$", "$\mu_3$", "$\sigma_3$","\nu"]
    hyp_colors = [cb0["mu"], cb0["sigma"], cb1["mu"], cb1["sigma"],
                  cb2["mu"], cb2["sigma"], cb3["mu"], cb3["sigma"]]
    mu, sigma = final_hyperparameters(res_dict)
    prior_labels = [fr"$\beta_0$ ~ N({mu[0]:.2f}, {sigma[0]:.2f})",
                    fr"$\beta_1$ ~ N({mu[1]:.2f}, {sigma[1]:.2f})",
                    fr"$\beta_2$ ~ N({mu[2]:.2f}, {sigma[2]:.2f})",
                    fr"$\beta_3$ ~ N({mu[3]:.2f}, {sigma[3]:.2f})"]
    prior_colors = [cb0["single"], cb1["single"],cb2["single"], cb3["single"]]
    
    _, axs = plt.subplots(3,1, constrained_layout=True, figsize = (5,5.5))
    
    axs[0].plot(tf.range(0,user_config["epochs"],1),
                  res_dict["loss_info"]["loss_total"], color = "black")
    for i in range(8):
        if i%2 != 0:
            axs[1].plot(tf.range(0,user_config["epochs"],1), 
                     tf.exp(hyp_epochs[i,:]), label = hyp_labels[i],
                     color = hyp_colors[i])
        else:
            axs[1].plot(tf.range(0,user_config["epochs"],1), 
                     hyp_epochs[i,:], label = hyp_labels[i],
                     color = hyp_colors[i])
    for j in range(4):
        sns.histplot(tf.reshape(res_dict["priors_info"]["priors_samples"][:,:,j],
                                (user_config["B"]*user_config["rep"])), 
                     bins = 50, stat = "density", ax = axs[2], 
                     label = prior_labels[j], color = prior_colors[j])
        sns.kdeplot(tf.reshape(prior_pred_res["priors"][:,:,j],
                                (user_config["rep"])), ax = axs[2], 
                    color = "black", linestyle = "dashed", linewidth = 2)
    axs[1].legend(handlelength=0.5, loc = "upper right", ncols = 2, 
                    fontsize = "small", labelspacing = 0.1)
    axs[2].legend(handlelength=0.5, loc = "upper left", ncols = 2, 
                    fontsize = "small", labelspacing = 0.1, columnspacing=8)
    [axs[i].set_xlabel("epochs") for i in range(2)]
    axs[2].set_xlabel(r"$\beta_k$")
    axs[0].set_title("Total loss", fontsize = "medium", ha = "left", x=0)
    axs[1].set_title("Hyperparameter convergence", fontsize = "medium", 
                       ha = "left", x=0)
    axs[2].set_title("Learned prior distributions", fontsize = "medium", 
                       ha = "left", x=0)


def plot_pois_nf(user_config, res_dict, prior_pred_res):
    def final_moments(res_dict):
        mean_res = tf.reduce_mean(res_dict["priors_info"]["priors_samples"], 
                                  axis=(0,1))
        sd_res = tf.reduce_mean(
            tf.math.reduce_std(res_dict["priors_info"]["priors_samples"],
                               axis=1), axis=0)
        mus = mean_res
        sigmas = sd_res
        return mus, sigmas
    
    mu, sigma = final_moments(res_dict)
    prior_labels = [fr"$\beta_0$ ~ N({mu[0]:.2f}, {sigma[0]:.2f})",
                    fr"$\beta_1$ ~ N({mu[1]:.2f}, {sigma[1]:.2f})",
                    fr"$\beta_2$ ~ N({mu[2]:.2f}, {sigma[2]:.2f})",
                    fr"$\beta_3$ ~ N({mu[3]:.2f}, {sigma[3]:.2f})"]
    prior_colors = [cb0["single"], cb1["single"],cb2["single"], cb3["single"]]
    
    _, axs = plt.subplots(2,1, constrained_layout=True, figsize = (5,4))
    
    axs[0].plot(tf.range(0,user_config["epochs"],1),
                  res_dict["loss_info"]["loss_total"], color = "black")
    for j in range(4):
        sns.histplot(tf.reshape(res_dict["priors_info"]["priors_samples"][:,:,j],
                                (user_config["B"]*user_config["rep"])), 
                     bins = 50, stat = "density", ax = axs[1], 
                     label = prior_labels[j], color = prior_colors[j])
        sns.kdeplot(tf.reshape(prior_pred_res["priors"][:,:,j],
                                (user_config["rep"])), ax = axs[1], 
                    color = "black",linestyle = "dashed", linewidth = 2)
    axs[1].legend(handlelength=0.5, loc = "upper right", ncols = 2, 
                    fontsize = "small")
    axs[0].set_xlabel("epochs") 
    axs[1].set_xlabel(r"$\beta_k$")
    axs[0].set_title("Total loss", fontsize = "medium", ha = "left", x=0)
    axs[1].set_title("Learned prior distributions", fontsize = "medium", 
                       ha = "left", x=0)
