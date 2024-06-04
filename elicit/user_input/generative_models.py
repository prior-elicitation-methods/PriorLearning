import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions
tfb = tfp.bijectors

class GenerativeBinomialModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,           
                total_count,       
                **kwargs        
                ):  
        """
        Binomial model with one continuous predictor.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        total_count : int
            total counts of Binomial model.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions

        """

        # linear predictor
        theta = design_matrix @ tf.expand_dims(prior_samples, axis=-1)

        # map linear predictor to theta
        epred = tf.sigmoid(theta)
        
        # define likelihood
        likelihood = tfd.Binomial(
            total_count = total_count, 
            probs = epred
        )
        
        return dict(likelihood = likelihood,     
                    ypred = None,                 
                    epred = epred,
                    prior_samples = prior_samples                 
                    )

class GenerativePoissonModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,      
                **kwargs        
                ):  
        """
        Poisson model with one continuous predictor and one categorical 
        predictor with three levels.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick 
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions

        """
        # linear predictor
        theta = design_matrix @ tf.expand_dims(prior_samples, -1)
        
        # map linear predictor to theta
        epred = tf.exp(theta)
        
        # define likelihood
        likelihood = tfd.Poisson(
            rate = epred
        )
        
        return dict(likelihood = likelihood,     
                    ypred = None,   
                    epred = epred[:,:,:,0],
                    prior_samples = prior_samples               
                    )

class GenerativeNormalModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,       
                **kwargs        
                ):  
        # compute linear predictor term
        epred = prior_samples[:,:,0:6] @ tf.transpose(design_matrix)
        
        # define likelihood
        likelihood = tfd.Normal(
            loc = epred, 
            scale = tf.expand_dims(tf.math.softplus(prior_samples[:,:,-1]), -1)
            )
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        # create contrast matrix
        cmatrix = tf.cast(pd.DataFrame(design_matrix).drop_duplicates(), tf.float32)
        
        # compute custom target quantity (here: group-differences)
        samples_grouped = tf.stack(
            [
                tf.boolean_mask(ypred, 
                                tf.math.reduce_all(cmatrix[i,:] == design_matrix, axis = 1),
                                    axis = 2) for i in range(cmatrix.shape[0])
            ], axis = -1)

        # compute mean difference between groups
        effect_list = []
        diffs = [(0,3), (1,4), (2,5)]
        
        for i in range(len(diffs)):
            # compute group difference
            diff = tf.math.subtract(
                samples_grouped[:, :, :, diffs[i][0]],
                samples_grouped[:, :, :, diffs[i][1]]
            )
            # average over individual obs within each group
            diff_mean = tf.reduce_mean(diff, axis=2)
            # collect all mean group differences
            effect_list.append(diff_mean)

        mean_effects = tf.stack(effect_list, axis=-1)

        # compute marginals
        ## factor repetition: new, repeated
        marg_ReP = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [range(3),range(3,6)]], 
                     axis = -1), 
            axis = 2)
     
        ## factor Encoding depth: deep, standard, shallow
        marg_EnC = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [[0,3],[1,4],[2,5]]], 
                     axis = -1), 
            axis = 2)
        
        # compute R2
        R2 = tf.divide(tf.math.reduce_variance(epred, -1), 
                       tf.math.reduce_variance(ypred, -1))
        
        # prior samples on correct scale
        prior_samples2 = tf.concat([prior_samples[:,:,:-1], 
                                   tf.math.softplus(prior_samples[:,:,-1])[:,:,None]],
                                   2)
     
        return dict(likelihood = likelihood,     
                    ypred = ypred,   
                    epred = epred,
                    prior_samples = prior_samples2,
                    mean_effects = mean_effects,
                    marginal_ReP = marg_ReP,
                    marginal_EnC = marg_EnC,
                    R2 = R2,
                    sigma = prior_samples2[:,:,-1]
                    )

class GenerativeNormalModel_param(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix,       
                **kwargs        
                ):  
        """
        Normal model for a 2 x 3 factorial design.

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        **kwargs : keyword argument, optional
            additional keyword arguments

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: model predictions; for discrete likelihoods ypred=None as it
          will be approximated via the softmax-gumble trick 
        - epred: predictions of linear predictor
        - prior samples: samples from prior distributions
        - mean_effects: Difference between both factors for each level of factor 2
        - marginal_ReP: marginal distribution of factor 1
        - marginal_EnC: marginal distribution of factor 2
        - R2: variance explained
        - sigma: samples from the noise parameter of the normal likelihood
        """
        # compute linear predictor term
        epred = prior_samples[:,:,0:6] @ tf.transpose(design_matrix)
        
        # define likelihood
        likelihood = tfd.Normal(
            loc = epred, 
            scale = tf.expand_dims(prior_samples[:,:,-1], -1)
            )
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        # create contrast matrix
        cmatrix = tf.cast(pd.DataFrame(design_matrix).drop_duplicates(), tf.float32)
        
        # compute custom target quantity (here: group-differences)
        samples_grouped = tf.stack(
            [
                tf.boolean_mask(ypred, 
                                tf.math.reduce_all(cmatrix[i,:] == design_matrix, axis = 1),
                                    axis = 2) for i in range(cmatrix.shape[0])
            ], axis = -1)

        # compute mean difference between groups
        effect_list = []
        diffs = [(0,3), (1,4), (2,5)]
        
        for i in range(len(diffs)):
            # compute group difference
            diff = tf.math.subtract(
                samples_grouped[:, :, :, diffs[i][0]],
                samples_grouped[:, :, :, diffs[i][1]]
            )
            # average over individual obs within each group
            diff_mean = tf.reduce_mean(diff, axis=2)
            # collect all mean group differences
            effect_list.append(diff_mean)

        mean_effects = tf.stack(effect_list, axis=-1)

        # compute marginals
        ## factor repetition: new, repeated
        marg_ReP = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [range(3),range(3,6)]], 
                     axis = -1), 
            axis = 2)
     
        ## factor Encoding depth: deep, standard, shallow
        marg_EnC = tf.reduce_mean(
            tf.stack([tf.math.add_n([samples_grouped[:, :, :, i] for i in j]) for j in [[0,3],[1,4],[2,5]]], 
                     axis = -1), 
            axis = 2)
        
        # compute R2
        R2 = tf.divide(tf.math.reduce_variance(epred, -1), 
                       tf.math.reduce_variance(ypred, -1))
        
        # prior samples on correct scale
     
        return dict(likelihood = likelihood,     
                    ypred = ypred,   
                    epred = epred,
                    prior_samples = prior_samples,
                    mean_effects = mean_effects,
                    marginal_ReP = marg_ReP,
                    marginal_EnC = marg_EnC,
                    R2 = R2,
                    sigma = prior_samples[:,:,-1]
                    )


class GenerativeMultilevelModel(tf.Module):
    def __call__(self, 
                prior_samples,        
                design_matrix, 
                selected_days,
                alpha_lkj,
                N_subj,
                N_days,
                **kwargs        
                ):
        """
        Multilevel model with normal likelihood, one continuous predictor and
        one by-subject random-intercept and random-slope

        Parameters
        ----------
        prior_samples : dict
            samples from prior distributions.
        design_matrix : tf.Tensor
            design matrix.
        selected_days : list of integers
            days for which expert should be queried.
        alpha_lkj : float
            parameter of the LKJ prior on the correlation which will be fixed to 1.
        N_subj : int
            number of participants.
        N_days : int
            number of days (selected days for elicitation).
        **kwargs : keyword argument, optional
            additional keyword arguments.

        Returns
        -------
        dictionary with the following keys:
            
        - likelihood: model likelihood
        - ypred: prior predictions (if likelihood is discrete `ypred=None` as it will be approximated using the Softmax-Gumble method)
        - epred: linear predictor
        - prior_samples: we use it here again as output for easier follow-up computations
        - meanperday: distribution of mean reaction time per selected day
        - R2day0: R2 for day 0 (incl. only variation of the random intercept (individual differences in RT without considering the treatment)
        - R2day9: R2 for day 9 (incl. variation of the random intercept and slope (individual differences in RT considering treatment effect)
        - mu0sdcomp:  standard deviation of linear predictor at day 0
        - mu9sdcomp:  standard deviation of linear predictor at day 9
        - sigma: random noise parameter

        """

        B = prior_samples.shape[0]
        rep = prior_samples.shape[1]
        
        # correlation matrix
        corr_matrix = tfd.LKJ(2, alpha_lkj).sample((B, rep))
        
        # SD matrix
        # shape = (B, 2)
        taus = tf.reduce_mean(
            tf.math.softplus(
                tf.gather(prior_samples, indices=[2,3], axis=-1)
                ), 
            axis=1)
        
        # shape = (B, 2, 2)
        S = tf.linalg.diag(taus)
        
        # covariance matrix: Cov=S*R*S
        # shape = (B, 2, 2)
        corr_mat = tf.linalg.diag(diagonal=(1.,1.), 
                                  padding_value=tf.reduce_mean(corr_matrix))
        # compute cov matrix
        # shape = (B, 2, 2)
        cov_mx_subj = tf.matmul(tf.matmul(S,corr_mat), S)
        
        # generate by-subject random effects: T0s, T1s
        # shape = (B, N_subj, 2)
        subj_rfx = tfd.Sample(
            tfd.MultivariateNormalTriL(
                loc= [0,0], 
                scale_tril=tf.linalg.cholesky(cov_mx_subj)), 
            N_subj).sample()
        
        # broadcast by-subject random effects
        # shape = (B, N_obs, 2) with N_obs = N_subj*N_days
        taus = tf.reshape(
            tf.broadcast_to(
                tf.expand_dims(subj_rfx, axis=2), 
                shape=(B, N_subj, N_days, 2)), 
            shape=(B, N_subj*N_days, 2))
        
        # reshape coefficients
        # shape = (B, rep, N_obs, 2) with N_obs = N_subj*N_days
        betas_reshaped = tf.broadcast_to(
            tf.expand_dims(
                tf.gather(prior_samples, indices=[0,1], axis=-1),
                axis=2), 
            shape=(B, rep, N_subj*N_days, 2))
        
        ## compute betas_s
        # shape = (B, rep, N_obs, 2) with N_obs = N_subj*N_days
        betas = tf.add(betas_reshaped, tf.expand_dims(taus, axis=1)) 
        
        # compute linear predictor term
        # shape = (B, rep, N_obs) with N_obs = N_subj*N_days
        epred = tf.add(betas[:,:,:,0]*design_matrix[:,0], 
                       betas[:,:,:,1]*design_matrix[:,1])
        
        # define likelihood
        likelihood = tfd.Normal(loc = epred,
                                scale = tf.expand_dims(prior_samples[:,:,-1], -1)
                                )
        
        # sample prior predictive data
        ypred = likelihood.sample()
        
        # custom target quantities 
        ## epred averaged over individuals
        epred_days = tf.stack([epred[:,:,i::N_days] for i in range(N_days)], 
                              axis = -1)
        mean_per_day = tf.reduce_mean(epred_days, axis=2)
        
        # R2 for initial day
        R2_day0 = tf.divide(tf.math.reduce_variance(epred[:,:,selected_days[0]::N_days], -1),
                            tf.math.reduce_variance(ypred[:,:,selected_days[0]::N_days], -1))
        
        # R2 for last day
        R2_day9 = tf.divide(tf.math.reduce_variance(epred[:,:,selected_days[-1]::N_days], -1),
                            tf.math.reduce_variance(ypred[:,:,selected_days[-1]::N_days], -1))
        
        # compute standard deviation of linear predictor 
        mu0_sd_comp = tf.math.reduce_std(epred[:,:,selected_days[0]::N_days], 
                                         axis=-1)
        mu9_sd_comp = tf.math.reduce_std(epred[:,:,selected_days[-1]::N_days], 
                                         axis=-1)
        
        return dict(likelihood = likelihood,     
                    ypred = ypred,   
                    epred = epred,
                    prior_samples = prior_samples,
                    meanperday = mean_per_day,
                    R2day0 = R2_day0,
                    R2day9 = R2_day9,
                    mu0sdcomp = mu0_sd_comp,
                    mu9sdcomp = mu9_sd_comp,
                    sigma = prior_samples[:,:,-1]
                    )

