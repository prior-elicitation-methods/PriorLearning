import polars as pl
import numpy as np
import tensorflow as tf
import pandas as pd
import patsy as pa

def load_design_matrix_haberman(scaling, selected_obs):
    """
    Loads the haberman data set, preprocess the data, and creates a design 
    matrix as used by the binomial model.
    source: https://archive.ics.uci.edu/dataset/43/haberman+s+survival
        
    Parameters
    ----------
    scaling : str or None
        whether the continuous predictor should be scaled; 
        possible values = ['divide_by_std', 'standardize']
    selected_obs : list of integers or None
        whether only specific observations shall be selected from the design
        matrix.

    Returns
    -------
    design_matrix : tf.Tensor
        design matrix.

    """
    # load dataset from repo
    from ucimlrepo import fetch_ucirepo 
    # fetch dataset 
    d_raw = fetch_ucirepo(id=43)["data"]
    # select predictors
    d_combi = d_raw["features"]
    # create new dataset with predictors and dependen variable
    d_combi["survival_status"] = d_raw["targets"]["survival_status"]
    # aggregate observations for Binomial format
    data = pl.DataFrame(d_combi).group_by("positive_auxillary_nodes").agg()
    # add intercept
    data_int = data.with_columns(intercept = pl.repeat(1, len(data["positive_auxillary_nodes"])))
    # reorder columns
    data_reordered = data_int.select(["intercept", "positive_auxillary_nodes"])
    # scale predictor if specified
    if scaling == "divide_by_std":
        sd = np.std(np.array(data_reordered["positive_auxillary_nodes"]))
        d_scaled = data_reordered.with_columns(X_scaled = np.array(data_reordered["positive_auxillary_nodes"])/sd)
        d_final = d_scaled.select(["intercept", "X_scaled"])
    if scaling == "standardize":
        sd = np.std(np.array(data_reordered["positive_auxillary_nodes"]))
        mean = np.mean(np.array(data_reordered["positive_auxillary_nodes"]))
        d_scaled = data_reordered.with_columns(X_scaled = (np.array(data_reordered["positive_auxillary_nodes"])-mean)/sd)
        d_final = d_scaled.select(["intercept", "X_scaled"])
    if scaling is None:
        d_final = data_reordered
    # select only relevant observations
    if selected_obs is not None:
        d_final = tf.gather(d_final, selected_obs, axis = 0)
    # convert pandas data frame to tensor
    array = tf.cast(d_final, tf.float32)
    return array

def load_design_matrix_equality(scaling, selected_obs):
    """
    Loads the equality index dataset from BayesRule!, preprocess the data, and 
    creates a design matrix as used in the poisson model.
    source: https://www.bayesrulesbook.com/chapter-12

    Parameters
    ----------
    scaling : str or None
        whether the continuous predictor should be scaled; 
        possible values = ['divide_by_std', 'standardize']
    selected_obs : list of integers or None
        whether only specific observations shall be selected from the design
        matrix.

    Returns
    -------
    design_matrix : tf.Tensor
        design matrix.

    """
    # load dataset from repo
    url = "https://github.com/bayes-rules/bayesrules/blob/404fbdbae2957976820f9249e9cc663a72141463/data-raw/equality_index/equality_index.csv?raw=true"
    df = pd.read_csv(url)
    # exclude california from analysis as extreme outlier
    df_filtered = df.loc[df["state"] != "california"]
    # select predictors
    df_prep = df_filtered.loc[:, ["historical", "percent_urban"]]
    # reorder historical predictor
    df_reordered = df_prep.sort_values(["historical", "percent_urban"])
    # add dummy coded predictors
    df_reordered["gop"] = np.where(df_reordered["historical"] == "gop", 1, 0)
    df_reordered["swing"] = np.where(df_reordered["historical"] == "swing", 1, 0)
    df_reordered["intercept"] = 1
    # select only required columns
    data_reordered = df_reordered[["intercept", "percent_urban", "gop","swing"]]
    # scale predictor if specified
    if scaling == "divide_by_std":
        sd = np.std(np.array(data_reordered["percent_urban"]))
        d_scaled = data_reordered.assign(percent_urban_scaled = np.array(data_reordered["percent_urban"])/sd)
        d_final = d_scaled.loc[:,["intercept", "percent_urban_scaled","gop","swing"]]
    if scaling == "standardize":
        sd = np.std(np.array(data_reordered["percent_urban"]))
        mean = np.mean(np.array(data_reordered["percent_urban"]))
        d_scaled = data_reordered.assign(percent_urban_scaled = (np.array(data_reordered["percent_urban"])-mean)/sd)
        d_final = d_scaled.loc[:, ["intercept", "percent_urban_scaled", "gop","swing"]]
    if scaling is None:
        d_final = data_reordered
    if selected_obs is not None:
        # select only relevant observations
        d_final = tf.gather(d_final, selected_obs, axis = 0)
    # convert pandas data frame to tensor
    array = tf.cast(d_final, tf.float32)
    return array

def load_design_matrix_truth(n_group):
    """
    Creates a design matrix for a 2 x 3 factorial design with n_group 
    observations per subgroup. This design matrix is used in the normal model.

    Parameters
    ----------
    n_group : int
        number of observations per subgroup.

    Returns
    -------
    design_matrix : tf.Tensor
        design matrix.

    """
    # construct design matrix with 2-level and 3-level factor
    df =  pa.dmatrix("a*b", pa.balanced(a = 2, b = 3, repeat = n_group), 
                    return_type="dataframe")
    # save in correct format
    d_final = tf.cast(df, dtype = tf.float32)
    return d_final

def load_design_matrix_sleep(scaling, N_days, N_subj, selected_days):
    """
    Creates a design matrix from the sleep study data set as used in the 
    multilevel mode.
    Source: Belenky, G. et al. Patterns of performance degradation and restoration 
    during sleep restriction and subsequent recovery: a sleep dose-response study. 
    J.Sleep Res. 12, 1–12 (2003)

    Parameters
    ----------
    scaling : str or None
        whether the continuous predictor should be scaled; 
        possible values = ['divide_by_std', 'standardize']
    N_days : int
        total number of days (here 10).
    N_subj : int
        total number of participants.
    selected_days : list of integers
        whether specific observations (=days) should be selected from the design
        matrix.

    Returns
    -------
    design_matrix : tf.Tensor
        design matrix.

    """
    # create vector of (0,...9, ...) with length according to no. subj and no. days
    X = tf.cast(tf.tile(tf.range(0., N_days, 1.), [N_subj]), tf.float32)
    # if scaling of predictor is specified
    if scaling == "standardize":
        X_scaled = (X - tf.reduce_mean(X))/tf.math.reduce_std(X)
    if scaling == "divide_by_std":
        X_scaled = X/(tf.math.reduce_std(X))
    if scaling is None:
        X_scaled = X
    # add intercept
    design_matrix = tf.stack([tf.ones(X_scaled.shape), X_scaled],-1)
    # select only relevant days
    if selected_days is not None:
        dmatrix = tf.stack([design_matrix[day::N_days, :] for day in selected_days], axis=1)
        dmatrix = tf.reshape(dmatrix, (N_subj*len(selected_days), 2))
    else:
        dmatrix = design_matrix
    return dmatrix