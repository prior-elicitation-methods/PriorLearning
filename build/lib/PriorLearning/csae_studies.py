import pandas as pd
import tensorflow as tf


def haberman_data_predictor(scaled = True):
    d = pd.read_csv('C:/Users/flobo/hyp_learn_prior/tests/haberman_prep.csv')
    X = tf.constant(d["no_axillary_nodes"], dtype=tf.float32)
    x_sd = tf.math.reduce_std(X)
    if scaled:
        # scale predictor
        X_scaled = tf.constant(X, dtype=tf.float32)/x_sd
        # select only data points that were selected from expert
        dmatrix = tf.gather(X_scaled, [0, 5, 10, 15, 20, 25, 30]) 
    else:
        dmatrix = tf.gather(tf.constant(X, dtype=tf.float32), [0, 5, 10, 15, 20, 25, 30])
    
    return dmatrix