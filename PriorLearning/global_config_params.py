import tensorflow_probability as tfp

tfd = tfp.distributions


_global_variables = dict(
    method = "softmax_gumbel_trick", 
    softmax_gumbel_temp = 1.0,
    coupling_design = "affine",  # spline
    coupling_layers = 7,
    units = 2**7, 
    activation = "relu",
    permutation = "fixed",       
    lr_step = 5,
    lr_perc = 0.90,
    clipnorm_val = 1.,
    task_balance_factor = 1.6,
    patience = 300
    )     