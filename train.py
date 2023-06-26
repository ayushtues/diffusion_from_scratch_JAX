import jax.numpy as jnp 
from dataloader import get_dataloader
from datetime import datetime
from utils import get_values, print_stats
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

dataloader = get_dataloader()
sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values()









