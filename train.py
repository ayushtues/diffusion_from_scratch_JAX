import jax.numpy as jnp 
from dataloader import get_dataloader
from datetime import datetime
from utils import get_values, print_stats
import os
from diffusion import Diffusion
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from models import UNet, get_position_embeddings
from jax import random

dataloader = get_dataloader()
sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values()
model = Diffusion(sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std, 1, 1)

x, y, t = next(iter(dataloader))
x = x.transpose(1,2).transpose(2, 3)
x = x.cpu().numpy()
y = y.cpu().numpy()
t = t.cpu().numpy().astype(jnp.int32)
eps = random.normal(random.PRNGKey(0), x.shape)
t_embed = get_position_embeddings(jnp.squeeze(t, -1))
params = model.init(random.PRNGKey(0), x, t, t_embed, eps, None)
# out  = model.apply(params, x, t, t_embed, eps, None)
out  = model.apply(params, method='sample')
print(out[0].shape)
print(len(out))










