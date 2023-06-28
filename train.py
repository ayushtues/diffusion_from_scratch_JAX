import jax.numpy as jnp 
from dataloader import get_dataloader
from datetime import datetime
from utils import get_values, print_stats
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from models import UNet, get_position_embeddings
from jax import random

dataloader = get_dataloader()
sqrt_alpha_hat_ts, sqrt_alpha_hat_ts_2, alpha_ts, beta_ts, post_std = get_values()

model  = UNet(n_channels=1, n_classes=1, bilinear=False)

x, y, t = next(iter(dataloader))
x = x.transpose(1,2).transpose(2, 3)
x = x.cpu().numpy()
y = y.cpu().numpy()
t = t.cpu().numpy()
t = get_position_embeddings(jnp.squeeze(t, -1))
params = model.init(random.PRNGKey(0), x, t, None)
o = model.apply(params, x, t, None)
print(o.shape)











