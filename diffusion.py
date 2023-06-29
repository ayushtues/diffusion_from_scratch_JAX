from models import UNet
from models import get_position_embeddings
from utils import print_stats
from copy import deepcopy
from flax import linen as nn
from typing import Optional
import jax.numpy as jnp
from jax import random

class Diffusion(nn.Module):
    sqrt_alpha_hat_ts : float
    sqrt_alpha_hat_ts_2 : float
    alpha_ts : float
    beta_ts : float
    post_std : float
    n_channels : int
    n_classes : int
    bilinear : Optional[bool] = False
    class_conditioned : Optional[bool] = False

    def setup(self):
        self.model = UNet(self.n_channels, self.n_classes, bilinear=False)
        self.ema_model = deepcopy(self.model)
        self.ema_decay = 0.999
        self.ema_start = 1000
        self.ema_update_rate = 1
        self.step = 0

        self.sqrt_alpha_ts = jnp.sqrt(self.alpha_ts)
        self.sigma_ts = jnp.sqrt(self.beta_ts)
        self.alpha_ts_2 = 1 - self.alpha_ts

    # def update_ema(self):
    #     self.step += 1
    #     if self.step % self.ema_update_rate == 0:
    #         if self.step < self.ema_start:
    #             self.ema_model.load_state_dict(self.model.state_dict())
    #         else:
    #             for current_params, ema_params in zip(self.model.parameters(), self.ema_model.parameters()):
    #                 old, new = ema_params.data, current_params.data
    #                 if old is not None:
    #                     ema_params.data = old * self.ema_decay + new * (1 - self.ema_decay)

    def __call__(self, x, t, t_embed, eps, y=None):
        c1 = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.take_along_axis(self.sqrt_alpha_hat_ts, jnp.squeeze(t, -1), 0), -1), -1), -1)
        # TODO, move this to the dataset itself instead of using gather
        c2 = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(jnp.take_along_axis(self.sqrt_alpha_hat_ts_2, jnp.squeeze(t, -1), 0), -1), -1), -1)

        input_x = x * c1 + eps * c2
        if self.class_conditioned:
            eps_pred = self.model(input_x, t_embed, y)
        else:
            eps_pred = self.model(input_x, t_embed)

        return eps_pred

    def sample(self, y=None):
        if y is None:
            y = jnp.zeros([10])
            y = y.at[0].set(1)
        x = random.normal(random.PRNGKey(0), [1, 32, 32, 1])
        x_returned = []
        for i in reversed(range(10)):
            t_embed = jnp.expand_dims(get_position_embeddings(i), 0)
            if self.class_conditioned:
                eps_pred = self.model(x, t_embed, y)
            else:
                eps_pred = self.model(x, t_embed)
            eps_pred = (
                jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(self.alpha_ts_2[i], -1), -1), -1)
                / jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(self.sqrt_alpha_hat_ts_2[i], -1), -1), -1) 
            ) * eps_pred
            x_old = x
            x = x - eps_pred
            x = x * (
                1 /  jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(self.sqrt_alpha_ts[i], -1), -1), -1) 
            )
            if i != 0:
                z = random.normal(random.PRNGKey(0), x.shape)
                z = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(self.sigma_ts[i], -1), -1), -1) * z

            else:
                z = jnp.zeros(x.shape)
            x = x + z

            if i % 50 == 0:
                x_img = (x + 1.0) / 2
                x_returned.append(jnp.squeeze(x_img, 0))

        return x_returned