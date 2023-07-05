from models import UNet
from models import get_position_embeddings
from utils import print_stats
from copy import deepcopy
from flax import linen as nn
from typing import Optional
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import jax


class Diffusion(nn.Module):
    sqrt_alpha_hat_ts: float
    sqrt_alpha_hat_ts_2: float
    alpha_ts: float
    beta_ts: float
    post_std: float
    n_channels: int
    n_classes: int
    bilinear: Optional[bool] = False
    class_conditioned: Optional[bool] = False

    def setup(self):
        self.model = UNet(self.n_channels, self.n_classes, bilinear=False)

        self.sqrt_alpha_ts = jnp.sqrt(self.alpha_ts)
        self.sigma_ts = jnp.sqrt(self.beta_ts)
        self.alpha_ts_2 = 1 - self.alpha_ts

    def __call__(self, x, t, t_embed, eps, y=None, train=True):
        c1 = jnp.expand_dims(
            jnp.expand_dims(
                jnp.expand_dims(
                    jnp.take_along_axis(self.sqrt_alpha_hat_ts, jnp.squeeze(t, -1), 0),
                    -1,
                ),
                -1,
            ),
            -1,
        )
        # TODO, move this to the dataset itself instead of using gather
        c2 = jnp.expand_dims(
            jnp.expand_dims(
                jnp.expand_dims(
                    jnp.take_along_axis(
                        self.sqrt_alpha_hat_ts_2, jnp.squeeze(t, -1), 0
                    ),
                    -1,
                ),
                -1,
            ),
            -1,
        )

        input_x = x * c1 + eps * c2
        if self.class_conditioned:
            eps_pred = self.model(input_x, t_embed, y, train)
        else:
            eps_pred = self.model(input_x, t_embed, None, train)

        return eps_pred

    def forward(self, x, t_embed, y=None, train=True):
        return self.model(x, t_embed, y, train)
    




