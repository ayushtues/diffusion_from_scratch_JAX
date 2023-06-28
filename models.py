""" Parts of the U-Net model """

from flax import linen as nn
import math
import jax.numpy as jnp
from typing import Optional


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = jnp.zeros((length, d_model))
    position = jnp.expand_dims(jnp.arange(0, length), 1)
    div_term = jnp.exp(
        (
            jnp.arange(0, d_model, 2, dtype=jnp.float32)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe = pe.at[:, 0::2].set(jnp.sin(position.astype(jnp.float32) * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position.astype(jnp.float32) * div_term))

    # pe[:, 0::2] = jnp.sin(position.astype(jnp.float32) * div_term)
    # pe[:, 1::2] = jnp.cos(position.astype(jnp.float32) * div_term)

    return pe


def get_position_embeddings(t):
    x = positionalencoding1d(32, 1000)
    emb = x[t]
    return emb


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    out_channels : int
    mid_channels : Optional[int] = None

    @nn.compact
    def __call__(self, x):
        if not self.mid_channels :
            mid_channels = self.out_channels
        else :
            mid_channels = self.mid_channels

        x = nn.Conv(mid_channels, kernel_size=(3, 3), padding=1, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding=1, use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    out_channels : int

    @nn.compact
    def __call__(self, x):
        x = nn.max_pool(x, (2,2), (2, 2))
        x = DoubleConv(self.out_channels)(x)
        return x

class Up(nn.Module):
    """Upscaling then double conv"""
    in_channels : int
    out_channels : int
    bilinear : Optional[bool] = False

    def setup(self):

        if self.bilinear:
            self.conv = DoubleConv(self.out_channels, self.in_channels//2)
        else:
            self.up = nn.ConvTranspose( self.in_channels //2, [2, 2], (2, 2))
            self.conv = DoubleConv(self.out_channels)

    def __call__(self, x1, x2):
        B, H, W, C = x1.shape
        if self.bilinear:
            x = jax.image.resize(x1, (B*2, H*2, W*2, C*2), method='bilinear')
        else:
            x = self.up(x1)

        diffY = x2.shape[1] - x1.shape[1]
        diffX = x2.shape[2] - x1.shape[2]

        x1 = jnp.pad(x1, [(0, 0), (diffX // 2, diffX - diffX // 2), (diffY // 2, diffY - diffY // 2), (0, 0)])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = jnp.concatenate([x2, x1], axis=3)
        return self.conv(x)



class OutConv(nn.Module):
    out_channels : int

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.out_channels, (1,1))(x)



class UNet(nn.Module):
    n_channels : int
    n_classes : int
    bilinear : bool

    def setup(self):

        self.inc = DoubleConv(64)
        self.down1 = Down(128)
        self.down2 = Down(256)
        self.down3 = Down(512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(self.n_classes)

        self.class_embed = nn.Dense(32)
        input_size = [32, 64, 128, 256, 512, 1024, 512, 256, 128, 64]

        self.linears = [
                nn.Dense(input_size[i + 1])
                for i in range(len(input_size) - 1)
                    ]

    def __call__(self, x, t, y=None):
        # print("x:", x.shape)
        x1 = self.inc(x)
        # print("x1:", x1.shape)
        if y is not None:
            y_embed = self.class_embed(y)
            t = t + y_embed
        # print("t:", t.shape)
        t1 = self.linears[0](t)
        # print("t1:", t1.shape)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        # print("t1:", t1.shape)
        x1 = x1 + t1
        # print("x1:", x1.shape)
        x2 = self.down1(x1)
        # print("x2: ", x2.shape)
        t1 = self.linears[1](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        # print("t1:", t1.shape)
        x2 = x2 + t1
        # print("x2: ", x2.shape)
        x3 = self.down2(x2)
        t1 = self.linears[2](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        x3 = x3 + t1
        x4 = self.down3(x3)
        t1 = self.linears[3](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        x4 = x4 + t1
        x5 = self.down4(x4)
        t1 = self.linears[4](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        x5 = x5 + t1
        x = self.up1(x5, x4)
        t1 = self.linears[5](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        x = x + t1
        x = self.up2(x, x3)
        t1 = self.linears[6](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        x = x + t1
        x = self.up3(x, x2)
        t1 = self.linears[7](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        x = x + t1
        x = self.up4(x, x1)
        t1 = self.linears[8](t)
        t1 = jnp.expand_dims(jnp.expand_dims(t1, 1), 1)
        x = x + t1
        logits = self.outc(x)
        return logits


