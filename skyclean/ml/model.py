# Adapted from: https://github.com/astro-informatics/s2ai
# Original code by: Matthew A. Price, Kevin Mulder, Jason D. McEwen
# License: MIT


import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", False)
from flax import nnx
from s2ai.blocks.core_blocks import (
    DiscoConvBlock,
    DiscoConvUpBlock,
    BotResBlock,
)



class S2_UNET(nnx.Module):
    """UNET architecture defined on the sphere, built on s2ai."""
    def __init__(self, L: int = 512, ch_in = 9, chs: list = None, filter_type="axisymmetric", rngs: nnx.Rngs = nnx.Rngs(0),):
        """
        Parameters:
            L (int): The maximum multipole for the wavelet transform.
            ch_in (int): Number of input channels.
            chs (list): List of channel dimensions for each layer. Default: [1, 16, 32, 32, 64]
            filter_type (str): Type of filter to use in the convolutional blocks.
            rngs (nnx.Rngs): Random number generators for initialization.
        """
        if chs is None:
            chs = [1, 16, 32, 32, 64]
        #chs = [1, 64, 64, 256]
        gr = chs[1]//2
        Ls = [int(L / pow(2, i)) for i in range(len(chs) - 1)]
        print(Ls)
        self.len_L = len(Ls)

        activation = nnx.celu  # or could use nnx.gelu, nnx.relu, or nnx.leaky_relu

        # Code local backup
        self.input_conv = DiscoConvBlock(
            (Ls[0], Ls[0]), (ch_in, ch_in), 1, filter_type=filter_type, activation=activation, rngs=rngs
        )
        self.input_res = BotResBlock(
            Ls[0], (ch_in, ch_in, chs[1]), filter_type=filter_type, rngs=rngs
        )

        self.down_convs = []
        self.down_ress = []

        for i in range(self.len_L)[1:]:
            self.down_convs.append(
                DiscoConvBlock(
                    (Ls[i - 1], Ls[i]),
                    (chs[i], chs[i]),
                    gr,
                    filter_type=filter_type,
                    activation=activation,
                    rngs=rngs,
                )
            )
            self.down_ress.append(
                BotResBlock(
                    Ls[i],
                    (chs[i], chs[i], chs[i] if i == self.len_L - 1 else chs[i + 1]),
                    filter_type=filter_type,
                    activation=activation,
                    rngs=rngs,
                )
            )

        self.up_convs = []
        self.up_ress = []

        cur_ch = chs[-2]  # 256
        for i in range(self.len_L - 1)[::-1]:
            self.up_convs.append(
                DiscoConvUpBlock(
                    (Ls[i + 1], Ls[i]),
                    (cur_ch, cur_ch),
                    groups=None,
                    filter_type=filter_type,
                    activation=activation, # ref model had no activation here?
                    rngs=rngs,
                )
            )
            self.up_ress.append(
                BotResBlock(
                    Ls[i],
                    (2 * cur_ch, 2 * cur_ch if i != 0 else chs[i], chs[i]),
                    filter_type=filter_type,
                    activation=activation,
                    rngs=rngs,
                )
            )
            cur_ch = chs[i]
        
        self.output_conv = DiscoConvBlock(
            (Ls[0], Ls[0]), (chs[0], 1), 1, filter_type=filter_type, activation=None, rngs=rngs
        )

    def __call__(self, x):

        # Input convolution
        x = self.input_conv(x)
        x = self.input_res(x)

        connections = {}

        # Down convolutions
        for i in range(self.len_L - 1):
            connections[i] = x
            x = self.down_convs[i](x)
            x = self.down_ress[i](x)

        # Up convolutions
        for i in range(self.len_L - 1):
            x = self.up_convs[i](x)
            x = jnp.concatenate([x, connections[self.len_L - 2 - i]], axis=-1)
            x = self.up_ress[i](x)

        x = self.output_conv(x)
        return jnp.clip(x, -6.5, 6.5)
    
