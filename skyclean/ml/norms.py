"""Numerically safe replacement for s2ai's SphericalGroupNorm.

s2ai computes the group variance as E[x^2] - E[x]^2 in float32. Summed over ~5e5 pixels, that subtraction carries a
rounding error of ~1e-6 relative to E[x^2], so a channel group whose activations drift towards a near-constant map
gets a *negative* variance; `sqrt(var + 1e-6)` is then nan, the loss and gradients are nan, and Adam's state is
poisoned for the rest of the run (see TRAIN_NAN.md). This subclass uses the two-pass form E[(x - mu)^2], which is
non-negative by construction and accurate relative to its own size. Parameters and state (gamma, beta, quad_weights)
are unchanged, so checkpoints trained with the original layer load as before.

`install()` rebinds the class name inside s2ai so every block s2ai builds (DiscoConvBlock, PointwiseConvBlock, ...)
uses this version; model.py calls it at import, before any S2_UNET is constructed.
"""
import jax.numpy as jnp

import s2ai.layers.norms as _s2ai_norms
import s2ai.blocks.core_blocks as _s2ai_core_blocks


class SphericalGroupNorm(_s2ai_norms.SphericalGroupNorm):
    """s2ai SphericalGroupNorm with a two-pass (non-negative) spherical variance."""

    def __call__(self, inputs):
        if inputs.shape[-1] % self.groups != 0:
            raise ValueError(f"Number of groups {self.groups} does not divide channels {inputs.shape[-1]}")
        group_size = inputs.shape[-1] // self.groups

        # Reshape channels into groups: [batch, theta, phi, group, channel-in-group]
        x = jnp.reshape(inputs, inputs.shape[:-1] + (self.groups, group_size))
        w = self.quad_weights.value

        # Spherical mean within each group
        mu = jnp.sum(jnp.einsum("btpgi,t->btpgi", x, w, optimize=True), axis=(1, 2, 4), keepdims=True) / group_size

        # Spherical variance within each group, two-pass: E[(x - mu)^2] >= 0 by construction
        xc = x - mu
        var = jnp.sum(jnp.einsum("btpgi,t->btpgi", jnp.square(xc), w, optimize=True),
                      axis=(1, 2, 4), keepdims=True) / group_size

        outputs = xc / jnp.sqrt(var + self.epsilon)
        return jnp.reshape(self.gamma.value * outputs + self.beta.value, inputs.shape)


def install() -> None:
    """Make s2ai build the two-pass norm everywhere it would build its own."""
    _s2ai_norms.SphericalGroupNorm = SphericalGroupNorm
    _s2ai_core_blocks.SphericalGroupNorm = SphericalGroupNorm
