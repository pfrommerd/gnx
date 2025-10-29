import itertools
import typing as tp

import jax
import jax.numpy as jnp

from ...methods.flow_map import Time
from ...core import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_features: int,
        hidden_layers: int,
        activation: tp.Callable = jax.nn.relu,
        rngs: nn.Rngs,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        dims = (in_features,) + (hidden_features,) * hidden_layers + (out_features,)
        self.layers = [
            nn.Linear(din, dout, rngs=rngs) for (din, dout) in itertools.pairwise(dims)
        ]

    @jax.jit
    def __call__(self, x):
        for l in self.layers[:-1]:
            x = l(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

    def __iter__(self):
        return iter(self.layers)


class FiLMMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        cond_features: int,
        *,
        embed_features: int,
        embed_layers: int,
        hidden_features: int,
        hidden_layers: int,
        activation: tp.Callable = jax.nn.relu,
        rngs: nn.Rngs,
    ):
        self.layer = nn.Linear(in_features, out_features, rngs=rngs)
        self.in_features = in_features
        self.out_features = out_features
        self.cond_features = cond_features
        self.embed_features = embed_features
        self.activation = activation

        dims = (in_features,) + (hidden_features,) * hidden_layers + (out_features,)
        self.layers = [
            nn.Linear(din, dout, rngs=rngs) for (din, dout) in itertools.pairwise(dims)
        ]

        embed_dims = (
            (cond_features,) + (2 * embed_features,) * embed_layers + (embed_features,)
        )
        self.cond_layers = [
            nn.Linear(din, dout, rngs=rngs)
            for (din, dout) in itertools.pairwise(embed_dims)
        ] if cond_features else [] # fmt: skip

        self.film_layers = [
            MLP(embed_features, 2*dout, hidden_features=embed_features, hidden_layers=1, activation=activation,
                rngs=rngs)
            for dout in dims[1:-1]
        ] if cond_features else [] # fmt: skip

    @jax.jit
    def __call__(
        self,
        x: jax.Array,
        cond: jax.Array | None = None,
        embed: jax.Array | None = None,
    ):
        if embed is not None:
            assert embed.shape[-1] == self.embed_features
        if cond is not None:
            assert cond.shape[-1] == self.cond_features
            y = cond
            for l in self.cond_layers[:-1]:
                y = self.activation(l(y))
            y = self.cond_layers[-1](y)
            embed = y if embed is None else (y + embed)

        for l, cond_mlp in zip(self.layers[:-1], self.film_layers):
            x = l(x)
            if embed is not None:
                shift, scale = jnp.split(cond_mlp(embed), 2, axis=-1)
                x = (1 + scale) * x + shift
            x = self.activation(x)
        x = self.layers[-1](x)
        return x
