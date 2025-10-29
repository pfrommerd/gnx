import jax
import jax.numpy as jnp
import typing as tp

from ...core import nn, graph_util
from ...core.dataclasses import dataclass
from ...util.distribution import Gaussian
from .. import GeneratorFactory, DiscriminatorFactory
from . import MLP, FiLMMLP


class MLPDiscriminator[T, Cond = None](nn.Module):
    def __init__(
        self,
        in_features: int,
        cond_features: int = 0,
        *,
        hidden_features: int,
        hidden_layers: int,
        embed_features: int = 32,
        embed_layers: int = 3,
        activation: tp.Callable = jax.nn.relu,
        rngs: nn.Rngs,
    ):
        self.mlp = FiLMMLP(
            in_features=in_features,
            out_features=1,
            cond_features=cond_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            embed_features=embed_features,
            embed_layers=embed_layers,
            activation=activation,
            rngs=rngs,
        )

    @jax.jit
    def __call__(self, x: T, cond: Cond):
        x_flat, _ = graph_util.ravel(x)
        cond_flat, _ = graph_util.ravel(cond)
        assert x_flat.size == self.mlp.in_features
        assert cond_flat.size == self.mlp.cond_features
        if cond is not None:
            return jnp.squeeze(self.mlp(x_flat, cond=cond_flat))
        else:
            return jnp.squeeze(self.mlp(x_flat))


class MLPGenerator[T, Cond](nn.Module):
    def __init__(
        self,
        noise_features: int,
        out_features: int,
        cond_features: int = 0,
        *,
        value_unflatten: tp.Callable[[jax.Array], T] = lambda x: x,
        hidden_features: int,
        hidden_layers: int,
        embed_features: int = 32,
        embed_layers: int = 3,
        activation: tp.Callable = jax.nn.relu,
        rngs: nn.Rngs,
    ):
        self.mlp = FiLMMLP(
            in_features=noise_features,
            out_features=out_features,
            cond_features=cond_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            embed_features=embed_features,
            embed_layers=embed_layers,
            activation=activation,
            rngs=rngs,
        )
        self.output_unflatten = value_unflatten

    @property
    def noise_distribution(self) -> Gaussian[jax.Array]:
        return Gaussian(
            jnp.zeros((self.mlp.in_features,)), jnp.ones((self.mlp.in_features,))
        )

    @jax.jit
    def __call__(self, noise: jax.Array, cond: Cond) -> T:
        cond_flat, _ = graph_util.ravel(cond)
        output = self.mlp(noise, cond_flat)
        return self.output_unflatten(output)


@dataclass
class MLPGANFactory(GeneratorFactory, DiscriminatorFactory):
    hidden_layers: int = 4
    hidden_features: int = 64
    embed_features: int = 64
    embed_layers: int = 3
    activation: tp.Callable = jax.nn.gelu

    def create_generator[T, Cond](
        self, value: T, cond: Cond, *, rngs: nn.Rngs
    ) -> MLPGenerator[T, Cond]:
        value_flat, value_uf = graph_util.ravel(value)
        cond_flat, _ = graph_util.ravel(cond)
        return MLPGenerator(
            noise_features=value_flat.size,
            out_features=value_flat.size,
            cond_features=cond_flat.size,
            value_unflatten=value_uf,
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            embed_features=self.embed_features,
            embed_layers=self.embed_layers,
            activation=self.activation,
            rngs=rngs,
        )

    def create_discriminator[T, Cond](
        self, value: T, cond: Cond, *, rngs: nn.Rngs
    ) -> MLPDiscriminator[T, Cond]:
        value_flat, _ = graph_util.ravel(value)
        cond_flat, _ = graph_util.ravel(cond)
        return MLPDiscriminator(
            in_features=value_flat.size,
            cond_features=cond_flat.size,
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            embed_features=self.embed_features,
            embed_layers=self.embed_layers,
            activation=self.activation,
            rngs=rngs,
        )
