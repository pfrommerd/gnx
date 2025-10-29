import typing as tp
import jax.numpy as jnp
import jax

from ...methods.diffusion import FlowParameterization
from ...core import graph_util, nn
from ...core.dataclasses import dataclass

from ..common.resnet import ActivationDef
from .. import DiffuserFactory
from . import MLP, FiLMMLP


class MLPDiffuser[T, Cond](nn.Module):
    def __init__(
        self,
        value_features: int,
        cond_features: int,
        *,
        hidden_features: int,
        hidden_layers: int,
        embed_features: int,
        embed_layers: int,
        activation: ActivationDef,
        #
        parameterization: FlowParameterization[T, Cond],
        rngs: nn.Rngs,
    ):
        self.parameterization = parameterization
        self.film_mlp = FiLMMLP(
            in_features=value_features,
            out_features=value_features,
            cond_features=2 + cond_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            embed_features=embed_features,
            embed_layers=embed_layers,
            activation=activation,
            rngs=rngs,
        )

    @jax.jit
    def __call__(self, x: T, /, t: jax.Array, *, cond: Cond) -> T:
        x_flat, x_uf = graph_util.ravel(x)
        cond_flat, _ = graph_util.ravel(cond)
        embed = jnp.stack((jnp.sin(jnp.pi / 2 * t), jnp.cos(jnp.pi / 2 * t)), axis=-1)
        embed = jnp.concatenate((embed, cond_flat), axis=-1)
        output = self.film_mlp(x_flat, embed)
        return x_uf(output)


@dataclass(frozen=True)
class MLPDiffuserFactory(DiffuserFactory):
    hidden_layers: int = 4
    hidden_features: int = 64
    embed_layers: int = 2
    embed_features: int = 64
    activation: tp.Callable[[jax.Array], jax.Array] = jax.nn.gelu

    def create_diffuser[T, Cond](
        self,
        parameterization: FlowParameterization[T, Cond],
        value: T,
        cond: Cond,
        *,
        rngs: nn.Rngs,
    ) -> MLPDiffuser[T, Cond]:
        value_flat, _ = graph_util.ravel(value)
        cond_flat, _ = graph_util.ravel(cond)
        return MLPDiffuser(
            value_flat.shape[-1],
            cond_flat.shape[-1],
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            embed_layers=self.embed_layers,
            embed_features=self.embed_features,
            activation=self.activation,
            parameterization=parameterization,
            rngs=rngs,
        )
