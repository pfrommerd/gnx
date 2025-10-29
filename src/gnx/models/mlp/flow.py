import jax
import jax.numpy as jnp
import typing as tp

from numpy import half


from ...core import graph_util, nn, asserts
from ...core.dataclasses import dataclass
from ...methods.flow_map import Time

from ..common import ActivationDef
from .. import FlowMapFactory
from . import FiLMMLP


class MLPFlowMap[T, Cond, FlowAux](nn.Module):
    def __init__(
        self,
        features: int,
        aux_features: int,
        cond_features: int,
        *,
        hidden_features: int,
        hidden_layers: int,
        embed_features: int,
        embed_layers: int,
        activation: ActivationDef,
        rngs: nn.Rngs,
        **kwargs,
    ):
        self.film_mlp = FiLMMLP(
            in_features=features,
            out_features=features,
            cond_features=4 + cond_features,
            hidden_layers=hidden_layers,
            hidden_features=hidden_features,
            embed_features=embed_features,
            embed_layers=embed_layers,
            activation=activation,
            rngs=rngs,
        )

    def __call__(self, x_s: T, /, s: Time, t: Time, *, cond: Cond, aux: FlowAux) -> T:
        x_flat, x_uf = graph_util.ravel(x_s)
        aux_flat, _ = graph_util.ravel(aux)
        cond_flat, _ = graph_util.ravel(cond)
        assert x_flat.shape[-1] == self.film_mlp.out_features
        assert x_flat.shape[-1] + aux_flat.shape[-1] == self.film_mlp.in_features
        assert cond_flat.shape[-1] + 4 == self.film_mlp.cond_features

        times = jnp.stack((s, t), axis=-1)
        embed = jnp.concatenate((jnp.sin(times), jnp.cos(times)), axis=-1)
        embed = jnp.concatenate((embed, cond_flat), axis=-1)

        output = self.film_mlp(jnp.concatenate((x_flat, aux_flat), axis=-1), embed)
        return x_uf(output)


@dataclass(frozen=True)
class MLPFlowMapFactory(FlowMapFactory):
    hidden_layers: int = 3
    hidden_features: int = 32
    embed_layers: int = 2
    embed_features: int = 32
    activation: tp.Callable[[jax.Array], jax.Array] = jax.nn.gelu

    def create_flow_map[T, Cond, FlowAux](
        self, value: T, cond: Cond, aux: FlowAux, *, rngs: nn.Rngs
    ) -> MLPFlowMap[T, Cond, FlowAux]:
        value_flat, _ = graph_util.ravel(value)
        aux_flat, _ = graph_util.ravel(aux)
        cond_flat, _ = graph_util.ravel(cond)
        return MLPFlowMap[T, Cond, FlowAux](
            value_flat.shape[-1],
            aux_flat.shape[-1],
            cond_flat.shape[-1],
            hidden_layers=self.hidden_layers,
            hidden_features=self.hidden_features,
            embed_layers=self.embed_layers,
            embed_features=self.embed_features,
            activation=self.activation,
            rngs=rngs,
        )
