import typing as tp
import jax

from ..core import nn
from ..methods.diffusion import Diffuser, FlowParameterization
from ..methods.flow_map import FlowMap
from ..methods.gan import Discriminator, Generator


class DiffuserFactory(tp.Protocol):
    def create_diffuser[T, Cond](
        self,
        parameterization: FlowParameterization[T, Cond],
        value: T,
        cond: Cond,
        *,
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ) -> Diffuser[T, Cond]: ...


class GeneratorFactory(tp.Protocol):
    def create_generator[T, Cond](
        self,
        value: T,
        cond: Cond,
        *,
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ) -> Generator[T, Cond, tp.Any]: ...


class DiscriminatorFactory(tp.Protocol):
    def create_discriminator[T, Cond](
        self,
        value: T,
        cond: Cond,
        *,
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ) -> Discriminator[T, Cond]: ...


class FlowMapFactory(tp.Protocol):
    def create_flow_map[T, Cond, Aux](
        self,
        value: T,
        cond: Cond,
        aux: Aux,
        *,
        precision: jax.lax.Precision | None = None,
        rngs: nn.Rngs,
    ) -> "FlowMap[T, Cond, Aux]": ...
