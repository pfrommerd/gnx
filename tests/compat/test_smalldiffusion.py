from gnx.models.unet import AttentionBlock
from gnx.models.unet.diffuser import UNetDiffuserFactory
from gnx.methods.noise_schedule import NoiseSchedule
from gnx.core import nn, asserts
from gnx.util import pytorch_convert

from smalldiffusion.diffusion import ScheduleLogLinear as TorchScheduleLogLinear
from smalldiffusion.model import Rngs as TorchRngs
from smalldiffusion.model_unet import (
    Unet as TorchUnet,
    AttnBlock as TorchAttnBlock,
    ResnetBlock as TorchResBlock,
)

from dataclasses import dataclass

import torch
import jax
import jax.numpy as jnp
import numpy as np


def test_attention():
    x = jax.random.normal(jax.random.key(42), (10, 16, 16, 32))

    attn = AttentionBlock(
        32, 2, heads=4, rngs=nn.Rngs(42), precision=jax.lax.Precision.HIGHEST
    )
    tattn = TorchAttnBlock(32, num_heads=4, rngs=TorchRngs(42))
    y = attn(x)

    with torch.no_grad():
        ty = (
            tattn(torch.from_numpy(np.array(x.transpose(0, 3, 1, 2))), None)
            .cpu()
            .numpy()
            .transpose(0, 2, 3, 1)
        )

    assert y.shape == x.shape
    diff = jnp.max(jnp.abs(y - ty))
    assert diff < 1e-6, f"Got attention error of {diff}"


@dataclass
class ModelConfig:
    seed: int
    width: int
    height: int
    model_channels: int
    embed_channels: int
    attn_levels: tuple[int, ...]
    ch_mults: tuple[int, ...]
    in_channels: int
    out_channels: int


def test_smalldiffusion():
    cfg = ModelConfig(
        seed=42,
        width=32,
        height=32,
        model_channels=128,
        embed_channels=512,
        attn_levels=(1,),
        ch_mults=(1, 2, 2, 2),
        in_channels=3,
        out_channels=3,
    )
    schedule = NoiseSchedule.log_linear_noise(0.001, 35).constant_variance()
    t = 0.5
    sigma = (schedule.sigma(t) / schedule.alpha(t)).item()
    input = jax.random.normal(jax.random.key(42), (cfg.height, cfg.width, 3))

    our_weights, _, our_output = our_model(cfg, schedule, input, t)
    their_weights, _, their_output = sd_model(cfg, input, sigma)

    order = {k: i for i, k in enumerate(their_weights.keys())}

    keys = set(their_weights.keys())
    keys.union(set(our_weights.keys()))
    # Order the keys so that they match the initialization order
    keys = sorted(list(keys), key=lambda k: order.get(k, -1))

    for k in keys:
        if k in their_weights and k in our_weights:
            assert (
                their_weights[k].shape == our_weights[k].shape
            ), f"""Shape mismatch for key {k}: {their_weights[k].shape} vs {our_weights[k].shape}"""
            assert np.all(
                their_weights[k] == our_weights[k]
            ), f"Value mismatch for key {k}"
        else:
            assert k in their_weights, f"Key {k} missing in smalldiffusion model"
            assert k in our_weights, f"Key {k} missing in our model"

    diff = jnp.max(jnp.abs(our_output - their_output))
    print("Output difference", diff)
    assert diff < 1e-5, f"Got error of {diff}"


def sd_model(cfg: ModelConfig, input: jax.Array, sigma: float):
    attn_res = tuple(cfg.width // (2**i) for i in cfg.attn_levels)
    torch_model = TorchUnet(
        cfg.width,
        cfg.in_channels,
        cfg.out_channels,
        ch=cfg.model_channels,
        ch_mult=cfg.ch_mults,
        attn_resolutions=attn_res,
        rngs=TorchRngs(cfg.seed),
    )
    torch_model.eval()
    sig = torch.tensor([sigma], dtype=torch.float32)
    with torch.no_grad():
        torch_input = torch.tensor(np.copy(jnp.transpose(input, (2, 0, 1))))
        torch_input = torch_input.unsqueeze(0)
        output = torch_model(torch_input, sig)
        output = output.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
    return (
        {k: v.cpu().numpy() for k, v in torch_model.state_dict().items()},
        torch_model,
        output,
    )


def our_model(cfg: ModelConfig, schedule: NoiseSchedule, input: jax.Array, t: float):
    factory = UNetDiffuserFactory(
        channels=cfg.model_channels,
        channel_mults=cfg.ch_mults,
        attention_levels=cfg.attn_levels,
        embed_features=cfg.embed_channels,
        blocks_per_level=2,
        dropout=0.1,
        snr_time_embed=True,
        film_conditioning=False,
        skip_every_block=True,
    )
    diffuser = factory.create_diffuser(
        schedule.parameterize(0.0, 1.0),
        value=jnp.zeros((cfg.height, cfg.width, cfg.in_channels)),
        rngs=nn.Rngs(cfg.seed),
        precision=jax.lax.Precision.HIGHEST,
    )
    diffuser.eval_mode()
    nested_weights = pytorch_convert.get_unet_diffuser_weights(diffuser)
    output = diffuser(input, t=t)
    return pytorch_convert.from_nested(nested_weights), diffuser, output


if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    jax.config.update("jax_platform_name", "cpu")

    test_attention()
    test_smalldiffusion()
