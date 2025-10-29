import logging
import jax
import jax.numpy as jnp
import typing as tp

from ..core import nn
from ..models.common.resnet import ResNetBlock, ShiftConditioner
from ..models.mlp import MLP
from ..models.unet import AttentionBlock, Downsample, Upsample
from ..models.unet.diffuser import UNetDiffuser

logger = logging.getLogger(__name__)

type NestedDict = dict[tp.Any, tp.Any]
type FlatDict = dict[str, jax.Array]


def set_linear_weights(linear: nn.Linear, weights: NestedDict):
    assert (
        linear.kernel.shape == weights["weight"].T.shape
    ), f"""
        Expected linear kernel shape to be {linear.kernel.shape}, but got {weights["weight"].shape}
    """
    linear.kernel[...] = weights["weight"].T
    del weights["weight"]
    if "bias" in weights:
        assert linear.bias is not None
        assert (
            linear.bias.shape == weights["bias"].shape
        ), f"""
          Expected linear bias shape to be {linear.bias.shape}, but got {weights["bias"].shape}
        """
        assert linear.bias is not None
        linear.bias[...] = weights["bias"]
        del weights["bias"]


def get_linear_weights(linear: nn.Linear) -> NestedDict:
    weights = {"weight": linear.kernel[...].T}
    if linear.bias is not None:
        weights["bias"] = linear.bias[...]
    return weights


def set_conv_weights(conv: nn.Conv, weights: NestedDict):
    assert (
        conv.kernel.shape == weights["weight"].transpose((2, 3, 1, 0)).shape
    ), f"""
        Expected conv kernel shape to be {conv.kernel.shape}, but got {weights["weight"].transpose((2, 3, 1, 0)).shape}
    """
    conv.kernel[...] = weights["weight"].transpose((2, 3, 1, 0))
    del weights["weight"]
    if "bias" in weights:
        assert conv.bias is not None
        assert (
            conv.bias.shape == weights["bias"].shape
        ), f"""
            Expected conv bias shape to be {conv.bias.shape}, but got {weights["bias"].shape}
        """
        conv.bias[...] = weights["bias"]
        del weights["bias"]


def get_conv_weights(conv: nn.Conv) -> NestedDict:
    weights = {"weight": conv.kernel[...].transpose((3, 2, 0, 1))}
    if conv.bias is not None:
        weights["bias"] = conv.bias[...]
    return weights


def set_norm_weights(gn: nn.GroupNorm | nn.BatchNorm | nn.LayerNorm, weights: dict):
    assert ("bias" in weights) == (gn.bias is not None)
    assert ("weight" in weights) == (gn.scale is not None)

    if "weight" in weights:
        assert gn.scale is not None
        assert (
            gn.scale.shape == weights["weight"].shape
        ), f"""
            Expected group norm scale shape to be {gn.scale.shape}, but got {weights["weight"].shape}
        """
        gn.scale[...] = weights["weight"]
        del weights["weight"]
    if "bias" in weights:
        assert gn.bias is not None
        assert (
            gn.bias.shape == weights["bias"].shape
        ), f"""
            Expected group norm bias shape to be {gn.bias.shape}, but got {weights["bias"].shape}
        """
        gn.bias[...] = weights["bias"]
        del weights["bias"]


def get_norm_weights(gn: nn.GroupNorm | nn.LayerNorm | nn.BatchNorm) -> dict:
    weights = {}
    if gn.scale is not None:
        weights["weight"] = gn.scale[...]
    if gn.bias is not None:
        weights["bias"] = gn.bias[...]
    return weights


def set_resblock_weights(resblock: ResNetBlock, weights: dict):
    assert (resblock.conv_proj is not None) == (
        "shortcut" in weights
    ), """
        Expected resblock to have a shortcut, but it does not.
    """
    assert isinstance(resblock.conditioner, ShiftConditioner)
    set_norm_weights(resblock.norm_a, weights["layer1"][0])
    set_conv_weights(resblock.conv_a, weights["layer1"][2])
    set_norm_weights(resblock.norm_b, weights["layer2"][0])
    set_conv_weights(resblock.conv_b, weights["layer2"][3])
    # set the time embedding projection weights
    set_linear_weights(resblock.conditioner.linear, weights["temb_proj"][1])
    if "shortcut" in weights:
        assert isinstance(resblock.conv_proj, nn.Conv)
        set_conv_weights(resblock.conv_proj, weights["shortcut"])


def get_resblock_weights(resblock: ResNetBlock) -> dict:
    assert isinstance(resblock.conditioner, ShiftConditioner)
    weights = {
        "layer1": {
            0: get_norm_weights(resblock.norm_a),
            2: get_conv_weights(resblock.conv_a),
        },
        "layer2": {
            0: get_norm_weights(resblock.norm_b),
            3: get_conv_weights(resblock.conv_b),
        },
        "temb_proj": {1: get_linear_weights(resblock.conditioner.linear)},
    }
    if resblock.conv_proj is not None:
        weights["shortcut"] = get_conv_weights(resblock.conv_proj)
    return weights


def set_attenblock_weights(attenblock: AttentionBlock, weights: dict):
    set_linear_weights(attenblock.qkv, weights["attn"]["qkv"])
    assert isinstance(attenblock.norm, nn.GroupNorm)
    set_norm_weights(attenblock.norm, weights["norm"])
    if "proj_out" not in weights:
        set_linear_weights(attenblock.proj, weights["attn"]["proj"])
    else:
        # combine the proj and proj_out weights...in the original there are two linear layers
        # which is equivalent to a single linear layer with combined weights
        proj_weights = weights["attn"]["proj"]["weight"]
        proj_bias = weights["attn"]["proj"]["bias"]
        proj_out_weights = weights["proj_out"]["weight"].squeeze((2, 3))
        proj_out_bias = weights["proj_out"]["bias"]
        del weights["attn"]["proj"]["weight"]
        del weights["attn"]["proj"]["bias"]
        del weights["proj_out"]["weight"]
        del weights["proj_out"]["bias"]

        proj_combined_bias = proj_out_bias + proj_out_weights @ proj_bias
        proj_combined_weights = proj_out_weights @ proj_weights
        set_linear_weights(
            attenblock.proj, {"weight": proj_combined_weights, "bias": proj_combined_bias}
        )



def get_attenblock_weights(attenblock: AttentionBlock) -> dict:
    # Get the combined projection weights and split them back
    proj_weights = get_linear_weights(attenblock.proj)
    qkv_weights = get_linear_weights(attenblock.qkv)
    norm_weights = get_norm_weights(attenblock.norm)
    weights = {
        "attn": {
            "qkv": qkv_weights,
            "proj": proj_weights,
        },
        "norm": norm_weights,
    }
    return weights


def set_sequence_weights(block: tp.Iterable, weights: NestedDict):
    # use the enumerated index to match weights to layers
    # as e.g. MLP layers do not include non-linearities
    if isinstance(block, MLP):

        def iter(layers):
            for layer in layers[:-1]:
                yield layer
                yield None
            yield layers[-1]

        block = iter(block.layers)

    layers = list(block)
    assert len(layers) == max(weights.keys()) + 1
    for i, sub_weights in weights.items():
        if i >= len(layers):
            raise ValueError(
                f"Index {i} out of range for block with {len(layers)} layers."
            )
        layer = layers[i]
        if isinstance(layer, nn.Conv):
            set_conv_weights(layer, sub_weights)
        elif isinstance(layer, nn.Linear):
            set_linear_weights(layer, sub_weights)
        elif isinstance(layer, nn.GroupNorm):
            set_norm_weights(layer, sub_weights)
        elif isinstance(layer, ResNetBlock):
            set_resblock_weights(layer, sub_weights)
        elif isinstance(layer, AttentionBlock):
            set_attenblock_weights(layer, sub_weights)
        elif isinstance(layer, nn.Module):
            raise ValueError(
                f"Unsupported layer: {layer} with weights {jax.tree.map(jnp.shape, sub_weights)}"
            )


def get_sequence_weights(block: tp.Iterable) -> dict:
    weights = {}

    # For MLP, space out the layers with None for non-linearities
    if isinstance(block, MLP):

        def iter(layers):
            for layer in layers[:-1]:
                yield layer
                yield None
            yield layers[-1]

        block = iter(block.layers)

    for i, layer in enumerate(block):
        if isinstance(layer, nn.Conv):
            weights[i] = get_conv_weights(layer)
        elif isinstance(layer, nn.Linear):
            weights[i] = get_linear_weights(layer)
        elif isinstance(layer, nn.GroupNorm):
            weights[i] = get_norm_weights(layer)
        elif isinstance(layer, ResNetBlock):
            weights[i] = get_resblock_weights(layer)
        elif isinstance(layer, AttentionBlock):
            weights[i] = get_attenblock_weights(layer)
        elif isinstance(layer, nn.Module):
            raise ValueError(f"Unsupported layer: {layer} of: {block}")
    return weights


def flatten_subweights(weights: NestedDict):
    flattened = {}
    for _, value in sorted(weights.items()):
        for key, sub_weights in sorted(value.items()):
            flattened[len(flattened)] = sub_weights
    return flattened


def unflatten_subweights(weights: NestedDict, mod: int):
    unflattened = {}
    for i, value in weights.items():
        unflattened.setdefault(i // mod, {})[i % mod] = value
    return unflattened


def set_unet_diffuser_weights(diffuser: UNetDiffuser, weights: NestedDict):
    # load the sigma embedding
    # weights
    assert isinstance(diffuser.time_embed, nn.Sequential)
    set_sequence_weights(diffuser.time_embed[1], weights["sig_embed"]["mlp"])

    # load the main weight sequence
    set_conv_weights(diffuser.unet.input_conv, weights["conv_in"])

    for i, down_block in weights["downs"].items():
        subblocks = (
            diffuser.unet.down_levels[i][:-1]
            if "downsample" in down_block else
            diffuser.unet.down_levels[i][:]
        )
        set_sequence_weights(
            subblocks, flatten_subweights(down_block["blocks"])
        )
        if "downsample" in down_block:
            set_conv_weights(
                diffuser.unet.down_levels[i][-1].conv,
                down_block["downsample"][1],
            )

    set_sequence_weights(diffuser.unet.middle_block, weights["mid"])

    for i, up_block in weights["ups"].items():
        subblocks = (
            diffuser.unet.up_levels[i][:-1]
            if "upsample" in up_block else
            diffuser.unet.up_levels[i][:]
        )
        set_sequence_weights(
            subblocks, flatten_subweights(up_block["blocks"])
        )
        if "upsample" in up_block:
            set_conv_weights(
                diffuser.unet.up_levels[i].layers[-1].conv,
                up_block["upsample"][1],
            )

    set_sequence_weights(diffuser.unet.out_final, weights["out_layer"])


def get_unet_diffuser_weights(diffuser: UNetDiffuser) -> NestedDict:
    weights = {}
    # get the sigma embedding weights
    assert isinstance(diffuser.time_embed, nn.Sequential)
    assert isinstance(diffuser.time_embed[1], MLP)
    weights["sig_embed"] = {
        "mlp": get_sequence_weights(diffuser.time_embed[1]),
    }
    # get the main weight sequence
    weights["conv_in"] = get_conv_weights(diffuser.unet.input_conv)
    weights["downs"] = {}
    for i, down_level in enumerate(diffuser.unet.down_levels):
        level_weights = {}
        has_attention = AttentionBlock in [type(b) for b in down_level]
        has_downsample = isinstance(down_level[-1], Downsample)
        level_weights["blocks"] = unflatten_subweights(
            get_sequence_weights(down_level[:-1] if has_downsample else down_level),
            2 if has_attention else 1,
        )
        if has_downsample:
            level_weights["downsample"] = {1: get_conv_weights(down_level[-1].conv)}
        weights["downs"][i] = level_weights

    weights["mid"] = get_sequence_weights(diffuser.unet.middle_block)

    weights["ups"] = {}
    for i, up_level in enumerate(diffuser.unet.up_levels):
        level_weights = {}
        has_attention = AttentionBlock in [type(b) for b in up_level]
        has_upsample = isinstance(up_level[-1], Upsample)
        level_weights["blocks"] = unflatten_subweights(
            get_sequence_weights(up_level[:-1] if has_upsample else up_level),
            2 if has_attention else 1,
        )
        if has_upsample:
            level_weights["upsample"] = {1: get_conv_weights(up_level[-1].conv)}
        weights["ups"][i] = level_weights

    weights["out_layer"] = get_sequence_weights(diffuser.unet.out_final)

    return weights


def to_nested(checkpoint: FlatDict) -> NestedDict:
    nested = {}
    for key, value in checkpoint.items():
        parts = key.split(".")
        current = nested
        for part in parts[:-1]:
            if part.isdigit():
                part = int(part)
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return nested


def from_nested(nested: NestedDict) -> FlatDict:
    def recurse(current: dict, path: str):
        for key, value in current.items():
            new_path = f"{path}.{key}" if path else str(key)
            if isinstance(value, dict):
                yield from recurse(value, new_path)
            else:
                yield new_path, value

    return dict(recurse(nested, ""))
