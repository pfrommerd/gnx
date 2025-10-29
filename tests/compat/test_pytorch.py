import jax
import numpy as np
import jax.numpy as jnp
import gnx.core.nn as nn

def test_gn():
    import torch
    import torch.nn as tnn

    gn = nn.GroupNorm(128, 2, 32, rngs=nn.Rngs(42), use_fast_variance=True)
    # by default should have scale 1 bias 0
    tgn = tnn.GroupNorm(32, 128, eps=1e-6)
    input = jax.random.normal(jax.random.key(42), (1, 32, 32, 128))
    out = gn(input)

    input = input.transpose(0, 3, 1, 2)
    with torch.no_grad():
        tout = tgn(torch.from_numpy(np.array(input)))
        tout = tout.cpu().numpy().transpose(0, 2, 3, 1)

    diff = np.max(np.abs(out - tout))
    print("GroupNorm difference", diff)
    assert diff < 1e-5, f"Got gn difference of {diff}"


def test_conv():
    import torch
    import torch.nn as tnn

    conv = nn.Conv(3, 32, (3, 3), padding=1, rngs=nn.Rngs(42))
    # by default should have scale 1 bias 0
    tconv = tnn.Conv2d(3, 32, 3, padding=1)
    assert conv.bias is not None and tconv.bias is not None
    tconv.weight.data.copy_(torch.from_numpy(np.copy(conv.kernel[...].transpose(3, 2, 0, 1))))
    tconv.bias.data.copy_(torch.from_numpy(np.copy(conv.bias[...])))

    input = jax.random.normal(jax.random.key(42), (24, 32, 32, 3))
    out = conv(input)
    with torch.no_grad():
        tout = tconv(torch.from_numpy(np.array(input.transpose(0, 3, 1, 2))))
    tout = tout.cpu().numpy().transpose(0, 2, 3, 1)
    diff = np.max(np.abs(out - tout))
    print("Conv difference", diff)
    assert diff < 1e-5, f"Got conv difference of {diff}"

if __name__=="__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    jax.config.update('jax_platform_name', 'cpu')

    test_conv()
    test_gn()