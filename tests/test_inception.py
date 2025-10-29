from gnx.models.common.inception import InceptionV3
from pytorch_fid.inception import InceptionV3 as TorchInceptionV3

import jax.numpy as jnp
import numpy as np
import torch
import jax
import warnings


def test_inception():
    jax_model = InceptionV3.load_fid_pretrained(
        include_head=False, precision=jax.lax.Precision.HIGHEST
    )
    # disable normalization as the jax model assumes (-1, 1)
    # input range
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch_model = TorchInceptionV3(normalize_input=False)
        torch_model.eval()

    input = jax.random.normal(jax.random.key(42), (1, 299, 299, 3)) / 2.0
    jax_output = jax_model(input)
    (torch_output,) = torch_model(torch.tensor(np.copy(input.transpose(0, 3, 1, 2))))
    torch_output = (
        torch_output.detach().cpu().numpy().squeeze(-1).squeeze(-1).squeeze(0)
    )
    jax_output = np.array(jax_output).squeeze(0)
    diff = np.max(np.abs(jax_output - torch_output))
    print(f"Got inception difference of {diff}")
    assert diff < 1e-5


if __name__ == "__main__":
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    test_inception()
