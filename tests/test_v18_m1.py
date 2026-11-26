from dataclasses import replace

import torch

from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.training import ModelConfig, build_model, synthetic_datasets
from a3_ultrasound.v171_protocol import FIXED_MODEL


def test_m1_readouts_share_initialization_and_support_exact_inference():
    config = ModelConfig(
        **(FIXED_MODEL | {"architecture_version": "a3-v1.8"}),
        backbone="resnet1d",
        head_type="local_independent",
        relative_time_attention=True,
    )
    batch = synthetic_datasets(23, 4)["train"].batch
    torch.manual_seed(23)
    global_model = build_model(replace(config, readout_scope="full_waveform"))
    torch.manual_seed(23)
    local_model = build_model(replace(config, readout_scope="indexed"))

    global_parameters = dict(global_model.named_parameters())
    local_parameters = dict(local_model.named_parameters())
    assert global_parameters.keys() == local_parameters.keys()
    for name, parameter in global_parameters.items():
        torch.testing.assert_close(parameter, local_parameters[name], rtol=0, atol=0)

    for model in (global_model, local_model):
        encoded = model.encode(batch)
        output = model.conditional(encoded, batch.y)
        loss = head_training_loss(output, batch, head_type=config.head_type)
        loss.total.backward()
        assert model.local_readout.relative_time_weight.grad.abs().sum() > 0
        with torch.no_grad():
            exact = model.infer_encoded(encoded)
        assert exact.path_mask.sum(1).tolist() == [4, 8, 4, 8]
        torch.testing.assert_close(exact.raw_log_normalizer, torch.zeros(4), atol=1e-6, rtol=0)
