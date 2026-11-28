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

def test_board_interval_scale_holds_its_unit_when_reference_moves():
    from pytest import raises

    cfg = ModelConfig(
        **(FIXED_MODEL | {"architecture_version": "a3-v1.8"}),
        backbone="resnet1d", head_type="local_independent", relative_time_attention=True,
        readout_scope="full_waveform",
    )
    torch.manual_seed(23)
    old = build_model(cfg).eval()
    model = build_model(replace(cfg, relative_time_scale="board_interval")).eval()
    model.load_state_dict(old.state_dict(), strict=True)
    base = synthetic_datasets(23, 4)["train"].batch
    nominal = replace(base, board_interval_samples=torch.tensor([180., 232., 576., 185.]))
    with raises(ValueError, match="Missing frozen board interval"), torch.inference_mode():
        model.encode(base)

    with torch.inference_mode():
        torch.testing.assert_close(old.encode(base).local_readout.token,
                                   model.encode(nominal).local_readout.token, rtol=0, atol=0)
        model.local_readout.relative_time_weight.fill_(0.7)
        old.local_readout.relative_time_weight.copy_(model.local_readout.relative_time_weight)
        unit80 = replace(base, board_interval_samples=torch.full((4,), 80.))
        torch.testing.assert_close(old.encode(base).local_readout.attention,
                                   model.encode(unit80).local_readout.attention, rtol=0, atol=0)
        captured = []
        raw_scores = []
        score_hook = model.local_readout.score_projection.register_forward_hook(
            lambda module, args, scores: raw_scores.append(scores.squeeze(-1).clone()))
        handle = model.local_readout.register_forward_pre_hook(
            lambda module, args, kw: captured.append(kw["relative_time"].clone()), with_kwargs=True)
        before = model.encode(nominal)
        meta = nominal.interface_meta.clone()
        length = nominal.length_mask.sum(1).float()
        meta[:, 1, 1] += 2. / length
        meta[:, 1, 4] = torch.minimum(meta[:, 1, 4], 1. - 1./length - meta[:, 1, 1])
        after = model.encode(replace(nominal, interface_meta=meta))
        handle.remove()
        score_hook.remove()
        valid = before.local_support[0, 1]
        delta = (raw_scores[1] - raw_scores[0])[0, 1, valid]
        assert (delta - delta.mean()).abs().max() > 1e-7
        displacement = captured[1] - captured[0]
        expected = (-2. / nominal.board_interval_samples)[:, None].expand_as(displacement[:, 1])
        torch.testing.assert_close(displacement[:, 1], expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(displacement[:, 0], torch.zeros_like(displacement[:, 0]))
        torch.testing.assert_close(before.query, after.query, rtol=0, atol=0)
        assert not torch.equal(before.local_readout.attention[:, 1], after.local_readout.attention[:, 1])
