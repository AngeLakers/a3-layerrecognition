from dataclasses import replace
from pathlib import Path
import sys

import torch

from a3_ultrasound import stream_training, training, v16_protocol
from a3_ultrasound.schema import InterfaceBatch
from scripts.a3_v17_g1d_runner_full import install


REPOSITORY = Path(__file__).resolve().parents[1]


def test_r31_adapter_builds_g1d_and_preserves_legacy_parameters(monkeypatch):
    legacy = training.ModelConfig(
        architecture_version="a3-v1.6",
        backbone="tcn",
        backbone_norm="group8",
        query_view="position2",
        head_type="local_pre_independent",
    )
    original_factory = training.build_model
    torch.manual_seed(23)
    original_model = original_factory(legacy)
    original_parameters = original_model.state_dict()

    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "a5_g1d_model", None)
    for module, names in (
        (training, ("build_model", "evaluate")),
        (stream_training, ("build_model", "evaluate", "head_training_loss")),
        (v16_protocol, ("validate_config",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, getattr(module, name))

    install(REPOSITORY, REPOSITORY / "scripts/a3_v17_g1d_model.py", REPOSITORY)
    torch.manual_seed(23)
    adapted_model = training.build_model(legacy)
    assert type(adapted_model) is type(original_model)
    adapted_parameters = adapted_model.state_dict()
    assert adapted_parameters.keys() == original_parameters.keys()
    for name, value in original_parameters.items():
        torch.testing.assert_close(adapted_parameters[name], value, rtol=0, atol=0)

    model = training.build_model(replace(legacy, backbone="resnet1d", head_type="g1d")).eval()
    assert model.head_type == "g1d"
    assert model.queries.num_embeddings == 3
    mask = torch.tensor([[True, True, False, False, False, False],
                         [True, True, True, False, False, False]])
    batch = InterfaceBatch(
        x_raw=torch.randn(2, 48),
        length_mask=torch.ones(2, 48, dtype=torch.bool),
        interface_meta=torch.zeros(2, 6, 5),
        interface_mask=mask,
    )
    with torch.inference_mode():
        encoding = model.encode(batch)
        result = model.infer_encoded(encoding)
        changed = model.infer_exact(replace(batch, interface_meta=torch.ones_like(batch.interface_meta)))
    assert not encoding.attention[~mask].any()
    assert not encoding.local_token_diagnostics.counts[~mask].any()
    assert result.path_mask.sum(1).tolist() == [4, 8]
    assert (result.map_states[~mask] == -1).all()
    torch.testing.assert_close(result.probabilities.sum(1), torch.ones(2))
    torch.testing.assert_close(result.probabilities, changed.probabilities, rtol=0, atol=0)
