import json

import numpy as np
import pytest
import yaml

from a3_ultrasound.data.preprocessing import RFPreprocessing, preprocess_npz, preprocess_rf


def test_dc_polarity_and_frequency_response():
    cfg = RFPreprocessing(200e6, 131, 256)
    t = np.arange(4096) / cfg.sample_rate_hz
    x = np.stack([np.full(t.size, 65.5), -131 * np.ones(t.size),
                  131 * np.sin(2 * np.pi * 20e6 * t), 131 * np.sin(2 * np.pi * 50e6 * t)])
    result = preprocess_rf(x, cfg)
    np.testing.assert_allclose(result[0], 0.5, atol=1e-6)
    np.testing.assert_allclose(result[1], -1, atol=1e-6)
    np.testing.assert_allclose(result[2, 512:-512], 0.5 * x[2, 512:-512] / 131, atol=1e-5)
    assert np.max(np.abs(result[3, 512:-512])) < 0.002


def test_padding_is_not_filtered_and_polarity_is_linear():
    cfg = RFPreprocessing(200e6, 131, 256)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 600))
    mask = np.arange(600)[None, :] < np.array([[400], [600]])
    x[~mask] = np.nan
    y = preprocess_rf(x, cfg, mask)
    np.testing.assert_array_equal(y[0, :400], preprocess_rf(x[:1, :400], cfg)[0])
    assert np.all(y[~mask] == 0)
    np.testing.assert_array_equal(preprocess_rf(-x, cfg, mask), -y)


@pytest.mark.parametrize("case", ["short", "nan", "gap"])
def test_invalid_input(case):
    x, mask = np.ones((1, 400)), np.ones((1, 400), dtype=bool)
    if case == "short":
        mask[:, 256:] = False
    elif case == "nan":
        x[0, 50] = np.nan
    else:
        mask[0, 20] = False
    with pytest.raises(ValueError):
        preprocess_rf(x, RFPreprocessing(200e6, 131, 256), mask)


@pytest.mark.parametrize("fs,div,pad", [(40e6, 131, 256), (200e6, 0, 256), (200e6, 131, -1)])
def test_invalid_parameters(fs, div, pad):
    with pytest.raises(ValueError):
        RFPreprocessing(fs, div, pad)


def test_npz_preserves_metadata_and_refuses_reprocessing_or_overwrite(tmp_path):
    profile = {"schema_version": "a3-rf-preprocessing-v1", "preprocessing_id": "test-v1",
               "acquisition": {"probe_mhz": 5, "gain_db": 18, "batch_id": "test"},
               "processing": {"sample_rate_hz": 200e6, "normalization_divisor": 131, "padlen": 256}}
    cfg, src, dst = tmp_path / "p.yaml", tmp_path / "raw.npz", tmp_path / "processed.npz"
    cfg.write_text(yaml.safe_dump(profile))
    np.savez(src, x_raw=np.ones((2, 500)), sample_ids=np.array(["a", "b"]), y=np.array([[0], [1]]))
    preprocess_npz(src, dst, cfg)
    with np.load(dst) as arrays:
        np.testing.assert_array_equal(arrays["y"], [[0], [1]])
        assert arrays["sample_ids"].tolist() == ["a", "b"]
        assert json.loads(str(arrays["preprocessing_record"]))["processing"]["baseline_subtracted"] is False
    with pytest.raises(ValueError, match="Already processed"):
        preprocess_npz(dst, tmp_path / "again.npz", cfg)
    with pytest.raises(FileExistsError):
        preprocess_npz(src, dst, cfg)
