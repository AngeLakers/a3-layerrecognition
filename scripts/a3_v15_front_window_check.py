"""CPU, train-only fixed-window audit. No training or label/trace modification."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.frozen_root / 'deployment/code/src'))
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.data.quick_input import physical_manifest
    from a3_ultrasound.data.time_prior import derive_time_prior_samples
    cfg = json.loads((a.frozen_root / 'analysis_inputs/proposed_ep12/resolved_config.json').read_text(encoding='utf-8'))
    data = RFShardDataset(a.catalog, cfg['fold']['train'], training=True, seed=23)
    output = []
    profiles = {}
    for shard in data.shards:
        arr = shard['arrays']; eligible = np.flatnonzero(arr['eligible'])
        # Deterministic spatial-order coverage, balanced only for waveform QA.
        ids = []
        for abnormal in (False, True):
            group = eligible[arr['y'][eligible].any(1) == abnormal]
            ids.extend(group[np.linspace(0, len(group)-1, min(512, len(group)), dtype=int)].tolist())
        ids = np.unique(ids)
        peaks = []; fractions = []; energies = []; front_peaks = []
        for i in ids:
            offset = int(arr['offsets'][i]); length = int(arr['lengths'][i])
            x = np.asarray(shard['rf'][offset:offset+length], dtype=np.float64) / data.config['normalization_divisor']
            prior = derive_time_prior_samples(physical_manifest(shard, data.config), trace_length=length)
            right = int(math.floor(prior.window_right_samples))
            pre_end = int(math.ceil(prior.window_start_samples[0]))
            front = x[:right+1]; pre = x[:pre_end]
            e = float(np.dot(front, front)); all_e = float(np.dot(pre, pre))
            peaks.append(int(np.argmax(np.abs(pre))))
            fractions.append(e / all_e if all_e else 0.)
            energies.append(e); front_peaks.append(float(np.max(np.abs(front))))
        peaks = np.asarray(peaks); fractions = np.asarray(fractions)
        profiles[shard['entity']] = dict(sample_rows=ids, peak_index=peaks,
            front_energy=np.asarray(energies), front_fraction_pre=fractions)
        output.append(dict(entity=shard['entity'], K=int(shard['K']), layer_mm=shard['layer_mm'],
            sampled_records=len(ids), eligible_records=len(eligible), right_samples=right,
            first_local_start=pre_end, peak_inside_front_fraction=float(np.mean(peaks <= right)),
            peak_index_quantiles=np.quantile(peaks, [0,.25,.5,.75,1]).tolist(),
            front_energy_quantiles=np.quantile(energies, [0,.25,.5,.75,1]).tolist(),
            front_energy_le_epsilon=int(np.count_nonzero(np.asarray(energies) <= 1e-8)),
            energy_fraction_of_pre_quantiles=np.quantile(fractions, [0,.25,.5,.75,1]).tolist()))
    a.output.mkdir(parents=True, exist_ok=False)
    for board, arrays in profiles.items():
        np.savez_compressed(a.output / (board+'.npz'), **arrays)
    result = dict(scope='F1 train-only deterministic QA subset; not population estimate or front-echo identity proof',
        cpu_only=True, training_called=False, outer_opened=False, source_catalog=str(a.catalog),
        catalog_sha256=hashlib.sha256(a.catalog.read_bytes()).hexdigest(), rows=output)
    (a.output/'front_window_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# V1.5 前表面固定窗口核查', '', '仅F1训练板，按完整标签正常/异常分别均匀抽最多512条，用于波形核查；不作为总体比例估计。',
        '检查窗口为0至70采样点（含端点）。前缀最大幅值的位置用于发现覆盖问题，不能单独证明波包的物理身份。', '',
        '| 板 | 样本数 | 前缀最大幅值落入窗口比例 | 峰位置中位数 | 前窗能量中位数 | 前窗/首界面前能量中位数 |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in output:
        lines.append(f"| {r['entity']} | {r['sampled_records']} | {r['peak_inside_front_fraction']:.2%} | {r['peak_index_quantiles'][2]:.1f} | {r['front_energy_quantiles'][2]:.6g} | {r['energy_fraction_of_pre_quantiles'][2]:.4%} |")
    (a.output/'front_window_check.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
