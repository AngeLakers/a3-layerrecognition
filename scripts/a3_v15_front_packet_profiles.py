"""Inspect saved train-only QA identities; no fitted classifier or eval RF."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--development', type=Path, required=True)
    p.add_argument('--frozen-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.frozen_root/'deployment/code/src'))
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.data.quick_input import physical_manifest
    from a3_ultrasound.data.time_prior import derive_time_prior_samples
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qa = json.loads((a.development/'diagnostics/front_window/front_window_check.json').read_text(encoding='utf-8'))
    cfg = json.loads((a.frozen_root/'analysis_inputs/proposed_ep12/resolved_config.json').read_text(encoding='utf-8'))
    data = RFShardDataset(qa['source_catalog'], cfg['fold']['train'], training=True, seed=23)
    a.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(7, 2, figsize=(14, 19))
    result = []
    for row, shard in enumerate(data.shards):
        with np.load(a.development/'diagnostics/front_window'/(shard['entity']+'.npz')) as z:
            ids = z['sample_rows']
        ar = shard['arrays']
        traces = []
        for i in ids:
            off = int(ar['offsets'][i]); length = int(ar['lengths'][i])
            assert length >= 450
            traces.append(np.asarray(shard['rf'][off:off+450], np.float64)/data.config['normalization_divisor'])
        traces = np.stack(traces)
        prior = derive_time_prior_samples(physical_manifest(shard, data.config), trace_length=length)
        # Symmetric 21-sample RMS for visualization only; RF energy remains unsmoothed.
        rms = np.sqrt(np.stack([np.convolve(x*x, np.ones(21)/21, 'same') for x in traces]))
        norm = rms/np.max(rms[:, :120], axis=1, keepdims=True)
        q = np.quantile(norm, [.1,.25,.5,.75,.9], axis=0)
        name = shard['entity'] if shard['entity'].isascii() else 'AL thin'
        ax = axes[row,0]
        for i in [0,len(ids)//2,len(ids)-1]: ax.plot(traces[i], lw=.7, alpha=.7, label=str(int(ids[i])))
        ax.set_title(name+' RF (QA rows)'); ax.legend(fontsize=7)
        ax = axes[row,1]
        ax.fill_between(np.arange(450), q[0], q[4], alpha=.15)
        ax.fill_between(np.arange(450), q[1], q[3], alpha=.25)
        ax.plot(q[2]); ax.set_ylim(0,1.5); ax.set_title(name+' RMS / early RMS peak')
        for ax in axes[row]:
            ax.axvline(70,color='red',ls='--',lw=1)
            ax.axvline(prior.window_start_samples[0],color='green',ls='--',lw=1)
            ax.axvline(prior.nominal_tof_samples[0],color='green',ls=':',lw=1)
            ax.grid(alpha=.2); ax.set_xlim(0,450)
        np.savez_compressed(a.output/(shard['entity']+'.npz'), sample_rows=ids, traces=traces, rms=rms, normalized_rms_quantiles=q)
        result.append(dict(board=shard['entity'], first_window_start=float(prior.window_start_samples[0]), first_tof=float(prior.nominal_tof_samples[0]), n=len(ids)))
    fig.tight_layout(); fig.savefig(a.output/'train_packet_profiles.png',dpi=130); plt.close(fig)
    (a.output/'profiles.json').write_text(json.dumps(dict(scope='train-only QA waveform morphology, not physical identity proof', rows=result),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__ == '__main__':
    main()
