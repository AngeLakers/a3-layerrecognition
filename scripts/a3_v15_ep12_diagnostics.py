"""Frozen V1.4 ep12, CPU-only attention and independently re-enumerated correction ablation."""
from __future__ import annotations
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frozen-root', type=Path, required=True)
    ap.add_argument('--catalog', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    args = ap.parse_args(); root = args.frozen_root
    sys.path.insert(0, str(root/'deployment/code/src'))
    from a3_ultrasound.training import ModelConfig, build_model
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.inference.enumerator import enumerate_pcc
    from a3_ultrasound.stream_metrics import PredictionAccumulator
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    source = root/'analysis_inputs/proposed_ep12'
    manifest = read(source/'source_manifest.json')
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest
    cp = torch.load(source/'epoch_12.pt', map_location='cpu', weights_only=True)
    cfg = cp['resolved_config']; assert cp['epoch'] == 12 and not cfg['test_opened']
    assert cfg['commit_hash'] == 'dbab7132403ad4c3bd3d43eb0f764745da445738'
    data = RFShardDataset(args.catalog, cfg['fold']['validation'])
    data.use_frozen_evaluation_selection(root/'deployment/validation_selection/F1.npz')
    assert len(data) == 40002 and int(data.evaluation_weights.sum()) == 714068
    model = build_model(ModelConfig(**cfg['model'])).cpu().eval()
    model.load_state_dict(cp['model_state_dict'], strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    attention = defaultdict(list); paired = []; sample_rows = []
    metrics = {mode: defaultdict(PredictionAccumulator) for mode in ('on', 'off')}
    def off(*values):
        result = model.head(*values)
        return replace(result, logits=result.base_logodds,
                       correction=torch.zeros_like(result.correction))
    started = time.perf_counter()
    with torch.inference_mode():
        for batch, indices in data.batches(args.batch_size, 'cpu'):
            encoded = model.encode(batch); tokens = encoded.responsibility_tokens
            outputs = {'on': model.infer_encoded(encoded),
                       'off': enumerate_pcc(off, tokens.r_loc, tokens.r_ctx, tokens.r_gate,
                                            encoded.query, encoded.interface_mask)}
            diagnostic = model.conditional(encoded, outputs['on'].map_states)
            b = diagnostic.base_logodds; basep = b.sigmoid()
            counts = batch.interface_mask.sum(1).tolist()
            centers = encoded.backbone.feature_centers
            if centers.ndim == 1:
                centers = centers[None].expand(len(indices), -1)
            half = float(cfg['input_config']['window_total_us']) * 1e-6 / 2 * cfg['input_config']['sampling_rate_hz']
            for i, index in enumerate(indices):
                sample, board = data.identity(index); k = counts[i]
                y = batch.y[i, :k].tolist(); lm = batch.label_mask[i, :k].tolist()
                w = int(data.evaluation_weights[index]); predictions = {}
                for mode, exact in outputs.items():
                    path = exact.path_mask[i]; pred = exact.map_states[i, :k].tolist()
                    predictions[mode] = pred
                    row = dict(sample_id=sample, specimen_id=board, K=k, y=y, label_mask=lm,
                        sample_weight=w, marginals=exact.marginals[i, :k].tolist(), map_states=pred,
                        probabilities=exact.probabilities[i, path].tolist(),
                        log_probabilities=exact.log_probabilities[i, path].tolist(),
                        base_logodds=b[i, :k].tolist(), base_probabilities=basep[i, :k].tolist(),
                        raw_logZ=float(exact.raw_log_normalizer[i]), rho=None,
                        conditional_probability_range=exact.conditional_probability_range[i, :k].tolist(),
                        indexed_local_token_counts=encoded.local_token_diagnostics.counts[i, :k].tolist(),
                        gate=None, correction=None, gate_times_correction=None)
                    metrics[mode]['all'].add(row); metrics[mode][board].add(row)
                sample_rows.append(dict(sample_id=sample, board=board, K=k, y=y, label_mask=lm,
                    sample_weight=w, on=predictions['on'], off=predictions['off']))
                if all(lm):
                    paired.append((board, w, any(y), predictions['on']==y, predictions['off']==y,
                                   predictions['on'] != predictions['off']))
                for j in range(k):
                    support = encoded.pre_support[i, j]
                    front = support & (centers[i] >= 0) & (centers[i] <= half)
                    upstream = torch.zeros_like(front)
                    for q in range(j):
                        upstream |= (centers[i] >= encoded.window_start[i,q]) & (centers[i] <= encoded.window_end[i,q])
                    upstream &= support & ~front
                    other = support & ~front & ~upstream
                    a = encoded.pre_readout.attention[i,j]
                    mass = [float(a[region].sum()) for region in (front,upstream,other)]
                    assert abs(sum(mass)-(1 if bool(support.any()) else 0)) < 2e-6
                    attention['board'].append(board); attention['k'].append(j+1)
                    attention['weight'].append(w); attention['mass'].append(mass)
            if (int(indices[-1])+1) % 2560 == 0 or int(indices[-1])+1 == len(data):
                progress = dict(records=int(indices[-1])+1,total=40002,seconds=time.perf_counter()-started,cpu=True)
                write(args.output/'progress.json', progress); print(json.dumps(progress), flush=True)
    final = {mode:{board:acc.result() for board,acc in groups.items()} for mode,groups in metrics.items()}
    expected = read(source/'history.json')[11]['validation']['reliable']
    for name in ('positive_joint_map_accuracy','normal_joint_false_positive_rate'):
        assert final['on']['all'][name] == expected[name]
    assert abs(final['on']['all']['joint_nll']-expected['joint_nll']) < 1e-5
    def pair_table(rows):
        return dict(raw_n=len(rows), ipw_n=sum(r[1] for r in rows),
            on_correct_off_wrong_raw=sum(r[3] and not r[4] for r in rows),
            on_wrong_off_correct_raw=sum(not r[3] and r[4] for r in rows),
            on_correct_off_wrong_ipw=sum(r[1] for r in rows if r[3] and not r[4]),
            on_wrong_off_correct_ipw=sum(r[1] for r in rows if not r[3] and r[4]),
            changed_joint_prediction_raw=sum(r[5] for r in rows))
    pairs = {board:{subset:pair_table([r for r in paired if (board=='all' or r[0]==board) and
                    (subset=='all' or r[2]==(subset=='abnormal'))]) for subset in ('all','abnormal','normal')}
             for board in ['all']+data.entities}
    arrays = {name:np.asarray(value) for name,value in attention.items()}
    np.savez_compressed(args.output/'pre_attention.npz', **arrays)
    groups = []
    for board in ['all']+data.entities:
        for k in sorted(set(arrays['k'])):
            mask=(arrays['k']==k)&((arrays['board']==board) if board!='all' else True)
            if mask.any():
                groups.append(dict(board=board,k=int(k),raw_n=int(mask.sum()),
                    q25_median_q75=np.quantile(arrays['mass'][mask],[.25,.5,.75],axis=0).tolist()))
    nonfirst=arrays['k']>=2
    median=float(np.median(arrays['mass'][nonfirst,0]))
    result=dict(epoch=12,device='cpu',evaluation_batch_size=args.batch_size,
        original_evaluation_batch_size=cfg['evaluation_batch_size'],
        training_called=False,outer_opened=False,source_commit=cfg['commit_hash'],
        metrics=final,paired=pairs,attention_groups=groups,
        D2_original_nominal_zone=dict(front_mass_nonfirst_raw_median=median,threshold=.5,
            nominal_rule_would_enable_M6=median>.5,
            decision_status='NOT_FROZEN: nominal front-window physical coverage failed train-only QA'),
        scope='same fixed checkpoint inference ablation; not a retrained baseline or generalization claim')
    write(args.output/'ep12_reenumeration.json',result)
    with (args.output/'paired_predictions.jsonl').open('w',encoding='utf-8') as f:
        for row in sample_rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    lines=['# V1.4 第12轮：修正开/关分别完整枚举','',result['scope'],'',
        '| 范围 | 开启异常联合正确率 | 关闭异常联合正确率 | 开启正常误报 | 关闭正常误报 |',
        '| --- | --- | --- | --- | --- |']
    for board in ['all']+data.entities:
        on=final['on'][board];offm=final['off'][board]
        lines.append(f"| {board} | {on['positive_joint_map_accuracy']:.4%} | {offm['positive_joint_map_accuracy']:.4%} | {on['normal_joint_false_positive_rate']:.4%} | {offm['normal_joint_false_positive_rate']:.4%} |")
    lines+=['', '完整配对计数、NLL和多缺陷诊断见JSON；逐样本真值、mask、IPW与两种预测见paired_predictions.jsonl。',
        f'原[0,70]区域非首界面注意力质量中位数={median:.8g}；由于训练板窗口覆盖检查失败，该数值不直接冻结M6。']
    (args.output/'ep12_reenumeration.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='COMPLETE',records=40002,seconds=time.perf_counter()-started)))


if __name__=='__main__':main()
