"""A--E diagnosis only. Does not import training or modify the model input path."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a3_ultrasound.data.timebase_calibration import first_negative_trough, resolve_saturation, timing_stats, complete_negative_trough


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def fit_boards(spec, output):
    """The A--C load allowlist excludes every label/cohort/quality array."""
    fs = spec['sampling_rate_hz']
    boards = []
    access = []
    load = np.load
    permitted = {str(Path(b[key]).resolve()) for b in spec['boards']
                 for key in ('rf_path', 'offsets_path', 'lengths_path', 'source_row_path')}

    def rf_only_load(file, *args, **kwargs):
        path = str(Path(file).resolve())
        assert path in permitted, 'A--C attempted to open a non-RF/index array.'
        assert kwargs.get('mmap_mode') == 'r', 'Source arrays must be read-only.'
        access.append(path)
        return load(file, *args, **kwargs)

    np.load = rf_only_load
    try:
        for b in spec['boards']:
            rf = np.load(b['rf_path'], mmap_mode='r', allow_pickle=False)
            off = np.load(b['offsets_path'], mmap_mode='r', allow_pickle=False)
            lengths = np.load(b['lengths_path'], mmap_mode='r', allow_pickle=False)
            source_row = np.load(b['source_row_path'], mmap_mode='r', allow_pickle=False)
            assert rf.ndim == 1 and len(off) == len(lengths) == len(source_row) == b['records']
            assert np.all(lengths >= 201) and np.all(off >= 0) and np.all(off + lengths <= len(rf))
            before = dict(bytes=Path(b['rf_path']).stat().st_size,
                          mtime_ns=Path(b['rf_path']).stat().st_mtime_ns)
            n = len(off)
            t0 = np.full(n, -1, np.int32); bw = np.full(n, -1, np.int32)
            sat = np.zeros(n, bool); bwsat = np.zeros(n, bool)
            unclosed = np.zeros(n, bool); bwunclosed = np.zeros(n, bool)
            boundary = np.zeros(n, bool)
            nominal_distance = 2*(b['K']+1)*b['layer_mm']*1e-3*fs/b['nominal_c_mps']
            for start in range(0, n, 2048):
                ids = np.arange(start, min(n, start+2048))
                x = np.asarray(rf[off[ids, None]+np.arange(201)], np.float32)
                found = first_negative_trough(x)
                record_min = np.full(len(ids), np.inf)
                for j in np.flatnonzero(found.plateau):
                    i = ids[j]
                    record_min[j] = np.min(rf[int(off[i]):int(off[i]+lengths[i])])
                pos, saturated = resolve_saturation(found, record_min)
                t0[ids] = pos; sat[ids] = saturated; unclosed[ids] = found.unclosed
                for j in np.flatnonzero(found.unclosed):
                    i=ids[j]
                    exact=complete_negative_trough(rf[int(off[i]):int(off[i]+lengths[i])],0,200)
                    t0[i]=exact['position'];sat[i]=exact['saturated'];unclosed[i]=exact['unclosed']
                pos=t0[ids]
                valid = ids[pos >= 0]
                if not len(valid):
                    continue
                centers = t0[valid] + nominal_distance
                left = np.ceil(centers-.65e-6*fs).astype(np.int64)
                right = np.floor(centers+.65e-6*fs).astype(np.int64)
                boundary[valid] = (left < 0) | (right >= lengths[valid])
                good = ~boundary[valid]
                valid = valid[good]; left = left[good]; right = right[good]
                if not len(valid):
                    continue
                widths = right-left+1
                for width in np.unique(widths):
                    sel = widths == width
                    ix = valid[sel]; origins = left[sel]
                    x = np.asarray(rf[off[ix, None]+origins[:, None]+np.arange(width)], np.float32)
                    found = first_negative_trough(x, origins)
                    record_min = np.full(len(ix), np.inf)
                    for j in np.flatnonzero(found.plateau):
                        i = ix[j]
                        record_min[j] = np.min(rf[int(off[i]):int(off[i]+lengths[i])])
                    pos, saturated = resolve_saturation(found, record_min)
                    bw[ix] = pos; bwsat[ix] = saturated; bwunclosed[ix] = found.unclosed
                    for j in np.flatnonzero(found.unclosed):
                        i=ix[j]
                        exact=complete_negative_trough(rf[int(off[i]):int(off[i]+lengths[i])],int(origins[j]),int(origins[j]+width-1))
                        bw[i]=exact['position'];bwsat[i]=exact['saturated'];bwunclosed[i]=exact['unclosed']
            valid = (t0 >= 0) & (bw >= 0)
            dbw = bw[valid]-t0[valid]
            ts = timing_stats(t0); ds = timing_stats(dbw)
            delta = ds['median']/(b['K']+1) if ds['count'] else None
            after = dict(bytes=Path(b['rf_path']).stat().st_size,
                         mtime_ns=Path(b['rf_path']).stat().st_mtime_ns)
            assert before == after, 'RF changed during diagnosis.'
            result = dict(board=b['board'], K=b['K'], layer_mm=b['layer_mm'], records=n,
                          nominal_c_mps=b['nominal_c_mps'], t0=ts,
                          saturated_count=int(sat.sum()), saturated_ratio=float(sat.mean()),
                          failed_count=int((t0 < 0).sum()), failed_ratio=float((t0 < 0).mean()),
                          t0_unclosed_count=int(unclosed.sum()), backwall=ds,
                          backwall_failed_count=int(((t0 >= 0) & (bw < 0)).sum()),
                          backwall_failed_ratio=float(np.mean((t0 >= 0) & (bw < 0))),
                          backwall_boundary_count=int(boundary.sum()),
                          backwall_saturated_count=int(bwsat.sum()),
                          backwall_unclosed_count=int(bwunclosed.sum()),
                          delta_samples=delta, delta_us=None if delta is None else delta/fs*1e6,
                          c_eff_mps=None if delta is None else 2*b['layer_mm']*1e-3*fs/delta,
                          rf_provenance=dict(path=b['rf_path'], before=before, after=after,
                                             mode='read-only mmap; no full RF rehash'),
                          index_sha256={key:digest(b[key+'_path']) for key in ('offsets','lengths','source_row')})
            np.savez_compressed(output/(b['board']+'.npz'), t0=t0, t_bw=bw, saturated=sat,
                                t0_unclosed=unclosed, backwall_saturated=bwsat,
                                backwall_unclosed=bwunclosed, backwall_boundary=boundary,
                                source_row=np.asarray(source_row))
            boards.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
    finally:
        np.load = load
        save(output/'rf_only_access_audit.json', dict(stage='A--C', status='PASS',
             allowed_paths=sorted(permitted), opened_paths=access,
             forbidden_label_access_count=0, all_source_array_modes='r'))
    per_board = all(b['t0']['count'] and b['t0']['iqr'] <= 10 for b in boards)
    ref = float(np.median([b['t0']['median'] for b in boards]))
    return dict(boards=boards, selected_scheme='per_board' if per_board else 'per_record_shift',
                t0_ref_samples=ref, calibration_inputs='All RF records, labels not opened',
                calibration_permitted_for_training=False)


def validate_train(spec, config, fit, output):
    """D only: open labels for the seven F1 training entities, never held-out."""
    train = config['folds'][0]['train']
    assert config['folds'][0]['fold'] == 'F1' and len(train) == 7
    assert not set(train) & set(config['folds'][0]['test'])
    rng = np.random.default_rng(23)
    rows = []; audit = []; arrays = {}
    fs = spec['sampling_rate_hz']
    for b, fitted in zip(spec['boards'], fit['boards']):
        assert b['board'] == fitted['board']
        if b['board'] not in train:
            continue
        base = Path(b['offsets_path']).parent
        labels = {}
        for name in ('y','label_mask','eligible'):
            path = base/(name+'.npy')
            labels[name] = np.load(path, mmap_mode='r', allow_pickle=False)
            audit.append(dict(board=b['board'], path=str(path), sha256=digest(path)))
        y = labels['y']; lm = labels['label_mask']; eligible = labels['eligible']
        rf = np.load(b['rf_path'], mmap_mode='r', allow_pickle=False)
        off = np.load(b['offsets_path'], mmap_mode='r', allow_pickle=False)
        lengths = np.load(b['lengths_path'], mmap_mode='r', allow_pickle=False)
        source = np.load(b['source_row_path'], mmap_mode='r', allow_pickle=False)
        with np.load(output/'rf_only'/(b['board']+'.npz'), allow_pickle=False) as z:
            t0 = z['t0']
        for k in range(1,b['K']+1):
            candidates = np.flatnonzero(eligible & lm[:, k-1] & (y[:, k-1] == 1))
            assert len(candidates) >= 50, 'Insufficient F1 train defect records; no silent count change.'
            ids = rng.choice(candidates, 50, replace=False)
            nominal = spec['batch_reference_samples'] + 2*k*b['layer_mm']*1e-3*fs/b['nominal_c_mps']
            delta = fitted['delta_samples']
            calibrated = ((np.full(50, fitted['t0']['median']) if fit['selected_scheme'] == 'per_board'
                           else t0[ids].astype(float)) + k*delta) if delta is not None else np.full(50,np.nan)
            observed = np.full(50,np.nan); observed_cal = np.full(50,np.nan)
            saturated = np.zeros(50,bool); boundary = np.zeros(50,bool)
            saved_windows = []
            for j,i in enumerate(ids):
                raw = rf[int(off[i]):int(off[i]+lengths[i])]
                full_min = None
                for alternate, center in ((False, nominal), (True, calibrated[j])):
                    if not np.isfinite(center) or t0[i] < 0:
                        continue
                    left = math.ceil(center-.65e-6*fs); right = math.floor(center+.65e-6*fs)
                    if left < 0 or right >= len(raw):
                        boundary[j] = True; continue
                    window = np.asarray(raw[left:right+1], np.float32)
                    found = first_negative_trough(window[None], left)
                    if not alternate:
                        saved_windows.append(dict(sample_row=int(i), origin=left, rf=window.tolist()))
                    if found.plateau[0] and full_min is None:
                        full_min = float(raw.min())
                    positions, sat = resolve_saturation(found, [np.inf if full_min is None else full_min])
                    if found.unclosed[0]:
                        exact=complete_negative_trough(raw,left,right)
                        positions[0]=exact['position'];sat[0]=exact['saturated']
                    if positions[0] >= 0:
                        (observed_cal if alternate else observed)[j] = positions[0]
                    if not alternate:
                        saturated[j] = sat[0]
            paired = np.isfinite(observed) & np.isfinite(calibrated)
            dnom = (observed[paired]-nominal)/fs*1e6
            dcal = (observed[paired]-calibrated[paired])/fs*1e6
            def stats(x):
                if not len(x):
                    return dict(median_us=None, iqr_us=None, within_035_ratio=None)
                q=np.quantile(x,[.25,.5,.75])
                return dict(median_us=float(q[1]),iqr_us=float(q[2]-q[0]),within_035_ratio=float(np.mean(np.abs(x)<=.35)))
            ns=stats(dnom);cs=stats(dcal)
            passed = bool(len(dcal) and abs(cs['median_us']) < abs(ns['median_us']))
            row = dict(board=b['board'], k=k, layer_mm=b['layer_mm'], selected=50,
                       valid_paired=int(paired.sum()), failed=int((~paired).sum()),
                       candidate_count=len(candidates), saturated_count=int(saturated.sum()),
                       boundary_count=int(boundary.sum()), nominal_tau_samples=float(nominal),
                       calibrated_tau_median_samples=float(np.median(calibrated)),
                       nominal=ns, calibrated=cs, adoption_required=k>=2,
                       adoption_pass=passed, surface_tail_overlap_known=b['layer_mm']==3 and k==1,
                       auxiliary_calibrated_search=stats((observed_cal-calibrated)[np.isfinite(observed_cal)]/fs*1e6),
                       auxiliary_search_peak_changed=int(np.sum(np.isfinite(observed_cal)&np.isfinite(observed)&(observed_cal!=observed))))
            rows.append(row)
            prefix=b['board']+'_k'+str(k)
            for name,value in dict(sample_rows=ids,source_rows=np.asarray(source[ids]),t0=t0[ids],
                                   nominal_tau=np.full(50,nominal),calibrated_tau=calibrated,
                                   observed=observed,observed_calibrated_search=observed_cal,paired=paired).items():
                arrays[prefix+'_'+name]=value
            save(output/'validation_windows'/f'{prefix}.json', saved_windows)
    np.savez_compressed(output/'validation_samples.npz', **arrays)
    save(output/'D_label_access_audit.json', dict(stage='D only',seed=23,training_entities=train,
         label_files=audit,held_out_label_access_count=0,outer_evaluation_opened=False))
    required=[r for r in rows if r['adoption_required']]
    passed=bool(required and all(r['adoption_pass'] for r in required))
    return dict(rows=rows,required_count=len(required),passed_count=sum(r['adoption_pass'] for r in required),
                adoption_pass=passed,failed_gates=[dict(board=r['board'],k=r['k']) for r in required if not r['adoption_pass']])


def report(spec, fit, validation, path):
    def f(x, digits=4):
        return 'NA' if x is None else f'{x:.{digits}f}'
    text=['# V1.5.2 时间基准校准诊断', '',
          '## 范围与判定', '',
          f"决定：**{'ADOPT_CANDIDATE' if validation['adoption_pass'] else 'STOP_REPORT_ONLY'}**。候选方案：{fit['selected_scheme']}。",
          f"E 要求每个板、每个 k≥2 均严格改善；实际 {validation['passed_count']}/{validation['required_count']} 项通过。",
          'A–C 使用全13板全部 RF，与标签无关。D 只读 F1 train 缺陷标签；outer 性能评价关闭。',
          '输入为冻结 catalog 的现有 LP20/裁剪 RF，在 divisor 之前计算；未修改源 RF。',
          '完整预设规则见 ../PROTOCOL.md；每条计时、随机抽样身份、原始搜索窗和访问审计随报告保存。', '',
          '## B：每板 t0（采样点）', '',
          '|板|记录数|中位数|q25|q75|IQR|q01|q99|saturated 数/比例|failed 数/比例|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in fit['boards']:
        s=b['t0']; text.append(f"|{b['board']}|{b['records']}|{f(s['median'])}|{f(s['q25'])}|{f(s['q75'])}|{f(s['iqr'])}|{f(s['q01'])}|{f(s['q99'])}|{b['saturated_count']} / {b['saturated_ratio']:.4%}|{b['failed_count']} / {b['failed_ratio']:.4%}|")
    text += ['', '## C：底面回波与层间时间', '',
             '|板|K|层厚 mm|有效底波|D_bw 中位数/点|IQR/点|Δ/点|Δ/us|名义 c m/s|c_eff m/s|底波失败|边界不足|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in fit['boards']:
        s=b['backwall']; text.append(f"|{b['board']}|{b['K']}|{b['layer_mm']}|{s['count']}|{f(s['median'])}|{f(s['iqr'])}|{f(b['delta_samples'])}|{f(b['delta_us'])}|{b['nominal_c_mps']}|{f(b['c_eff_mps'],2)}|{b['backwall_failed_count']}|{b['backwall_boundary_count']}|")
    text += ['', 'C 的底波候选可能来自遮挡/弱回波区的其他负半周期；这里不引入标签过滤或逐条 Δ。', '',
             '## D：F1 train 每板每界面50条，seed23', '',
             '同一个谷点（名义搜索窗）分别计算两种参考的偏移；比例分母为有效成对记录，失败不补抽。', '',
             '|板|k|有效/50|名义偏移中位数 us|名义 IQR us|名义 ±0.35us|校准偏移中位数 us|校准 IQR us|校准 ±0.35us|E|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def drow(r):
        n,c=r['nominal'],r['calibrated']
        decision=('PASS' if r['adoption_pass'] else 'FAIL') if r['adoption_required'] else '不参与'
        return f"|{r['board']}|{r['k']}|{r['valid_paired']}/50|{f(n['median_us'])}|{f(n['iqr_us'])}|{f(n['within_035_ratio'])}|{f(c['median_us'])}|{f(c['iqr_us'])}|{f(c['within_035_ratio'])}|{decision}|"
    text += [drow(r) for r in validation['rows']]
    text += ['', '### 3 mm 板第1界面（表面波拖尾覆盖，单列）', '',
             '|板|k|有效/50|名义偏移中位数 us|名义 IQR us|名义 ±0.35us|校准偏移中位数 us|校准 IQR us|校准 ±0.35us|E|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    text += [drow(r) for r in validation['rows'] if r['surface_tail_overlap_known']]
    text += ['', '## E：事先固定的决定', '',
             f"13板 t0 IQR 全部≤10点：{fit['selected_scheme']=='per_board'}；统一参考 t0_ref={fit['t0_ref_samples']} 点（仅逐条平移使用）。",
             '失败项：'+json.dumps(validation['failed_gates'],ensure_ascii=False),
             '采用条件不通过时：不接入 Local/Gaussian/能量比/截断，不改统计量，不创建 v1.5.2 commit/tag，不 push，不启动四格。' if not validation['adoption_pass'] else '仅通过 A–E；F 的实现、回归、审查、发布与运行另据真实证据登记。', '',
             '## 验证与限制', '',
             '- A–C 的 np.load 执行期白名单断言只允许 RF、offsets、lengths、source_row，全部 r 模式；D 标签文件仅限7块训练板。',
             '- RF 前后 size/mtime 一致；索引重新计算 SHA256。未重新全量计算 RF SHA256，不将 size/mtime 当作内容哈希。',
             '- 所有实际搜索窗检查信号边界；边界不足记失败，不截短窗口冒充完整检出。',
             '- 这是时间诊断，不是缺陷识别性能、最终测试或泛化结果。',
             '- 当前 RF 为既有裁剪和滤波数组；按首负半周期规则输出候选，不能由算法本身证明其物理反射身份。', '']
    path.write_text('\n'.join(text),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text(encoding='utf-8'))
    catpath=Path(cfg['catalog']);cat=json.loads(catpath.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True,exist_ok=False)
    for directory in ('rf_only','reports','validation_windows'):
        (a.output/directory).mkdir()
    inp=cat['input_config']
    spec=dict(sampling_rate_hz=inp['sampling_rate_hz'],batch_reference_samples=inp['batch_reference_samples'],
              catalog_sha256=digest(catpath),config_sha256=digest(a.config),
              boards=[dict(board=b['entity'],records=b['records'],K=b['K'],layer_mm=b['layer_mm'],
                      nominal_c_mps=inp['sound_speed_mps']['AL' if b['group']=='AL' else 'TC4'],
                      rf_path=b['rf_path'],**{key+'_path':str(catpath.parent/b['index_dir']/(key+'.npy'))
                      for key in ('offsets','lengths','source_row')}) for b in cat['entities']])
    assert len(spec['boards'])==13
    save(a.output/'rf_only_input_spec.json',spec)
    fit=fit_boards(spec,a.output/'rf_only')
    save(a.output/'calibration_diagnostic.json',fit)
    # Persist the calibration before the sole label-reading step begins.
    fit_sha=digest(a.output/'calibration_diagnostic.json')
    validation=validate_train(spec,cfg,fit,a.output)
    assert digest(a.output/'calibration_diagnostic.json')==fit_sha
    save(a.output/'validation.json',validation)
    decision=dict(status='ADOPT_CANDIDATE' if validation['adoption_pass'] else 'STOP_REPORT_ONLY',
                  selected_scheme=fit['selected_scheme'],adoption=validation,
                  calibration_sha256=fit_sha,calibration_used_labels=False,
                  outer_evaluation_opened=False,training_started=False,
                  created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    save(a.output/'decision.json',decision)
    report(spec,fit,validation,a.output/'reports/v152_timebase_calibration.md')
    save(a.output/'artifact_hashes.json',{str(path.relative_to(a.output)):digest(path)
         for path in a.output.rglob('*') if path.is_file()})
    print(json.dumps({k:v for k,v in decision.items() if k!='adoption'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    main()
