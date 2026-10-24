"""Raw inferred-MAP correction values with explicit upstream truth masks."""
from pathlib import Path
import json

import numpy as np

from a3_ultrasound.runtime_supervisor import atomic_write


class CorrectionArchive:
    """One row per valid non-first interface, including masked current truths."""
    def __init__(self):
        self.rows = []

    def add(self, row):
        if row.get('gate') is None:
            return
        k_count = len(row['map_states'])
        y = row.get('y')
        mask = row.get('label_mask') or [False] * k_count
        for k in range(1, k_count):
            upstream = [int(y[j]) if y is not None and mask[j] else -1 for j in range(k)]
            known_bad = any(v == 1 for v in upstream)
            group = 1 if known_bad else 0 if all(v == 0 for v in upstream) else -1
            self.rows.append(dict(sample_id=row['sample_id'], specimen_id=row['specimen_id'],
                interface_index=k, K=k_count, sample_weight=row.get('sample_weight', 1),
                g=row['gate'][k], c=row['correction'][k], gc=row['gate_times_correction'][k],
                current_truth=int(y[k]) if y is not None and mask[k] else -1,
                map_state=int(row['map_states'][k]), upstream_truth=upstream,
                upstream_any_abnormal=group))
            if row.get('upstream_flag') is not None:
                self.rows[-1].update(upstream_flag=int(row['upstream_flag'][k]),
                    raw_gc=row['gate_times_raw_correction'][k],
                    b=row['base_logodds'][k], eta=row['base_logodds'][k]+row['gate_times_correction'][k],
                    upstream_map=row['map_states'][:k])

    def save(self, path, *, epoch, checkpoint, commit_hash, config_sha256):
        """Replace only the NPZ paired with the named checkpoint's same epoch update."""
        rows = self.rows
        width = max((r['K'] for r in rows), default=1)
        upstream = np.full((len(rows), width), -1, dtype=np.int8)
        upstream_mask = np.zeros((len(rows), width), dtype=bool)
        for i, row in enumerate(rows):
            k = row['interface_index']
            upstream[i, :k] = row['upstream_truth']
            upstream_mask[i, :k] = True
        arrays = {}
        for key in ('sample_id', 'specimen_id'):
            arrays[key] = np.asarray([r[key] for r in rows], dtype=str)
        for key in ('interface_index', 'K', 'sample_weight', 'current_truth', 'map_state', 'upstream_any_abnormal'):
            arrays[key] = np.asarray([r[key] for r in rows], dtype=np.int64)
        for key in ('g', 'c', 'gc'):
            arrays[key] = np.asarray([r[key] for r in rows], dtype=np.float64)
            if not np.isfinite(arrays[key]).all():
                raise ValueError('Nonfinite raw correction values.')
        if rows and 'upstream_flag' in rows[0]:
            if not all('upstream_flag' in r for r in rows):
                raise ValueError('Mixed correction contracts in one archive.')
            arrays['effective_gc'] = arrays['gc'].copy()
            arrays['gc'] = np.asarray([r['raw_gc'] for r in rows], dtype=np.float64)
            arrays['upstream_flag'] = np.asarray([r['upstream_flag'] for r in rows], dtype=np.int8)
            for key in ('b', 'eta'):
                arrays[key] = np.asarray([r[key] for r in rows], dtype=np.float64)
            maps = np.full((len(rows), width), -1, dtype=np.int8)
            for i,r in enumerate(rows):maps[i,:r['interface_index']]=r['upstream_map']
            arrays['upstream_map'] = maps
            arrays['correction_contract'] = np.asarray('gc=g*c; effective_gc=g*c*upstream_flag; eta=b+effective_gc')
        arrays.update(upstream_truth=upstream, upstream_interface_mask=upstream_mask,
            upstream_label_mask=upstream_mask & (upstream >= 0),
            metadata_json=np.asarray(json.dumps(dict(epoch=epoch, checkpoint=checkpoint,
                commit_hash=commit_hash, config_sha256=config_sha256,
                path='inferred_joint_MAP', interface_index='zero based; first excluded',
                unknown_truth=-1, upstream_group='1:any known abnormal; 0:all upstream known normal; -1:undetermined'),
                ensure_ascii=False)))
        def validate(p):
            with np.load(p, allow_pickle=False) as pack:
                assert len(pack['g']) == len(rows)
                assert json.loads(str(pack['metadata_json']))['epoch'] == epoch
        atomic_write(Path(path), lambda handle: np.savez_compressed(handle, **arrays), validate)
