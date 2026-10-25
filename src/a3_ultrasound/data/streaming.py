"""Memory-mapped RF/index access; specimen membership stays outside the model."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from a3_ultrasound.data.quick_input import transform_trace
from a3_ultrasound.schema import InterfaceBatch


class RFShardDataset:
    """Load only selected entities; never materialize the full RF tensor."""

    def __init__(self, catalog_path, entities, *, training=False, seed=23,
                 normal_ratio=3, diagnostic_per_class=None, physical_config=None,
                 tail_policy=None, window_policy=None, timebase_config=None, window_jitter=None):
        self.path=Path(catalog_path)
        self.catalog=json.loads(self.path.read_text(encoding='utf-8'))
        self.config=dict(self.catalog['input_config'])
        if tail_policy is not None:
            self.config['tail']=dict(self.config['tail'],policy=tail_policy)
        if window_policy is not None:
            if window_policy == {'name': 'board_timebase', 'halfwidth_samples': 80}:
                if timebase_config is None:
                    raise ValueError('Board timebase artifact is required.')
                self.config.update(window_halfwidth_policy='board_timebase',window_halfwidth_samples=80,
                    timebase_calibration_sha256=timebase_config['_artifact_sha256'])
            elif window_policy == {'name': 'half_interval_capped', 'cap_samples': 110}:
                self.config.update(window_halfwidth_policy=window_policy['name'],
                                   window_max_halfwidth_samples=window_policy['cap_samples'])
            else:
                raise ValueError('Unknown versioned input window policy.')
        self.timebase_rows={r['board']:r for r in timebase_config['rows']} if timebase_config is not None else {}
        if window_jitter is not None and (timebase_config is None or window_jitter !=
                {'enabled':True,'max_shift_samples':24,'seed':23,'distribution':'uniform'}):
            raise ValueError('Unknown training window jitter strategy.')
        self.window_jitter=window_jitter
        self.jitter_epoch=0
        self.physical_config=physical_config
        if physical_config is not None:
            if physical_config.get('input_window_policy') != window_policy:
                raise ValueError('Physical statistics and dataset window policy disagree.')
            expected = physical_config.get('effective_input_config')
            if expected is not None and expected != self.config:
                raise ValueError('Physical statistics and effective dataset input config disagree.')
        self.shards=[]; refs=[]
        requested=list(entities)
        if not requested or len(set(requested))!=len(requested):
            raise ValueError('Nonempty unique entity list required.')
        registry={r['entity']:r for r in self.catalog['entities']}
        if not set(requested)<=set(registry):raise ValueError('Unknown entity.')
        self.entities=requested
        for entity in requested:
            entry=registry[entity]; base=self.path.parent/entry['index_dir']
            arrays={k:np.load(base/f'{k}.npy',mmap_mode='r') for k in
                    ('source_row','offsets','lengths','y','label_mask','design_y','design_mask','eligible','xy','quality')}
            ids=np.flatnonzero(arrays['eligible']) if training else np.arange(len(arrays['source_row']))
            rng=np.random.default_rng(np.random.SeedSequence([seed,entry['ordinal']]))
            if training:
                positive=ids[arrays['y'][ids].any(1)]
                negative=ids[~arrays['y'][ids].any(1)]
                xy=arrays['xy'][negative]
                negative=negative[np.lexsort((xy[:,0],xy[:,1]))]
                target=min(len(negative),int(normal_ratio*len(positive)))
                # One random position in each spatially ordered stratum.
                bins=np.linspace(0,len(negative),target+1,dtype=np.int64)
                chosen=np.array([rng.integers(a,b) for a,b in zip(bins[:-1],bins[1:])],dtype=np.int64)
                negative=negative[chosen]
                if diagnostic_per_class is not None:
                    n=int(diagnostic_per_class)
                    if n<1:raise ValueError('Diagnostic count must be positive.')
                    positive=rng.choice(positive,min(n,len(positive)),replace=False)
                    negative=rng.choice(negative,min(n,len(negative)),replace=False)
                ids=np.sort(np.r_[positive,negative])
            elif diagnostic_per_class is not None:
                raise ValueError('Evaluation uses natural proportions; no diagnostic sampling.')
            if not len(ids):raise ValueError(f'No records selected for {entity}.')
            shard=dict(entry,arrays=arrays,rf=np.load(entry['rf_path'],mmap_mode='r'))
            ordinal=len(self.shards);self.shards.append(shard)
            refs.append(np.column_stack((np.full(len(ids),ordinal,dtype=np.int64),ids)))
        self.refs=np.concatenate(refs)
        self.training=training
        self.evaluation_weights=None

    def set_epoch(self,epoch):
        if type(epoch) is not int or epoch<0:
            raise ValueError('A nonnegative epoch is required.')
        self.jitter_epoch=epoch

    def use_frozen_evaluation_selection(self, path):
        if self.training or self.evaluation_weights is not None:
            raise ValueError('Evaluation selection can only be applied once to an evaluation dataset.')
        with np.load(path,allow_pickle=False) as pack:
            indices=pack['indices'];weights=pack['weights'];refs=pack['refs'];entities=pack['entities'].tolist()
        if entities!=self.entities or indices.dtype!=np.int64 or weights.dtype!=np.int64:
            raise ValueError('Frozen validation entities or index/weight dtypes mismatch.')
        if indices.ndim!=1 or weights.shape!=indices.shape or not len(indices) or np.any(weights<1):
            raise ValueError('Invalid validation selection weights.')
        if np.any(indices<0) or np.any(indices>=len(self)) or np.any(np.diff(indices)<=0):
            raise ValueError('Validation indices must be unique, ordered and in range.')
        if not np.array_equal(self.refs[indices],refs) or int(weights.sum())!=len(self):
            raise ValueError('Frozen validation refs or population weight mismatch.')
        self.refs=refs.copy();self.evaluation_weights=weights.copy()

    def __len__(self):return len(self.refs)

    def identity(self,index):
        s,i=self.refs[int(index)];shard=self.shards[s]
        return f"{shard['entity']}:{int(shard['arrays']['source_row'][i])}",shard['entity']

    def selection_summary(self):
        return {s['entity']:int((self.refs[:,0]==j).sum()) for j,s in enumerate(self.shards)}

    def annotations(self,indices):
        output=[]
        for index in indices:
            s,i=self.refs[int(index)];a=self.shards[s]['arrays']
            output.append({'nominal_y':a['design_y'][i].tolist(),
                           'nominal_label_mask':a['design_mask'][i].tolist(),
                           'quality_flagged':not bool(a['quality'][i]),
                           'sample_weight':int(self.evaluation_weights[int(index)]) if self.evaluation_weights is not None else 1})
        return output

    def make_batch(self,indices,device='cpu'):
        traces=[];metas=[];ys=[];masks=[];label_masks=[]
        physical_loc=[];physical_ctx=[]
        for index in indices:
            s,i=self.refs[int(index)];shard=self.shards[s];a=shard['arrays'];k=shard['K']
            start=int(a['offsets'][i]);length=int(a['lengths'][i])
            shift=0.
            if self.training and self.window_jitter is not None:
                from a3_ultrasound.data.board_timebase import shared_window_shift
                shift=shared_window_shift(self.window_jitter['seed'],self.jitter_epoch,
                    int(shard['ordinal']),int(a['source_row'][i]),self.window_jitter['max_shift_samples'])
            x,meta,info=transform_trace(shard['rf'][start:start+length],shard,self.config,
                timebase=self.timebase_rows.get(shard['entity']),window_shift=shift)
            if self.physical_config is not None:
                from a3_ultrasound.data.physical_features import energy_log_ratios,physical_tokens,packed_energy_bounds
                p=self.physical_config;tau=np.asarray(info['nominal_tof_samples'])
                if shard['entity'] in self.timebase_rows:
                    left,right=packed_energy_bounds(meta,info['valid_length'],k)
                else:
                    left,right=tau-info['halfwidth_samples'],tau+info['halfwidth_samples']
                ratios=energy_log_ratios(x,left,right,front_end=p['front_end_samples'],epsilon=p['epsilon'])
                loc,ctx=physical_tokens(ratios,mean=p['mean'],std=p['std'])
                physical_loc.append(loc);physical_ctx.append(ctx)
            y=np.zeros(6,dtype=np.int64);y[:k]=a['y'][i]
            mask=np.arange(6)<k;lm=np.zeros(6,dtype=bool);lm[:k]=a['label_mask'][i]
            if self.training and not lm[:k].all():raise ValueError('Training label became incomplete.')
            traces.append(x);metas.append(meta);ys.append(y);masks.append(mask);label_masks.append(lm)
        lengths=np.array([len(x) for x in traces]);padded=np.zeros((len(traces),int(lengths.max())),np.float32)
        for j,x in enumerate(traces):padded[j,:len(x)]=x
        values={'x_raw':padded,'length_mask':np.arange(padded.shape[1])[None]<lengths[:,None],
                'interface_meta':np.stack(metas),'interface_mask':np.stack(masks),
                'label_mask':np.stack(label_masks),'y':np.stack(ys)}
        if self.physical_config is not None:
            values.update(phys_loc=np.stack(physical_loc),phys_ctx=np.stack(physical_ctx))
        return InterfaceBatch(**{k:torch.from_numpy(v).to(device) for k,v in values.items()})

    def batches(self,size,device='cpu',*,generator=None):
        if size<1:raise ValueError('Positive batch size required.')
        order=torch.randperm(len(self),generator=generator).numpy() if generator is not None else np.arange(len(self))
        for start in range(0,len(order),size):
            ids=order[start:start+size]
            yield self.make_batch(ids,device),ids
