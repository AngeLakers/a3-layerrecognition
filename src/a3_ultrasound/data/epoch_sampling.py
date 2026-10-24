"""Head-independent, pre-frozen epoch samples and order for V1.4."""
from pathlib import Path
import hashlib
import json
import numpy as np


def epoch_refs(shards, seed, epoch, normal_ratio=3, positive_fraction=1.0):
    if type(epoch) is not int or epoch < 1 or not 0 < positive_fraction <= 1 or normal_ratio != 3:
        raise ValueError('Expected epoch>=1, fraction in (0,1], and frozen normal ratio 3.')
    refs=[]
    for ordinal,shard in enumerate(shards):
        a=shard['arrays'];ids=np.flatnonzero(a['eligible'])
        positive=ids[a['y'][ids].any(1)];negative=ids[~a['y'][ids].any(1)]
        if positive_fraction<1:
            # Fixed positive subset across epochs, shared by all Heads.
            prng=np.random.default_rng(np.random.SeedSequence([seed,shard['ordinal'],0]))
            positive=np.sort(prng.choice(positive,max(1,int(len(positive)*positive_fraction)),replace=False))
        rng=np.random.default_rng(np.random.SeedSequence([seed,shard['ordinal'],epoch]))
        xy=a['xy'][negative];negative=negative[np.lexsort((xy[:,0],xy[:,1]))]
        target=int(normal_ratio*len(positive))
        if not len(positive) or len(negative)<target:
            raise ValueError('Every training board needs positives and enough normals for 3:1.')
        bins=np.linspace(0,len(negative),target+1,dtype=np.int64)
        selected=np.array([rng.integers(lo,hi) for lo,hi in zip(bins[:-1],bins[1:])])
        ids=np.sort(np.r_[positive,negative[selected]])
        refs.append(np.column_stack((np.full(len(ids),ordinal,dtype=np.int64),ids)))
    refs=np.concatenate(refs)
    order=np.random.default_rng(np.random.SeedSequence([seed,epoch,0xA314])).permutation(len(refs))
    return refs[order]


def freeze_epochs(dataset, root, *, seed, epochs, catalog_signature, positive_fraction=1.0):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    manifest=dict(version='a3-epoch-selection-v1.4',seed=seed,epochs=epochs,
        catalog_signature=catalog_signature,entities=dataset.entities,positive_fraction=positive_fraction,
        normal_ratio=3,order='stored refs are the complete batch order',files={})
    for epoch in range(1,epochs+1):
        path=root/f'epoch_{epoch:02d}.npy'
        refs=epoch_refs(dataset.shards,seed,epoch,positive_fraction=positive_fraction)
        np.save(path,refs,allow_pickle=False)
        manifest['files'][str(epoch)]=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=len(refs))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return manifest


def use_epoch(dataset, root, epoch, *, seed, catalog_signature):
    if not dataset.training:raise ValueError('Only training datasets accept epoch selection.')
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['seed']!=seed or manifest['entities']!=dataset.entities or manifest['catalog_signature']!=catalog_signature:
        raise ValueError('Epoch selection seed/entities/catalog mismatch.')
    item=manifest['files'][str(epoch)];path=root/item['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
        raise ValueError('Frozen epoch index hash mismatch.')
    refs=np.load(path,allow_pickle=False)
    expected=epoch_refs(dataset.shards,seed,epoch,positive_fraction=manifest['positive_fraction'])
    if not np.array_equal(refs,expected):raise ValueError('Frozen epoch indices differ from declared sampler.')
    dataset.refs=refs
    return item['sha256']
