"""CPU-only full frozen-epoch batching audit; no model or optimizer execution."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import torch

def digest(plan):
    sizes=np.array([len(b) for b in plan],dtype=np.int64)
    return hashlib.sha256(sizes.tobytes()+np.concatenate(plan).tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from a3_ultrasound.speed import SpeedOptions,dataset_from_config
    from a3_ultrasound.data.epoch_sampling import use_epoch
    from a3_ultrasound.stream_training import catalog_signature
    from a3_ultrasound.data.length_batches import matched_smoke_indices
    cfg=json.loads(a.config.read_text(encoding='utf-8'));assert cfg['freeze_training_permitted'] is False and cfg['outer_evaluation_permitted'] is False
    cfg=dict(cfg,speed=SpeedOptions.for_smoke('final_bucket').to_dict());ds=dataset_from_config(cfg,'train')
    manifest=json.loads((Path(cfg['epoch_selection_root'])/'manifest.json').read_text(encoding='utf-8'))
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for epoch in range(1,41):
        selection_sha=use_epoch(ds,cfg['epoch_selection_root'],epoch,seed=23,catalog_signature=catalog_signature(ds.catalog))
        assert selection_sha==manifest['files'][str(epoch)]['sha256'] and len(ds)==100732
        before=ds.refs.copy();lengths=np.array([ds._input_length(i) for i in range(len(ds))],np.int64)
        plan=ds.batch_plan(16);flat=np.concatenate(plan)
        assert np.array_equal(np.sort(flat),np.arange(len(ds))) and len(np.unique(flat))==len(ds)
        assert np.array_equal(ds.refs,before)
        assert all(len(b)<=16 and np.all(lengths[b]==lengths[b[0]]) for b in plan)
        for length,count in zip(*np.unique(lengths,return_counts=True)):
            batches=[b for b in plan if lengths[b[0]]==length]
            assert sum(len(b) for b in batches)==count and len(batches)==int(np.ceil(count/16))
        batch_lengths=np.array([lengths[b[0]] for b in plan]);assert not np.array_equal(batch_lengths,np.sort(batch_lengths))
        row=dict(epoch=epoch,records=len(ds),batches=len(plan),tail_sizes=[len(b) for b in plan if len(b)<16],
            frozen_selection_sha256=selection_sha,plan_sha256=digest(plan),population_preserved=True,homogeneous=True,randomized_batch_order=True)
        if epoch==1:
            signatures=[]
            for ordinal,head in enumerate(cfg['heads']):
                torch.manual_seed(ordinal+41);state=torch.get_rng_state()
                again=ds.batch_plan(16,generator=torch.Generator().manual_seed(ordinal+111))
                assert torch.equal(state,torch.get_rng_state());signatures.append(digest(again))
            assert len(set(signatures))==1 and signatures[0]==digest(plan)
            row['head_order_sha256']=dict(zip(cfg['heads'],signatures))
            chosen=matched_smoke_indices(lengths,16,200,seed=23,epoch=1)
            assert len(chosen)==3200
            np.save(a.output/'common_3200_refs.npy',ds.refs[chosen])
            np.savez_compressed(a.output/'epoch01_plan.npz',refs=ds.refs,flat_indices=flat,sizes=np.array([len(b) for b in plan]),lengths=lengths)
        rows.append(row)
    result=dict(status='PASS_ALL_40_FROZEN_EPOCH_BATCH_PLANS',rows=rows,model_forwards=0,optimizer_updates=0,outer_opened=False,
        RNG='SeedSequence([seed,epoch,0xA3153]); within-length permutation then whole-batch permutation')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],epochs=40,records_each=100732)))
if __name__=='__main__':main()
