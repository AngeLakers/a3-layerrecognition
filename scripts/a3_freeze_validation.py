"""Build metadata-only fixed validation subsets; never inspect predictions."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from a3_ultrasound.data.validation_sampling import select_validation_rows
from a3_ultrasound.stream_training import catalog_signature


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();catalog=json.loads(a.catalog.read_text(encoding='utf-8'));cfg=json.loads(a.config.read_text(encoding='utf-8'))
    registry={e['entity']:e for e in catalog['entities']};out=a.output;out.mkdir(parents=True,exist_ok=False)
    manifest={'policy':'all reliable positives + spatial mask/quality strata; one uniform sample per bin; integer inverse-probability weights',
              'id':'validation_compact_v1_20261023','seed':20261023,'target_per_entity':13334,
              'catalog_signature':catalog_signature(catalog),'folds':{}}
    for fold in cfg['folds']:
        selected=[];refs=[];weights=[];offset=0;reports=[]
        for s,entity in enumerate(fold['validation']):
            e=registry[entity];d=a.catalog.parent/e['index_dir']
            arrays={key:np.load(d/f'{key}.npy',mmap_mode='r') for key in ('y','label_mask','xy','quality')}
            ids,w=select_validation_rows(arrays['y'],arrays['label_mask'],arrays['xy'],arrays['quality'],ordinal=e['ordinal'])
            selected.append(ids+offset);refs.append(np.column_stack((np.full(len(ids),s,dtype=np.int64),ids)));weights.append(w)
            known_positive=(arrays['label_mask']&(arrays['y']==1)).any(1)
            reports.append({'entity':entity,'population':len(arrays['y']),'selected':len(ids),
                            'reliable_positives_all':int(known_positive.sum()),'reliable_positives_selected':int(known_positive[ids].sum()),
                            'weight_sum':int(w.sum()),'max_weight':int(w.max())})
            offset+=len(arrays['y'])
        path=out/f"{fold['fold']}.npz"
        np.savez_compressed(path,indices=np.concatenate(selected),refs=np.concatenate(refs),weights=np.concatenate(weights),entities=np.array(fold['validation']))
        manifest['folds'][fold['fold']]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'entities':reports,
            'population':offset,'selected':sum(x['selected'] for x in reports)}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'selected':v['selected'],'population':v['population']} for k,v in manifest['folds'].items()}))

if __name__=='__main__':main()
