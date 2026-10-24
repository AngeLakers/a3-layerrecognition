"""F1 TRAIN metadata-only fixed-stop audit; does not open outer RF or labels."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.stats import rankdata


def main():
    p=argparse.ArgumentParser();p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--frozen-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.frozen_root/'deployment/code/src'))
    from a3_ultrasound.data.quick_input import physical_manifest
    from a3_ultrasound.data.time_prior import derive_time_prior_samples
    catalog=json.loads(a.catalog.read_text(encoding='utf-8'))
    config=json.loads((a.frozen_root/'analysis_inputs/proposed_ep12/resolved_config.json').read_text())
    rows=[]
    for e in catalog['entities']:
        if e['entity'] not in config['fold']['train']:continue
        folder=a.catalog.parent/e['index_dir']
        eligible=np.load(folder/'eligible.npy',mmap_mode='r')
        lengths=np.load(folder/'lengths.npy',mmap_mode='r')[eligible]
        y=np.load(folder/'y.npy',mmap_mode='r')[eligible].any(1)
        prior=derive_time_prior_samples(physical_manifest(e,catalog['input_config']),trace_length=int(lengths.max()))
        nominal=math.ceil(prior.backwall_tof_samples+catalog['input_config']['tail']['nominal_backwall_guard_us']*1e-6*catalog['input_config']['sampling_rate_hz'])
        stop=np.minimum(lengths,nominal);positive=int(y.sum());negative=len(y)-positive
        auc=None if not positive or not negative else float((rankdata(stop)[y].sum()-positive*(positive+1)/2)/(positive*negative))
        values,counts=np.unique(stop,return_counts=True)
        rows.append(dict(board=e['entity'],records=len(y),positive=positive,normal=negative,
            nominal_stop=nominal,shorter_than_nominal=int((lengths<nominal).sum()),
            stop_values=values.tolist(),stop_counts=counts.tolist(),stop_abnormal_auc=auc))
    a.output.write_text(json.dumps(dict(scope='F1 complete reliable TRAIN cohort metadata only; no RF read or outer labels evaluated',rows=rows),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',boards=len(rows))))


if __name__=='__main__':main()
