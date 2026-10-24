"""Create small mmap indexes referencing preserved RF; no RF rewrite."""
from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a3_ultrasound.data.quick_input import physical_manifest
from a3_ultrasound.data.time_prior import derive_time_prior_samples


def run(config_path,eligibility,output):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('status','').startswith('RETIRED'):raise ValueError('Retired configuration.')
    rows=json.loads(Path(config['specimen_inventory']).read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=False);entries=[]
    for ordinal,row in enumerate(rows):
        dest=output/f'entity_{ordinal:02d}';dest.mkdir()
        with np.load(row['pool'],allow_pickle=False) as z, np.load(eligibility/row['entity']/'training_eligibility.npz',allow_pickle=False) as e:
            source=z['source_row'];n=len(source)
            assert np.array_equal(source,e['source_row'])
            exclude=e['review_region_excluded']; clean=e['eligible']
            y=z['reliable_y'];label_mask=z['reliable_label_mask'] & ~z['uncertain_interface_mask'] & ~exclude[:,None]
            design_y=z['y_design'];design_mask=z['design_label_mask'] & ~exclude[:,None]
            assert np.all(label_mask[clean]) and np.array_equal(y[clean],design_y[clean])
            lengths=z['valid_length'] if 'valid_length' in z else z['lengths']
            # Geometry check includes every retained row length, independent of amplitude.
            prior=derive_time_prior_samples(physical_manifest(row,config),trace_length=int(lengths.min()))
            coords=next((k for k in ('global_design_xy_mm','xy_front_mm','xy_index') if k in z.files),None)
            arrays={'source_row':source,'offsets':z['offsets'][:n],'lengths':lengths,
                    'y':y,'label_mask':label_mask,'design_y':design_y,'design_mask':design_mask,
                    'eligible':clean,'xy':z[coords], 'quality':z['signal_quality_candidate']}
            for name,value in arrays.items():np.save(dest/f'{name}.npy',value)
            entries.append({'entity':row['entity'],'ordinal':ordinal,'group':row['group'],'material':row['material'],
                'K':row['K'],'layer_mm':row['layer_mm'],'index_dir':dest.name,
                'rf_path':str(Path(row['index']).parent/'rf_lp20_float32.npy'),
                'source_pool':row['pool'],'pool_sha256':row['pool_sha256'],'source_index':row['index'],
                'index_sha256':row['index_sha256'],'records':n,'train_eligible':int(clean.sum()),
                'positive':int(y[clean].any(1).sum()),'min_length':int(lengths.min()),
                'coordinate_field_for_sampling':coords,'prior_centers_samples':list(prior.nominal_tof_samples)})
        print(row['entity'],n,flush=True)
    catalog={'schema_version':'a3-rf-shards-v1','input_config':config,'entities':entries,
        'training_label_basis':'complete reliable, design-consistent, exclude uncertainty and nine regions',
        'evaluation_label_basis':'reliable per-interface mask excluding uncertainty and nine regions; nominal secondary separate',
        'source_eligibility':str(eligibility),'rf_copied':False}
    (output/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--eligibility',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.config,a.eligibility,a.output)
