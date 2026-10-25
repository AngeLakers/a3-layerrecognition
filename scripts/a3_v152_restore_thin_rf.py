"""Restore one existing spatial cohort's full RF tail without reading labels."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil,sys,time
import numpy as np
from scipy.signal import butter,sosfiltfilt

ALLOWED={'raw_RF','full_filtered_RF','stored_RF','source_row','xy_index','crop_start',
         'crop_end_original_exclusive','valid_length','offsets','source_sha256'}

def guarded_load(field,path,access,**kwargs):
    assert field in ALLOWED, 'RF-only restore forbidden field: '+field
    access.append(dict(field=field,path=str(path)))
    return np.load(path,allow_pickle=False,**kwargs)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def restore(root,cfg,model_prefix=900):
    start_time=time.monotonic();catalog=load(cfg['catalog'])
    entry=next(r for r in catalog['entities'] if r['entity']=='薄')
    base=Path(entry['source_index']).parent.parent
    audit=load(base/'audit.json');access=[];keys=('source_row','xy_index','crop_start','crop_end_original_exclusive','valid_length','offsets')
    with np.load(entry['source_index'],allow_pickle=False) as z:
        index={}
        for key in keys:
            assert key in ALLOWED;access.append(dict(field=key,path=entry['source_index']));index[key]=z[key]
    raw=guarded_load('raw_RF',base/'raw_cache.npy',access,mmap_mode='r')
    full=guarded_load('full_filtered_RF',base/'rf_front_lp20_float32.npy',access,mmap_mode='r')
    stored=guarded_load('stored_RF',entry['rf_path'],access,mmap_mode='r')
    rows=index['source_row'];starts=index['crop_start'];n=int(audit['points'])
    assert n==1394 and raw.shape==(audit['files'],n) and len(rows)==266072
    oldrows=guarded_load('source_row',Path(cfg['catalog']).parent/entry['index_dir']/'source_row.npy',access,mmap_mode='r')
    assert np.array_equal(rows,oldrows)
    lengths=n-starts;offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    assert full.shape==(int(offsets[-1]),)
    common=int(lengths.min());assert common==1314
    if type(model_prefix) is not int or not 1<=model_prefix<=common:
        raise ValueError('Model prefix must fit every restored source RF.')
    outdir=root/'rf';outdir.mkdir(exist_ok=False)
    out=np.lib.format.open_memmap(outdir/'rf_lp20_float32.npy',mode='w+',dtype=np.float32,shape=(int(offsets[-1]),))
    sos=butter(3,20e6,fs=200e6,output='sos');done=0
    for crop in np.unique(starts):
        members=np.flatnonzero(starts==crop);L=n-int(crop);t=np.arange(L)
        for st in range(0,len(members),256):
            ids=members[st:st+256]
            f=sosfiltfilt(sos,np.asarray(raw[rows[ids],int(crop):],np.float64),axis=-1,padtype='odd',padlen=256).astype(np.float32)
            pos=offsets[ids,None]+t
            assert np.array_equal(f,full[pos]),'New filter differs from historical complete filter'
            oldlen=index['valid_length'][ids];valid=t[None,:]<oldlen[:,None]
            oldpos=index['offsets'][ids,None]+np.minimum(t[None,:],oldlen[:,None]-1)
            assert np.array_equal(f[valid],stored[oldpos][valid]),'Old saved RF prefix changed'
            out[pos]=f;done+=len(ids)
        print(json.dumps(dict(phase='restore',crop_start=int(crop),complete=done,total=len(rows))),flush=True)
    out.flush();assert done==len(rows)
    newidx=dict(source_row=rows,xy_index=index['xy_index'],crop_start=starts,
        crop_end_original_exclusive=np.full(len(rows),n,np.int32),valid_length=lengths.astype(np.int32),offsets=offsets)
    np.savez_compressed(outdir/'index.npz',**newidx)
    hashes=guarded_load('source_sha256',base/'source_sha256.npy',access,mmap_mode='r')
    selected=np.unique(np.r_[np.random.default_rng(23).choice(len(rows),64,replace=False),starts.argmin(),starts.argmax()])
    direct=[]
    for j in selected:
        src=int(rows[j]);x,y=map(int,index['xy_index'][j]);path=Path(audit['source'])/f'x{x}y{y}.txt';blob=path.read_bytes()
        assert hashlib.sha256(blob).digest()==hashes[src].tobytes()
        a=np.fromstring(blob.decode('ascii'),sep=' ',dtype=np.int16);assert np.array_equal(a,raw[src])
        f=sosfiltfilt(sos,a[int(starts[j]):].astype(float),padtype='odd',padlen=256).astype(np.float32)
        assert np.array_equal(f,out[offsets[j]:offsets[j+1]])
        direct.append(dict(source_row=src,path=str(path),sha256=hashlib.sha256(blob).hexdigest()))
    save(root/'restore_verification.json',dict(status='PASS_RF_ONLY_RESTORE',records=done,
        original_points=n,front_start_min=int(starts.min()),front_start_max=int(starts.max()),
        restored_lengths=[int(lengths.min()),int(lengths.max())],common_available_prefix=common,
        model_fixed_prefix=model_prefix,
        all_complete_filter_outputs_bitwise_equal=True,all_old_saved_prefixes_bitwise_equal=True,
        fresh_original_text_rereads=direct,read_labels=False,access=access,
        data_file=str(outdir/'rf_lp20_float32.npy'),data_sha256=sha(outdir/'rf_lp20_float32.npy'),
        index_sha256=sha(outdir/'index.npz'),parent_index_sha256=sha(entry['source_index']),
        parent_full_filtered_sha256=sha(base/'rf_front_lp20_float32.npy'),
        raw_cache_sha256=sha(base/'raw_cache.npy'),source_hash_inventory_sha256=sha(base/'source_sha256.npy'),
        seconds=time.monotonic()-start_time))

def make_catalog(root,cfg):
    from a3_ultrasound.stream_training import catalog_signature
    oldpath=Path(cfg['catalog']);old=load(oldpath);cat=copy.deepcopy(old)
    restored=load(root/'restore_verification.json');assert restored['status']=='PASS_RF_ONLY_RESTORE'
    assert restored['read_labels'] is False and sha(root/'rf/rf_lp20_float32.npy')==restored['data_sha256']
    dest=root/'catalog';dest.mkdir(exist_ok=False);copied=[]
    for entry,oldentry in zip(cat['entities'],old['entities']):
        olddir=oldpath.parent/oldentry['index_dir']
        if entry['entity']!='薄':entry['index_dir']=str(olddir.resolve());continue
        newdir=dest/oldentry['index_dir'];newdir.mkdir()
        for path in olddir.glob('*.npy'):
            if path.stem in ('lengths','offsets'):continue
            target=newdir/path.name;shutil.copy2(path,target);assert sha(path)==sha(target)
            copied.append(dict(field=path.stem,old=str(path),new=str(target),sha256=sha(target)))
        with np.load(root/'rf/index.npz',allow_pickle=False) as z:
            np.save(newdir/'lengths.npy',z['valid_length'],allow_pickle=False)
            # Catalog offsets are record starts; source index also has final sentinel.
            np.save(newdir/'offsets.npy',z['offsets'][:-1],allow_pickle=False)
            assert np.array_equal(z['source_row'],np.load(newdir/'source_row.npy',allow_pickle=False))
            assert np.array_equal(z['xy_index'],np.load(newdir/'xy.npy',allow_pickle=False))
        entry.update(rf_path=str(root/'rf/rf_lp20_float32.npy'),source_index=str(root/'rf/index.npz'),
            index_sha256=restored['index_sha256'],min_length=restored['model_fixed_prefix'],
            rf_sha256=restored['data_sha256'])
    cat['input_config']['tail']['fixed_prefix_samples_by_entity']={'薄':restored['model_fixed_prefix']}
    cat['RF_revision']=dict(scope='Thin board only; spatial cohort and all labels preserved',
        parent_catalog=str(oldpath),parent_catalog_sha256=sha(oldpath),restore_verification=str(root/'restore_verification.json'))
    save(dest/'catalog.json',cat);sig=catalog_signature(cat);assert sig!=catalog_signature(old)
    for key in ('epoch_selection_root','validation_selection_root'):
        source=Path(cfg[key]);target=root/('epoch_indices' if key=='epoch_selection_root' else 'validation_selection')
        shutil.copytree(source,target)
        manifest=load(target/'manifest.json');assert manifest['catalog_signature']==catalog_signature(old)
        manifest['catalog_signature']=sig;manifest['parent_catalog_signature']=catalog_signature(old)
        manifest['revision_note']='RF tail changed; all selection identities, weights, orders and array bytes unchanged'
        save(target/'manifest.json',manifest)
        for path in source.iterdir():
            if path.is_file() and path.name!='manifest.json':assert sha(path)==sha(target/path.name)
        cfg[key]=str(target)
    cfg.update(catalog=str(dest/'catalog.json'),catalog_signature=sig,
        epoch_manifest_sha256=sha(Path(cfg['epoch_selection_root'])/'manifest.json'),
        physical_statistics_path=str(root/'physical_statistics/normalization.json'),
        physical_statistics_sha256='PENDING_REFIT',freeze_training_permitted=False)
    cfg['RF_revision']=dict(scope='thin_rf_tail_restore',model_input_samples=restored['model_fixed_prefix'],
        data_sha256=restored['data_sha256'],all_other_entities_unchanged=True)
    save(root/'local_config.json',cfg)
    save(root/'spatial_label_inheritance.json',dict(status='IDENTICAL_SPATIAL_AND_LABEL_FILES',records=266072,
        copied_files=copied,other_entities_unchanged=True,source_row_order_unchanged=True,xy_unchanged=True,
        relabelled=False,resampled=False,epoch_and_validation_array_bytes_unchanged=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-config',type=Path,required=True);p.add_argument('--phase',choices=['restore','catalog'],required=True)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    cfg=load(a.baseline_config)
    if a.phase=='restore':restore(a.root,cfg)
    else:make_catalog(a.root,cfg)

if __name__=='__main__':main()
