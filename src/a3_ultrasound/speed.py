"""Opt-in V1.5.3 execution switches; no switch enables a real training run."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict,dataclass,fields,replace
from pathlib import Path
import hashlib,json,math
import torch
from torch import nn

@dataclass(frozen=True)
class SpeedOptions:
    workers:int=0
    prefetch_factor:int=2
    pin_memory:bool=False
    non_blocking:bool=False
    coalesced_finite:bool=False
    cpu_structure:bool=False
    validation_cache:bool=False
    cpu_correction:bool=False
    skip_minibatch_summary:bool=False
    fast_nominal_fixed:bool=False
    fused_adam:bool=False
    tf32:bool=False
    bf16_autocast:bool=False
    cudnn_benchmark:bool=False
    batch_size:int|None=None
    length_buckets:bool=False

    def __post_init__(self):
        if type(self.workers) is not int or not 0<=self.workers<=8:raise ValueError('workers must be0..8')
        if type(self.prefetch_factor) is not int or self.prefetch_factor<1:raise ValueError('positive prefetch factor required')
        if self.batch_size not in (None,16,32,64):raise ValueError('speed batch_size must be16/32/64 orNone')
        for f in fields(self):
            if f.name not in ('workers','prefetch_factor','batch_size') and type(getattr(self,f.name)) is not bool:raise ValueError('speed flags must be booleans')

    def to_dict(self):return asdict(self)

    @classmethod
    def for_smoke(cls,mode):
        if mode=='off':return cls()
        base=cls(workers=2,pin_memory=True,non_blocking=True,coalesced_finite=True,cpu_structure=True,
            validation_cache=True,cpu_correction=True,skip_minibatch_summary=True,fast_nominal_fixed=True,fused_adam=True)
        modes={'A':base,'A_no_fused':replace(base,fused_adam=False),'A_tf32':replace(base,tf32=True),
            'A_tf32_benchmark':replace(base,tf32=True,cudnn_benchmark=True),
            'A_bf16':replace(base,bf16_autocast=True),'A_bf16_benchmark':replace(base,bf16_autocast=True,cudnn_benchmark=True),
            'A_bf16_benchmark_b32':replace(base,bf16_autocast=True,cudnn_benchmark=True,batch_size=32),
            'A_bf16_benchmark_b64':replace(base,bf16_autocast=True,cudnn_benchmark=True,batch_size=64),
            'bucket':replace(base,bf16_autocast=True,cudnn_benchmark=True,batch_size=64,length_buckets=True)}
        final=replace(base,fused_adam=False,tf32=True,batch_size=16)
        modes.update(final_base=final,final_bucket=replace(final,length_buckets=True),
            final_benchmark=replace(final,cudnn_benchmark=True))
        if mode not in modes:raise ValueError('Unknown bounded smoke mode')
        return modes[mode]

_ACTIVE=ContextVar('a3_speed',default=SpeedOptions())
_CPU=ContextVar('a3_speed_cpu_batch',default=None)
_TRAIN=ContextVar('a3_speed_training',default=False)
_FINITE=ContextVar('a3_speed_finite',default=None)
def options():return _ACTIVE.get()
def cpu_batch():return _CPU.get()
def fast_structure():return options().cpu_structure and cpu_batch() is not None
def complete_training():return fast_structure() and _TRAIN.get()

def validate_cpu_batch(batch,*,training=False):
    if batch.x_raw.device.type!='cpu':raise ValueError('Structural validation requires the CPU input batch')
    batch.__post_init__()
    n=batch.length_mask.sum(1);k=batch.interface_mask.sum(1)
    if not len(n) or torch.any(n<1) or torch.any(k<1):raise ValueError('Nonempty valid prefixes required')
    if not torch.equal(batch.length_mask,torch.arange(batch.x_raw.shape[-1])[None]<n[:,None]):raise ValueError('Noncontiguous length mask')
    if not torch.equal(batch.interface_mask,torch.arange(batch.interface_mask.shape[1])[None]<k[:,None]):raise ValueError('Noncontiguous interface mask')
    if batch.label_mask is not None and torch.any(batch.label_mask & ~batch.interface_mask):raise ValueError('Labels cannot supervise padding')
    tau=batch.interface_meta[...,1]*n[:,None];left=tau-batch.interface_meta[...,3]*n[:,None];right=tau+batch.interface_meta[...,4]*n[:,None]
    if not torch.isfinite(batch.interface_meta[batch.interface_mask]).all():raise ValueError('Nonfinite interface metadata')
    if torch.any(batch.interface_mask & ((left<-.0001)|(right>n[:,None]-1+.0001)|(left>right)|(left.ceil()>right.floor()))):raise ValueError('Invalid or empty Local window')
    if batch.y is not None:
        mask=batch.interface_mask if batch.label_mask is None else batch.label_mask
        if torch.any(((batch.y!=0)&(batch.y!=1))&mask):raise ValueError('Reliable labels must be binary')
    if training and (batch.y is None or batch.label_mask is not None and not torch.equal(batch.label_mask,batch.interface_mask)):raise ValueError('Complete training labels required')

def transfer_batch(batch,device,opts,*,training=False):
    from a3_ultrasound.schema import InterfaceBatch
    if not (opts.pin_memory or opts.non_blocking or opts.cpu_structure):
        return InterfaceBatch(**{f.name:None if (v:=getattr(batch,f.name)) is None else v.to(device) for f in fields(batch)})
    validate_cpu_batch(batch,training=training)
    host=batch.pin_memory() if opts.pin_memory and str(device).startswith('cuda') and not batch.x_raw.is_pinned() else batch
    result=object.__new__(InterfaceBatch)
    for f in fields(batch):
        value=getattr(host,f.name);object.__setattr__(result,f.name,None if value is None else value.to(device,non_blocking=opts.non_blocking))
    object.__setattr__(result,'_cpu_source',host)
    object.__setattr__(result,'_cpu_validated_training',bool(training))
    return result

@contextmanager
def speed_context(opts,batch=None,*,training=False):
    opts=opts or SpeedOptions();cpu=getattr(batch,'_cpu_source',None) if batch is not None else None
    if opts.cpu_structure and cpu is None:
        if batch is not None and batch.x_raw.device.type=='cpu':validate_cpu_batch(batch,training=training);cpu=batch
        else:raise ValueError('CPU structural validation receipt missing')
    if opts.cpu_structure and training and cpu is not None and not getattr(batch,'_cpu_validated_training',False):
        validate_cpu_batch(cpu,training=True)
    tokens=[(_ACTIVE,_ACTIVE.set(opts)),(_CPU,_CPU.set(cpu)),(_TRAIN,_TRAIN.set(training)),(_FINITE,_FINITE.set([]))]
    try:yield
    finally:
        for var,token in reversed(tokens):var.reset(token)

def finite_tensor(value,mask=None):
    good=torch.isfinite(value.masked_fill(~mask,0) if mask is not None else value).all()
    pending=_FINITE.get()
    if options().coalesced_finite and pending is not None:pending.append(good)
    elif not good:raise ValueError('Nonfinite model value')

def finish_step(model,loss):
    checks=list(_FINITE.get() or [])+[torch.isfinite(loss).all()]
    checks.extend(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if not torch.stack(checks).all().item():raise RuntimeError('Nonfinite loss, model output or gradient')

def configure_backend(opts):
    if opts.bf16_autocast and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('BF16 CUDA support is required; no emulated or silent fallback')
    torch.backends.cuda.matmul.allow_tf32=opts.tf32;torch.backends.cudnn.allow_tf32=opts.tf32
    torch.backends.cudnn.benchmark=opts.cudnn_benchmark;torch.backends.cudnn.deterministic=not opts.cudnn_benchmark
    torch.use_deterministic_algorithms(not opts.cudnn_benchmark)

def precision_context(opts):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=bool(opts and opts.bf16_autocast))

class FP32GroupNorm(nn.GroupNorm):
    def forward(self,x):
        with torch.autocast(x.device.type,enabled=False):
            return torch.nn.functional.group_norm(x.float(),self.num_groups,self.weight,self.bias,self.eps)

def configure_model(model,opts):
    if opts.bf16_autocast:
        for parent in model.modules():
            for name,child in list(parent.named_children()):
                if type(child) is nn.GroupNorm:
                    norm=FP32GroupNorm(child.num_groups,child.num_channels,child.eps,child.affine)
                    norm.weight=child.weight;norm.bias=child.bias;setattr(parent,name,norm)

def dataset_from_config(cfg,split):
    """One RF/physical/timebase factory for train, validation, test and diagnostics."""
    from a3_ultrasound.data.streaming import RFShardDataset
    from a3_ultrasound.v15_protocol import PROTOCOL as V15_PROTOCOL,load_physical_artifacts,load_timebase_artifact
    if split not in ('train','validation','test','diagnostic'):raise ValueError('Unknown dataset split')
    if split=='test' and not cfg.get('outer_evaluation_permitted',False):raise ValueError('Outer evaluation remains closed')
    fold=next(f for f in cfg['folds'] if f['fold']==cfg.get('fold_name',cfg['folds'][0]['fold']))
    physical,_=load_physical_artifacts(cfg) if cfg.get('protocol')==V15_PROTOCOL else (None,None)
    table=load_timebase_artifact(cfg) if cfg.get('protocol')==V15_PROTOCOL else None
    m3=None
    from a3_ultrasound.v16_protocol import is_v16
    if is_v16(cfg):
        # V1.6: frozen 2026-11-05 centres, M6 stays off, optional fold-fixed M3 channel.
        from a3_ultrasound.v16_protocol import load_v16_timebase,verify_m6_inactive,load_m3_channel
        verify_m6_inactive(cfg);table=load_v16_timebase(cfg);m3=load_m3_channel(cfg,fold['fold'])
    training=split in ('train','diagnostic');seed=cfg.get('seed',cfg['seeds'][0])
    data=RFShardDataset(cfg['catalog'],fold['train' if training else split],training=training,seed=seed,
        normal_ratio=cfg['normal_to_positive'],diagnostic_per_class=cfg.get('diagnostic',{}).get('per_entity_per_class') if split=='diagnostic' else None,
        physical_config=physical,tail_policy=cfg.get('input_tail_policy'),window_policy=cfg.get('input_window_policy'),
        timebase_config=table,window_jitter=cfg.get('window_jitter') if training else None,m3_config=m3)
    data.speed_options=SpeedOptions(**cfg.get('speed',{}))
    if split=='validation' and cfg.get('validation_selection_root'):
        from a3_ultrasound.stream_training import catalog_signature
        root=Path(cfg['validation_selection_root']);p=root/(fold['fold']+'.npz');manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        if manifest['catalog_signature']!=catalog_signature(data.catalog) or hashlib.sha256(p.read_bytes()).hexdigest()!=manifest['folds'][fold['fold']]['sha256']:raise ValueError('Frozen validation signature mismatch')
        population=len(data);data.use_frozen_evaluation_selection(p)
        data.validation_sampling=dict(policy=manifest['policy'],selection_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            evaluated_records=len(data),population_records=population,seed=manifest['seed'])
    return data
