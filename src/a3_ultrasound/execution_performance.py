"""Ordered CPU prefetch and bounded, adaptive evaluation batching only."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
import json
import time

import numpy as np
import torch


def fast_eval_assembly(original,x_raw,length_mask,encoder,*,output_dim):
    """Keep every encoder call identical; vectorize only eval-time exact copies."""
    if encoder.training or torch.is_grad_enabled():
        return original(x_raw,length_mask,encoder,output_dim=output_dim)
    from a3_ultrasound.models.backbone import _validate_temporal_inputs,TemporalBackboneOutput
    lengths=_validate_temporal_inputs(x_raw,length_mask)
    maximum=int(lengths.max());groups=[];indices=[]
    for value in torch.unique(lengths,sorted=True):
        length=int(value.item());selected=torch.nonzero(lengths==value,as_tuple=False).squeeze(1)
        group=x_raw.index_select(0,selected)[...,:length]
        features=encoder(group[:,None,:] if group.ndim==2 else group)
        if features.ndim!=3 or features.shape[1]!=output_dim or features.shape[2]!=length:
            raise RuntimeError('Time-preserving encoder contract changed.')
        groups.append(torch.nn.functional.pad(features.transpose(1,2),(0,0,0,maximum-length)))
        indices.append(selected)
    features=torch.cat(groups,dim=0).index_select(0,torch.argsort(torch.cat(indices)))
    positions=torch.arange(maximum,device=x_raw.device)
    mask=positions[None,:]<lengths[:,None]
    centers=positions.to(x_raw.dtype).expand(len(x_raw),-1).masked_fill(~mask,0)
    return TemporalBackboneOutput(features=features,feature_centers=centers,feature_mask=mask,effective_stride=1.)


def install_fast_assembly():
    import a3_ultrasound.models.backbone as backbone
    original=backbone._encode_by_valid_length
    def encoded(x_raw,length_mask,encoder,*,output_dim):
        return fast_eval_assembly(original,x_raw,length_mask,encoder,output_dim=output_dim)
    backbone._encode_by_valid_length=encoded
    return original


class EvaluationBudget:
    def __init__(self,maximum=512,reserve_fraction=.1):
        self.maximum=maximum;self.current=min(128,maximum);self.reserve_fraction=reserve_fraction
        self.bytes_per_row=0.;self.oom_retries=0;self.events=[]
    def limit(self):
        if not torch.cuda.is_available():return self.current
        free,total=torch.cuda.mem_get_info();reserve=max(1024**3,int(total*self.reserve_fraction))
        reusable=max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
        if self.bytes_per_row:
            safe=int(max(1,free+reusable-reserve)*.75/self.bytes_per_row)
            self.current=min(self.maximum,max(1,safe),max(self.current,128)*2)
        if free<reserve:torch.cuda.empty_cache();self.current=max(1,self.current//2)
        return self.current
    def observed(self,rows,baseline):
        if torch.cuda.is_available():
            peak=max(1,torch.cuda.max_memory_allocated()-baseline)
            self.bytes_per_row=max(self.bytes_per_row,peak/max(rows,1))
            free,total=torch.cuda.mem_get_info()
            self.events.append(dict(rows=rows,peak_extra_bytes=peak,free_bytes=free,total_bytes=total))
            if free<max(1024**3,int(total*self.reserve_fraction)):torch.cuda.empty_cache()


def effective_lengths(data):
    root=data.path.parent.parent/'effective_lengths';values=np.zeros(len(data),np.int64)
    for s,shard in enumerate(data.shards):
        path=root/f"entity_{shard['ordinal']:02d}_lengths.npy"
        if not path.exists():return None
        cached=np.load(path,mmap_mode='r');positions=np.flatnonzero(data.refs[:,0]==s)
        values[positions]=cached[data.refs[positions,1]]
    return values


def plans(data,size,generator,lengths,budget):
    if generator is not None:
        order=torch.randperm(len(data),generator=generator).numpy()
        for start in range(0,len(order),size):yield order[start:start+size]
    elif lengths is None:
        for start in range(0,len(data),size):yield np.arange(start,min(len(data),start+size))
    else:
        order=np.argsort(lengths,kind='stable');ordered_lengths=lengths[order]
        edges=np.r_[0,np.flatnonzero(np.diff(ordered_lengths))+1,len(order)]
        for lo,hi in zip(edges[:-1],edges[1:]):
            start=int(lo)
            while start<hi:
                stop=min(int(hi),start+budget.limit());yield order[start:stop];start=stop


class HostPrefetchBudget:
    """Bound in-flight CPU work; back off under host CPU or memory pressure."""
    def __init__(self,maximum,*,enabled=False,busy_threshold=94.,reserve_gib=4):
        self.maximum=maximum;self.current=maximum;self.enabled=enabled
        self.busy_threshold=busy_threshold;self.reserve_bytes=reserve_gib*1024**3
        self.checked=0.;self.events=[]
    def limit(self):
        if not self.enabled or time.monotonic()-self.checked<2:return self.current
        import psutil
        self.checked=time.monotonic();cpu=psutil.cpu_percent();available=psutil.virtual_memory().available
        if cpu>=self.busy_threshold or available<self.reserve_bytes:
            self.current=max(1,self.current//2)
        elif cpu<self.busy_threshold-4 and available>self.reserve_bytes*1.5:
            self.current=min(self.maximum,self.current+1)
        self.events.append(dict(cpu_percent=cpu,available_bytes=available,depth=self.current))
        self.events=self.events[-20:]
        return self.current


def prefetch_batches(data,size,device,generator,budget,*,workers=2,depth=3,bucket_evaluation=True,
                     host_budget=None):
    lengths=effective_lengths(data) if generator is None and not data.training and bucket_evaluation else None
    stream=iter(plans(data,size,generator,lengths,budget))
    def prepare(ids):
        batch=data.make_batch(ids,'cpu')
        if lengths is not None and not np.array_equal(batch.length_mask.sum(1).numpy(),lengths[ids]):
            raise ValueError('Effective-length cache differs from actual input transform.')
        return batch,ids
    pool=ThreadPoolExecutor(max_workers=workers);pending=deque()
    host_budget=host_budget or HostPrefetchBudget(depth)
    def refill():
        while len(pending)<host_budget.limit():
            try:pending.append(pool.submit(prepare,next(stream)))
            except StopIteration:break
    try:
        refill()
        while pending:
            batch,ids=pending.popleft().result()
            if device!='cpu':
                from a3_ultrasound.schema import InterfaceBatch
                values={f.name:(None if (value:=getattr(batch,f.name)) is None else value.pin_memory().to(device,non_blocking=True)) for f in fields(batch)}
                batch=InterfaceBatch(**values)
            yield batch,ids
            refill()
    finally:
        pool.shutdown(wait=True,cancel_futures=True)


def adaptive_evaluate(original,budget,model,data,*,batch_size,device):
    count=min(batch_size,budget.maximum,budget.limit() if str(device).startswith('cuda') else budget.maximum)
    while True:
        baseline=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        try:
            result=original(model,data,batch_size=count,device=device)
            budget.observed(count,baseline);return result
        except torch.OutOfMemoryError:
            if count<=1:raise
            count=max(1,count//2);budget.current=count;budget.maximum=min(budget.maximum,count)
            budget.oom_retries+=1
            if torch.cuda.is_available():torch.cuda.empty_cache()


def install(policy,output):
    import a3_ultrasound.data.streaming as streaming
    import a3_ultrasound.training as training
    import a3_ultrasound.stream_training as stream_training
    from a3_ultrasound.runtime_supervisor import atomic_json
    output=Path(output);budget=EvaluationBudget(policy['evaluation_max_batch'],policy['gpu_reserve_fraction'])
    original=training.evaluate;started=time.perf_counter();stats={'batches':0,'records':0}
    host_budget=HostPrefetchBudget(policy['prefetch_depth'],enabled=policy.get('adaptive_cpu_prefetch',False),
        busy_threshold=policy.get('cpu_busy_threshold',94.),reserve_gib=policy.get('host_memory_reserve_gib',4))
    original_batches=streaming.RFShardDataset.batches
    if policy.get('fast_eval_assembly',False):install_fast_assembly()
    def batches(self,size,device='cpu',*,generator=None):
        prior_threads=torch.get_num_threads()
        if not self.training and generator is None:torch.set_num_threads(policy.get('cpu_threads_evaluation',2))
        workers=policy.get('training_prefetch_workers',0) if self.training else policy.get('prefetch_workers',0)
        source=(prefetch_batches(self,size,device,generator,budget,workers=workers,
            depth=policy['prefetch_depth'],bucket_evaluation=policy['bucket_evaluation'],host_budget=host_budget) if workers else
            original_batches(self,size,device,generator=generator))
        try:
            for batch,ids in source:
                stats['batches']+=1;stats['records']+=len(ids)
                yield batch,ids
        finally:
            if hasattr(source,'close'):source.close()
            torch.set_num_threads(prior_threads)
        if output.exists():
            atomic_json(output/'execution_performance.json',dict(policy=policy,stats=stats,
                elapsed_seconds=time.perf_counter()-started,oom_retries=budget.oom_retries,
                adaptive_maximum=budget.maximum,training_batch_order_preserved=True,precision='float32',
                recent_memory_events=budget.events[-20:],recent_host_events=host_budget.events))
    def evaluate(model,data,*,batch_size=16,device='cpu'):
        return adaptive_evaluate(original,budget,model,data,batch_size=batch_size,device=device)
    streaming.RFShardDataset.batches=batches
    training.evaluate=evaluate;stream_training.evaluate=evaluate
    return budget
