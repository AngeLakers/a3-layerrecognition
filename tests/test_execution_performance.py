import numpy as np
import pytest
import torch

from a3_ultrasound.execution_performance import EvaluationBudget,HostPrefetchBudget,plans,adaptive_evaluate,prefetch_batches,fast_eval_assembly


def test_training_order_and_batch_members_are_identical():
    class Data:
        def __len__(self):return 103
    a=torch.Generator().manual_seed(23);b=torch.Generator().manual_seed(23)
    expected=torch.randperm(103,generator=a).numpy()
    groups=list(plans(Data(),16,b,None,EvaluationBudget()))
    assert np.array_equal(np.concatenate(groups),expected)
    assert [len(x) for x in groups]==[16]*6+[7]
    assert torch.equal(a.get_state(),b.get_state())


def test_evaluation_groups_exact_lengths_and_visits_each_row_once():
    class Data:
        def __len__(self):return 6
    lengths=np.array([4,2,4,3,2,2]);budget=EvaluationBudget(2)
    groups=list(plans(Data(),128,None,lengths,budget))
    assert sorted(np.concatenate(groups).tolist())==list(range(6))
    assert all(len(set(lengths[g]))==1 and len(g)<=2 for g in groups)


def test_evaluation_oom_halves_only_evaluation_batch_and_retries_same_rows():
    seen=[];data=object();model=object();budget=EvaluationBudget(8)
    def evaluate(m,d,*,batch_size,device):
        assert m is model and d is data;seen.append(batch_size)
        if batch_size>2:raise torch.OutOfMemoryError('injected OOM')
        return {'ok':True},list(range(7))
    metrics,rows=adaptive_evaluate(evaluate,budget,model,data,batch_size=8,device='cpu')
    assert seen==[8,4,2] and rows==list(range(7)) and budget.oom_retries==2
    assert metrics=={'ok':True} and budget.maximum==2


def test_prefetch_preserves_order_despite_out_of_order_cpu_completion():
    import time
    class Data:
        training=True
        def __len__(self):return 10
        def make_batch(self,ids,device):
            if ids[0]==0:time.sleep(.05)
            return ids.copy()
    observed=[ids.tolist() for _,ids in prefetch_batches(Data(),2,'cpu',None,EvaluationBudget(),bucket_evaluation=False)]
    assert observed==[[0,1],[2,3],[4,5],[6,7],[8,9]]


def test_host_prefetch_backs_off_for_cpu_and_memory_then_recovers(monkeypatch):
    import psutil
    from types import SimpleNamespace
    cpu=[97.,20.,20.];memory=[10*1024**3,2*1024**3,10*1024**3]
    monkeypatch.setattr(psutil,'cpu_percent',lambda:cpu.pop(0))
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=memory.pop(0)))
    budget=HostPrefetchBudget(12,enabled=True)
    assert budget.limit()==6
    budget.checked=0.;assert budget.limit()==3
    budget.checked=0.;assert budget.limit()==4


def test_changing_prefetch_depth_preserves_every_row_and_order():
    class Data:
        training=True
        def __len__(self):return 23
        def make_batch(self,ids,device):return ids.copy()
    class Changing:
        count=0
        def limit(self):
            self.count+=1
            return 1 if self.count%3==0 else 4
    observed=[ids.tolist() for _,ids in prefetch_batches(Data(),2,'cpu',None,EvaluationBudget(),
        workers=4,depth=4,bucket_evaluation=False,host_budget=Changing())]
    assert [i for batch in observed for i in batch]==list(range(23))


def test_vectorized_eval_assembly_is_exact_and_training_keeps_original_path():
    from a3_ultrasound.models.backbone import _encode_by_valid_length
    torch.manual_seed(23);encoder=torch.nn.Conv1d(1,3,3,padding=1).eval()
    x=torch.randn(5,17);lengths=torch.tensor([17,12,17,9,12]);mask=torch.arange(17)[None,:]<lengths[:,None]
    with torch.no_grad():
        expected=_encode_by_valid_length(x,mask,encoder,output_dim=3)
        actual=fast_eval_assembly(_encode_by_valid_length,x,mask,encoder,output_dim=3)
    for name in ('features','feature_centers','feature_mask'):assert torch.equal(getattr(actual,name),getattr(expected,name))
    calls=[]
    def original(*args,**kwargs):calls.append(True);return 'training path'
    encoder.train()
    with torch.no_grad():assert fast_eval_assembly(original,x,mask,encoder,output_dim=3)=='training path'
    assert calls==[True]
