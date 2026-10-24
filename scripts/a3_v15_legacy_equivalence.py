"""Compare old/new output and one optimizer update in isolated CPU interpreters."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CHILD = r'''
import sys,hashlib,json
sys.path.insert(0,sys.argv[1])
import torch
from a3_ultrasound.training import ModelConfig,TrainingConfig,build_model,build_optimizer,synthetic_datasets
from a3_ultrasound.models.losses import head_training_loss
torch.set_num_threads(1)
def digest(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
result={}
configuration=json.loads(sys.argv[2])
for head in ['local_independent','local_pre_independent','vanilla_chain','proposed','proposed_shadow_aux']:
    torch.manual_seed(23)
    model=build_model(ModelConfig(**dict(configuration,head_type=head)))
    data=synthetic_datasets(61,4)['train'].batch
    optim,_=build_optimizer(model,TrainingConfig())
    out=model(data,data.y).conditional
    loss=head_training_loss(out,data,head_type=head)
    before=dict(logits=digest(out.logits),base=digest(out.base_logodds),loss=digest(loss.total))
    loss.total.backward()
    gradients={n:digest(p.grad) if p.grad is not None else None for n,p in model.named_parameters()}
    optim.step();model.eval()
    with torch.inference_mode():exact=model.infer_exact(data)
    result[head]=dict(before=before,gradients=gradients,parameters={n:digest(p) for n,p in model.named_parameters()},
                      joint=digest(exact.log_probabilities),map=digest(exact.map_states),rng=digest(torch.get_rng_state()))
print(json.dumps(result))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-source',type=Path,required=True)
    p.add_argument('--model-config',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    current=Path(__file__).resolve().parents[1]/'src'
    configuration=json.loads(a.model_config.read_text(encoding='utf-8'))['model'] if a.model_config else dict(backbone='tcn',rho=3.)
    outputs=[]
    for src in [a.legacy_source,current]:
        run=subprocess.run([sys.executable,'-B','-c',CHILD,str(src),json.dumps(configuration)],capture_output=True,text=True,
            encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)
        outputs.append(json.loads(run.stdout.strip().splitlines()[-1]))
    same=outputs[0]==outputs[1]
    value=dict(status='PASS' if same else 'FAIL',legacy_source=str(a.legacy_source),current_source=str(current),
        scope='five legacy heads, synthetic inputs CPU, same seed: logits/loss/gradients/one AdamW update/exact inference/RNG bitwise hashes',
        model_configuration=configuration,
        results=outputs)
    a.output.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=value['status'],heads=list(outputs[0]))))
    if not same:raise RuntimeError('Legacy path changed')


if __name__=='__main__':main()
