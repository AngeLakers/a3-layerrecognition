"""Cloud stages: cache frozen features, fit a calibration working point, fit a residual, report."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from a3_ultrasound.v185_cloud import read, role_dataset
from a3_ultrasound.speed import SpeedOptions, configure_backend, configure_model, precision_context, speed_context
from a3_ultrasound.training import ModelConfig, TrainingConfig, build_model, scheduled_learning_rate
from a3_ultrasound.inference.partial_report import state_probabilities, partial_report
from a3_ultrasound.models.shallow_review import ShallowReview, eligible_donors


@torch.inference_mode()
def extract(config, run, role, device):
    cfg = read(config)
    summary = read(run/"summary.json")
    key = hashlib.sha256(config.read_bytes()).hexdigest()
    if (summary["config_sha256"] != key or summary["guard_violated"] is not False
            or summary["status"] != "COMPLETED" or summary["selected_checkpoint"] != "best_guarded.pt"):
        raise ValueError("Expected matching guarded checkpoint")
    path = run/summary["selected_checkpoint"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["resolved_config"]["config_sha256"] != key:
        raise ValueError("Checkpoint/config mismatch")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    speed = SpeedOptions(**cfg.get("speed", {}))
    configure_backend(speed)
    model = build_model(ModelConfig(**cfg["model"])).to(device).eval()
    configure_model(model,speed)
    model.load_state_dict(checkpoint["model_state_dict"])
    data = role_dataset(cfg, role)
    fold = cfg["folds"][0]["fold"]
    panel = config.parent/role/f"{fold}.npz"
    if hashlib.sha256(panel.read_bytes()).hexdigest() != read(panel.parent/"manifest.json")["folds"][fold]["sha256"]:
        raise ValueError("Frozen role panel mismatch")
    data.use_frozen_evaluation_selection(panel)
    fields = {k: [] for k in ("h","z","tau","valid","y","known")}
    sample_ids, boards, weights = [], [], []
    for batch, ids in data.batches(cfg.get("evaluation_batch_size", 16), device):
        with speed_context(speed,batch), precision_context(speed):
            enc = model.encode(batch)
            prefix = torch.zeros_like(batch.interface_mask, dtype=torch.long)
            z = model.conditional(enc, prefix).logits
        values=(enc.local_readout.token, z, batch.interface_meta[...,1], batch.interface_mask,batch.y,batch.label_mask)
        for (name, buffer), value in zip(fields.items(),values):
            pad = model.max_interfaces - value.shape[1]
            buffer.append(F.pad(value.cpu(), (0,0,0,pad) if name=="h" else (0,pad)))
        for i,anno in zip(ids,data.annotations(ids)):
            sid, board = data.identity(i)
            sample_ids.append(sid)
            boards.append(board)
            weights.append(anno["sample_weight"])
    return dict(**{k:torch.cat(v) for k,v in fields.items()},sample_ids=sample_ids,boards=boards,
                weights=torch.tensor(weights),role=role, config=key,loss_arm=cfg["training"]["v185_loss"],
                checkpoint=hashlib.sha256(path.read_bytes()).hexdigest())


def report(z, valid, temperature, knob, mode, locked=None, frozen=None):
    result=torch.full_like(z,-1,dtype=torch.long)
    for k in torch.unique(valid.sum(1)).tolist():
        at=(valid.sum(1)==k).nonzero().flatten()
        values=z[at,:k]
        prob=state_probabilities(values,temperature)
        if mode in ("cost","additive"):
            result[at,:k]=partial_report(prob,k,point_fee=knob,exchange=float(mode=="cost"),
                                         locked=None if locked is None else locked[at,:k],
                                         frozen=None if frozen is None else frozen[at,:k])[0]
        else:
            a=(values>0).long()
            confidence=torch.sigmoid(values.abs()/temperature)
            if mode=="whole":
                a[prob.max(1).values<knob]=-1
            elif mode=="layer":
                a[confidence<knob]=-1
            elif mode!="map":
                raise ValueError("Unknown decision mode")
            result[at,:k]=a
    return result


def board_weights(cache):
    w=cache["weights"].double().clone()
    _,groups=np.unique(cache["boards"],return_inverse=True)
    for g in np.unique(groups):
        at=torch.from_numpy(groups==g)
        w[at]/=w[at].sum()
    return w/w.sum()


def fit_policy(c,budget):
    z,y,v=c["z"],c["y"],c["valid"]
    mask=v & c["known"]
    w=board_weights(c)
    complete=(mask==v).all(1)
    if not complete.any(): raise ValueError("Calibration requires complete reliable records")
    def nll(t):
        loss=F.binary_cross_entropy_with_logits(z/t,y.clamp_min(0).float(),reduction="none")
        return (w[complete]*(loss*mask).sum(1)[complete]).sum().item()
    temp=min((1.,.5,.75,1.25,1.5,2.,3.),key=nll)
    if not 0 <= budget <= 1:
        raise ValueError("Review budget must be between zero and one")
    normal = complete & ~((y == 1) & v).any(1)
    normal_w = w.clone().masked_fill(~normal, 0.)
    board_ids = np.array(c["boards"])
    for board in np.unique(board_ids):
        at = torch.from_numpy(board_ids == board)
        mass = normal_w[at].sum()
        if mass > 0:
            normal_w[at] /= mass
    normal_weight = normal_w.sum().item()
    operating_points = {}
    knobs={}
    achieved={}
    for mode in ("whole","layer","additive","cost"):
        candidates=np.linspace(0,1,101) if mode in ("whole","layer") else np.linspace(0,2,81)
        choices=[]
        for knob in candidates:
            action=report(z,v,temp,float(knob),mode)
            rate=w[((action==-1)&v).any(1)].sum().item()
            if rate<=budget+1e-9:
                fp = normal_w[normal & ((action == 1) & v).any(1)].sum().item()
                fpr = fp/normal_weight if normal_weight else None
                passes = fpr is not None and fpr <= .02 + 1e-9
                choices.append((not passes,budget-rate,float(knob),rate,fpr))
        if not choices:
            raise ValueError(f"No calibration operating point within budget for {mode}")
        failed,_,knobs[mode],achieved[mode],fpr=min(choices)
        operating_points[mode] = dict(normal_fpr=fpr,guard_met=not failed)
    support = {}
    for board in np.unique(board_ids):
        at = torch.from_numpy(board_ids == board)
        support[str(board)] = dict(K=int(v[at].sum(1).unique().item()), points=int(at.sum()),
                                   complete=int(complete[at].sum()), normal=int(normal[at].sum()))
    return dict(temperature=temp,knobs=knobs,achieved=achieved,operating_points=operating_points,
                budget=budget,checkpoint=c["checkpoint"],calibration_boards=support)


def review_inputs(c,policy):
    t=policy["temperature"]
    a=report(c["z"],c["valid"],t,policy["knobs"]["cost"],"cost")
    return (c["h"],c["z"],c["tau"],c["valid"],a,torch.sigmoid(c["z"]/t))


def review_targets(cache, policy):
    inputs = review_inputs(cache, policy)
    _, _, _, valid, action, p = inputs
    known = valid & action.eq(-1) & eligible_donors(valid, action, p).any(-1) & cache["known"]
    weight = known.double() * cache["weights"][:, None]
    board_ids = np.asarray(cache["boards"])
    support = {}
    for board in np.unique(board_ids):
        at = torch.from_numpy(board_ids == board)
        support[str(board)] = known[at].sum(0).tolist()
        mass = weight[at].sum()
        if mass > 0:
            weight[at] /= mass
    rows = known.any(1)
    weight = weight[rows]
    if rows.any():
        weight *= int(rows.sum()) / weight.sum()
    return tuple(t[rows] for t in inputs), cache["y"][rows], weight, support


def review_bce(z, y, weights, temperature):
    terms = F.binary_cross_entropy_with_logits(z / temperature, y.clamp_min(0).float(), reduction="none")
    return (terms * weights).sum() / len(y)


def fit_review(c,policy,mode,stop_cache):
    if (c["role"] != "residual_fit" or stop_cache["role"] != "early_stop"
            or c["checkpoint"] != stop_cache["checkpoint"] or c["checkpoint"] != policy["checkpoint"]
            or c["config"] != stop_cache["config"] or c["loss_arm"] != "exchange"
            or stop_cache["loss_arm"] != "exchange"):
        raise ValueError("Review fitting needs residual and stopping caches from the same exchange model")
    fit_boards, stop_boards = set(c["boards"]), set(stop_cache["boards"])
    calibration_boards = set(policy["calibration_boards"])
    if fit_boards & stop_boards or (fit_boards | stop_boards) & calibration_boards:
        raise ValueError("Review training, stopping and calibration boards must be disjoint")
    torch.manual_seed(23)
    inputs, y, weights, fit_support = review_targets(c, policy)
    stop_inputs, stop_y, stop_weights, stop_support = review_targets(stop_cache, policy)
    result = dict(mode=mode, policy=policy, targets=int((weights>0).sum()), fit_support=fit_support,
                  stop_support=stop_support, history=[], selected_epoch=0)
    if not len(y) or not len(stop_y):
        return dict(result, status="skipped_no_fit_targets" if not len(y) else "skipped_no_stop_targets")
    model=ShallowReview(c["h"].size(-1),mode)
    schedule = TrainingConfig(epochs=10, lr_schedule="cosine", lr_warmup_epochs=1, lr_min=1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=schedule.learning_rate, weight_decay=0)
    t = policy["temperature"]
    best_loss = review_bce(stop_inputs[1], stop_y, stop_weights, t).item()
    result["initial_stop_bce"] = best_loss
    best_state = deepcopy(model.state_dict())
    for epoch in range(1, schedule.epochs + 1):
        batches = torch.randperm(len(y)).split(256)
        for step, ix in enumerate(batches):
            lr = scheduled_learning_rate(schedule, epoch - 1 + step / len(batches))
            opt.param_groups[0]["lr"] = lr
            z, _ = model(*(v[ix] for v in inputs))
            loss = review_bce(z, y[ix], weights[ix], t)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite review training loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            valid_z, _ = model(*stop_inputs)
            score = review_bce(valid_z, stop_y, stop_weights, t).item()
        if not np.isfinite(score):
            raise ValueError("Non-finite review validation loss")
        result["history"].append(dict(epoch=epoch, stop_bce=score, lr_last=lr))
        if score < best_loss:
            best_loss, best_state = score, deepcopy(model.state_dict())
            result["selected_epoch"] = epoch
        if epoch - result["selected_epoch"] >= 3:
            break
    return dict(result, status="fitted", state=best_state, epochs_run=epoch,
                stop_reason="patience" if epoch < schedule.epochs else "max_epochs")


def shuffled_source(c, inputs):
    """Permute only the donor representation, within board, slot, sign and confidence bin."""
    h, _, _, valid, action, prob = inputs
    source = h.clone()
    rng = torch.Generator().manual_seed(23)
    _, boards = np.unique(c["boards"], return_inverse=True)
    board = torch.from_numpy(boards)
    confidence = torch.maximum(prob, 1-prob)
    for k in range(h.size(1)-1):
        trusted = valid[:, k] & action[:, k].ne(-1) & confidence[:, k].ge(.9)
        group = board*4 + action[:, k]*2 + confidence[:, k].ge(.95).long()
        for g in torch.unique(group[trusted]):
            ids = (trusted & group.eq(g)).nonzero().flatten()
            order = ids[torch.randperm(len(ids), generator=rng)]
            source[order, k] = h[order.roll(1), k]
    return source


@torch.no_grad()
def evaluate(c,policy,head,mode,out,shuffle=False):
    z,v=c["z"],c["valid"]
    initial=report(z,v,policy["temperature"],policy["knobs"]["cost"],"cost")
    donors = eligible_donors(v, initial, torch.sigmoid(z/policy["temperature"]))
    eligible = v & initial.eq(-1) & donors.any(-1)
    active=torch.zeros_like(v)
    updated=z
    if head and (mode!="cost" or head["policy"]!=policy):
        raise ValueError("Residual requires its original policy")
    if head and head.get("status","fitted") == "fitted":
        model=ShallowReview(c["h"].size(-1),head["mode"])
        model.load_state_dict(head["state"])
        inputs=review_inputs(c,policy)
        source = shuffled_source(c, inputs) if shuffle else None
        if shuffle and head["mode"] != "features":
            raise ValueError("Shuffle requires the feature head")
        updated,active=model(*inputs, shallow_features=source)
        # Only initial pending, eligible slots may change. Others retain their initial action.
        action=report(updated,v,policy["temperature"],policy["knobs"][mode],mode,locked=initial,frozen=~active)
        action=torch.where(active,action,initial)
    else:
        if shuffle: raise ValueError("Shuffle requires a fitted feature head")
        action=report(z,v,policy["temperature"],policy["knobs"].get(mode,0),mode)
    with out.open("x",encoding="utf-8") as f:
        for i,k in enumerate(v.sum(1).tolist()):
            row=dict(sample_id=c["sample_ids"][i],board=c["boards"][i],weight=float(c["weights"][i]),
                     checkpoint=c["checkpoint"], budget=policy["budget"], mode=mode,
                     point_fee=policy["knobs"]["cost"], shuffled=shuffle,
                     review_head=head["mode"] if head else None, review_status=head.get("status","fitted") if head else "no_head",
                     logits=z[i,:k].tolist(),final_logits=updated[i,:k].tolist(),temperature=policy["temperature"],
                     truth=c["y"][i,:k].tolist(),known=c["known"][i,:k].tolist(),initial=initial[i,:k].tolist(),
                     calibration_guard_met=policy["operating_points"].get(mode,{}).get("guard_met"),
                     eligible=eligible[i,:k].tolist(),map=(z[i,:k]>0).long().tolist(),report=action[i,:k].tolist(),active=active[i,:k].tolist())
            f.write(json.dumps(row,ensure_ascii=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="op",required=True)
    cache=sub.add_parser("cache")
    cache.add_argument("config",type=Path); cache.add_argument("run",type=Path)
    cache.add_argument("role",choices=("early_stop","residual_fit","calibration_workpoint","dev_evaluation"))
    cache.add_argument("--device",default="cuda")
    for op in ("calibrate","fit-review","evaluate"):
        subp=sub.add_parser(op)
        subp.add_argument("cache",type=Path)
        if op=="calibrate":subp.add_argument("--budget",type=float,default=.1)
        else:subp.add_argument("policy",type=Path)
        if op=="fit-review":
            subp.add_argument("mode",choices=("self","scores","features"))
            subp.add_argument("--early-stop", type=Path, required=True)
        if op=="evaluate":
            subp.add_argument("--mode",choices=("map","whole","layer","additive","cost"),default="cost")
            subp.add_argument("--head",type=Path)
            subp.add_argument("--shuffle",action="store_true")
    for subp in sub.choices.values():subp.add_argument("output",type=Path)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.op=="cache": result=extract(a.config,a.run,a.role,a.device)
    else:
        c=torch.load(a.cache,map_location="cpu",weights_only=True)
        role={"calibrate":"calibration_workpoint","fit-review":"residual_fit","evaluate":"dev_evaluation"}[a.op]
        if c["role"]!=role:raise ValueError("Invalid role for operation")
        if a.op=="calibrate":result=fit_policy(c,a.budget)
        else:
            policy=read(a.policy)
            if policy["checkpoint"]!=c["checkpoint"]:raise ValueError("Checkpoint mismatch")
            if a.op=="fit-review":
                stop_cache=torch.load(a.early_stop,map_location="cpu",weights_only=True)
                result=fit_review(c,policy,a.mode,stop_cache)
            else:
                head=torch.load(a.head,map_location="cpu",weights_only=True) if a.head else None
                evaluate(c,policy,head,a.mode,a.output,a.shuffle)
                return
    if a.op=="calibrate":a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    else:torch.save(result,a.output)


if __name__=="__main__":
    main()

