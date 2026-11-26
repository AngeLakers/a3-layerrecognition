from dataclasses import replace
from itertools import product

import pytest
import torch

from a3_ultrasound.inference.partial_report import partial_report, state_probabilities
from a3_ultrasound.models.combination_cost import combination_cost
from a3_ultrasound.models.losses import head_training_loss
from a3_ultrasound.models.shallow_review import ShallowReview
from a3_ultrasound.training import ModelConfig, build_model, synthetic_datasets
from a3_ultrasound.v171_protocol import FIXED_MODEL
from a3_ultrasound.v185_cloud import BACKBONES, loss_arms
from a3_ultrasound.v185_metrics import events
from a3_ultrasound.v185_post import fit_policy, evaluate, fit_review


@pytest.mark.parametrize("backbone", BACKBONES)
def test_four_backbones_share_readout_and_loss(backbone):
    batch = synthetic_datasets(23, 4)["train"].batch
    cfg = ModelConfig(**(FIXED_MODEL | {"architecture_version":"a3-v1.8"}), backbone=backbone,
                      head_type="local_independent", relative_time_attention=True, readout_scope="full_waveform")
    model = build_model(replace(cfg,reference_bias="late"))
    enc = model.encode(batch)
    out = model.conditional(enc,batch.y)
    loss = head_training_loss(out,batch,head_type=cfg.head_type,v185_loss="exchange")
    loss.total.backward()
    assert torch.isfinite(loss.total) and model.local_readout.relative_time_weight.grad is not None
    assert enc.local_readout.token.shape[:2] == batch.interface_mask.shape
    for row,k in enumerate(batch.interface_mask.sum(1).tolist()):
        p = state_probabilities(out.logits[row:row+1,:k])
        a,_ = partial_report(p,k,point_fee=.3)
        assert a.shape == (1,k)


def test_exchange_cost_and_gradient_against_enumeration():
    z = torch.tensor([[.5,-.2,.3,1.7],[-.4,.6,.1,0.]],requires_grad=True)
    truth = torch.tensor([[1,0,1,0],[0,1,-1,-1]])
    valid = truth >= 0
    costs = []
    for i,k in enumerate(valid.sum(1).tolist()):
        states = torch.tensor(list(product((0,1),repeat=k)))
        fn = ((truth[i,:k]==1) & (states==0)).sum(1)
        fp = ((truth[i,:k]==0) & (states==1)).sum(1)
        cost = fn + fp + ((fn==fp) & (fn>0))
        costs.append((state_probabilities(z[i:i+1,:k])*cost).sum()/k)
    actual = combination_cost(z,truth,valid,exchange=True)
    expected = torch.stack(costs)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),z)[0],
                               torch.autograd.grad(expected.sum(),z)[0])


def test_pending_and_committed_can_both_be_locked():
    p = torch.tensor([[0.,0.,0.,0.,0.,0.,0.,1.]])
    a,r = partial_report(p,3,point_fee=.4,locked=torch.tensor([[0,-1,-1]]),
                         frozen=torch.tensor([[True, True, False]]))
    assert a.tolist()==[[0,-1,1]] and r.item()==pytest.approx(1.65)


def test_shallow_review_uses_only_eligible_donors():
    m = ShallowReview(4,"features")
    h = torch.randn(2,4,4,requires_grad=True)
    logits = torch.tensor([[5.,0.,0.,0.],[-5.,0.,0.,0.]])
    valid = torch.tensor([[True,True,True,True],[True,True,False,False]])
    action = torch.tensor([[1,-1,0,-1],[-1,-1,-1,-1]])
    args = (h,logits,torch.arange(4).float().expand(2,-1),valid,action,logits.sigmoid())
    updated,active = m(*args)
    torch.testing.assert_close(updated,logits)
    assert active.tolist()==[[False,True,False,True],[False,False,False,False]]
    updated.sum().backward()
    assert h.grad is None and m.update[-1].bias.grad.item()==2
    assert sum(len(loss_arms(b)) for b in BACKBONES)*5 == 50


def test_partial_reports_keep_all_known_targets_in_metrics():
    row = dict(truth=[0,1,1],report=[1,-1,0],known=[True]*3,active=[False,True,False])
    c = events(row)
    assert (c["deep_detected"],c["deep_pending"],c["deep_missed"],c["deep_positive"])==(0,1,1,2)
    assert c["exchange"]==1 and c["review_points"]==1 and c["trigger_pending"]==1
    c = events(dict(row, truth=[0]*3))
    assert c["normal_fp"]==1 and c["normal"]==1


def test_postprocessing_synthetic_replay(tmp_path):
    import json
    y = torch.tensor([[0,0],[1,1],[1,0],[0,1]])
    z = torch.tensor([[-5.,-4.],[5.,.01],[5.,-.01],[-5.,4.]])
    cache = dict(z=z,y=y,h=torch.randn(4,2,4),valid=torch.ones_like(y).bool(),
                 tau=torch.tensor([[.2,.7]]*4),known=torch.ones_like(y).bool(),weights=torch.ones(4),
                 boards=["A","B","B","A"],sample_ids=list("abcd"),checkpoint="synthetic")
    policy = fit_policy(cache,.5)
    head = fit_review(cache,policy,"features")
    out = tmp_path/"report.jsonl"
    evaluate(cache,policy,head,"cost",out,True)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows)==4 and head["targets"]>0
    for row in rows:
        assert all(a==b for a,b,can_change in zip(row["initial"],row["report"],row["active"]) if not can_change)



def test_state_calibration_scores_and_common_eligibility():
    c = events(dict(truth=[1,0],report=[1,-1],known=[True,True],active=[False,False],
                    eligible=[False,True],logits=[0.,0.],temperature=1.))
    assert c["triggered"] == 1 and c["trigger_pending"] == 1
    assert c["raw_state_nll_sum"] == pytest.approx(1.38629436)
    assert c["cal_state_brier_sum"] == pytest.approx(.75)


def test_infeasible_normal_guard_is_marked():
    y = torch.tensor([[0,0],[0,1]])
    z = torch.tensor([[16.,-16.],[-16.,16.]])
    v = torch.ones_like(y).bool()
    p = fit_policy(dict(y=y, z=z, valid=v, known=v, weights=torch.ones(2),
                        boards=["A","A"],checkpoint="demo"), .1)
    assert all(not point["guard_met"] and point["normal_fpr"] == 1
               for point in p["operating_points"].values())
