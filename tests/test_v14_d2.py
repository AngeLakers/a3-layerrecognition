"""D2 budget, selection, and independently countable error diagnostics."""
import pytest

from a3_ultrasound.d2_diagnostic import DiagnosticSelector, summarize_rows, validate_config


def config():
    return dict(protocol='V1.4-D2', development_stage='D2', seeds=[23],
        heads=['proposed', 'vanilla_chain'], label_version='V2', folds=[dict(fold='F1')],
        normal_to_positive=3, evaluation_batch_size=128, development_training_permitted=True,
        formal_training_permitted=False, outer_evaluation_permitted=False,
        model=dict(architecture_version='a3-v1.4', backbone='resnet1d', backbone_norm='group8',
                   query_view='position2', rho=3.),
        training=dict(seed=23, epochs=10, batch_size=16, learning_rate=.001, weight_decay=0.,
            lambda_local=.3, positive_class_weight=None, disable_local_auxiliary=False,
            scheduler_step_size=None, device='cuda', min_delta=.002,
            selection_rule='diagnostic_joint', normal_fpr_limit=None,
            outer_evaluation_permitted=False, real_training_permitted=True))


def test_high_fpr_never_excludes_best_or_stops():
    selector = DiagnosticSelector()
    first = dict(positive_joint_map_accuracy=.7, joint_nll=.4, normal_joint_false_positive_rate=.6)
    assert selector.update(1, first)['best_updated']
    second = dict(first, positive_joint_map_accuracy=.701, joint_nll=.3, normal_joint_false_positive_rate=.9)
    assert selector.update(2, second)['best_updated']
    assert selector.best['epoch'] == 2
    for epoch in range(3, 11):
        assert selector.update(epoch, first)['stop'] is False
    assert selector.best['normal_fpr'] == .9


def test_point_two_pp_band_uses_nll_and_outside_uses_accuracy():
    selector = DiagnosticSelector()
    def m(score, nll):
        return dict(positive_joint_map_accuracy=score, joint_nll=nll, normal_joint_false_positive_rate=.5)
    selector.update(1, m(.7, .4))
    assert selector.update(2, m(.698, .3))['best_updated']
    assert not selector.update(3, m(.699, .5))['best_updated']
    assert selector.update(4, m(.701, .8))['best_updated']
    assert not selector.update(5, m(.701, .8))['best_updated']
    assert selector.best['epoch'] == 4


@pytest.mark.parametrize('section,key,value', [
    (None, 'heads', ['proposed', 'proposed_shadow_aux']),
    (None, 'seeds', [41]), (None, 'outer_evaluation_permitted', True),
    (None, 'formal_training_permitted', True), (None, 'development_training_permitted', False),
    ('training', 'epochs', 9), ('training', 'normal_fpr_limit', .01),
    ('training', 'selection_rule', 'guarded_joint'), ('model', 'backbone_norm', 'batch'),
    ('model', 'rho', 1.), ('model', 'query_view', 'legacy3'),
    (None, 'initial_checkpoint', 'old.pt')])
def test_scope_rejects_old_or_expanded_protocol(section, key, value):
    cfg = config()
    (cfg if section is None else cfg[section])[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('section,key,value', [(None,'evaluation_batch_size',64),
    ('training','amp',True),('training','dtype','float16'),(None,'optimizer','SGD')])
def test_frozen_eval_batch_fp32_adamw(section,key,value):
    cfg=config();(cfg if section is None else cfg[section])[key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_draft_can_be_checked_without_opening_training():
    cfg = config(); cfg['development_training_permitted'] = False
    cfg['save_epoch_checkpoints'] = True
    validate_config(cfg, require_permission=False)


def row(y, pred, *, w=1, c=None, g=None, mask=None):
    return dict(y=y, map_states=pred, label_mask=mask or [True]*len(y), sample_weight=w,
        correction=c, gate=g, gate_times_correction=[a*b for a, b in zip(c, g)] if c else None,
        rho=3 if c else None)


def test_error_categories_partition_complete_errors_and_separate_abnormals():
    rows = [row([1, 0], [1, 1], w=3), row([1, 1], [1, 0], w=2),
            row([0, 1], [1, 0], w=4), row([0, 0], [0, 1], w=5),
            row([1, 1], [1, 1]), row([1, 0], [1, 1], mask=[True, False])]
    result = summarize_rows(rows)
    assert result['errors']['raw'] == dict(single_to_multiple=1, multiple_to_single=1, other=2)
    assert result['errors']['ipw'] == dict(single_to_multiple=3, multiple_to_single=2, other=9)
    assert result['errors']['abnormal_ipw']['other'] == 4
    assert result['proposed'] is None
    assert summarize_rows([dict(y=None,label_mask=None,map_states=[])])['evaluated_rows'] == 1


def test_wrong_position_saturation_gate_medians_and_prefix_direction():
    rows = [row([1, 0, 1], [1, 1, 1], w=2, c=[0, -3, 2], g=[0, .2, .8]),
            row([1, 1], [0, 0], w=3, c=[0, 3], g=[0, .6])]
    r = summarize_rows(rows)
    allp, wrong = r['proposed']['all_nonfirst'], r['proposed']['wrong_nonfirst']
    assert allp['raw_count'] == 3 and allp['c_saturation_raw'] == pytest.approx(2/3)
    assert allp['c_saturation_ipw'] == pytest.approx(5/7)
    assert wrong['raw_count'] == 2 and wrong['gate_median'] == pytest.approx(.4)
    assert wrong['c_saturation_raw'] == 1
    assert r['correction_direction']['upper_correct']['raw_correct'] == 1
    assert r['correction_direction']['upper_wrong']['ipw_accuracy'] == 1


def test_training_loop_finishes_ten_despite_bad_recorded_metrics(tmp_path, monkeypatch):
    """Exercise the actual loop/checkpoint path with tiny disposable fake tensors."""
    import json
    from types import SimpleNamespace
    import torch
    import a3_ultrasound.d2_diagnostic as d2

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = torch.nn.Linear(1, 1, bias=False)
        def to(self, _device):
            return self
        def forward(self, batch, y):
            return SimpleNamespace(conditional=self.backbone.weight.square().sum())

    class Data:
        def __len__(self):
            return 1
        def batches(self, *args):
            yield SimpleNamespace(y=torch.zeros(1)), [0]

    def evaluate(_model, _data, *, prediction_path, **kwargs):
        prediction_path.write_text(json.dumps(row([1, 0], [1, 1], c=[0, 3], g=[0, 1]))+'\n')
        return dict(reliable=dict(positive_joint_map_accuracy=.3, joint_nll=2.,
                                  normal_joint_false_positive_rate=.99))

    monkeypatch.setattr(d2, 'datasets', lambda cfg: (Data(), Data()))
    monkeypatch.setattr(d2, 'build_model', lambda cfg: Tiny())
    monkeypatch.setattr(d2, 'use_epoch', lambda *a, **k: 'same-frozen-order')
    monkeypatch.setattr(d2, 'evaluate_stream', evaluate)
    monkeypatch.setattr(d2, 'head_training_loss', lambda cond, *a, **k: SimpleNamespace(total=cond))
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'test-only')
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('D2 must not load initialization checkpoints'))
    cfg=config(); cfg['catalog_signature']='test'; cfg['epoch_selection_root']='test'
    result=d2.train_one(cfg,tmp_path/'run','proposed')
    history=json.loads((tmp_path/'run/history.json').read_text())
    assert result['epochs']==len(history)==10
    assert len(list((tmp_path/'run').glob('epoch_*.pt')))==10
    assert all(r['diagnostics']['proposed']['all_nonfirst']['c_saturation_raw']==1 for r in history)
    assert result['best_epoch']==1 and result['test_opened'] is False
