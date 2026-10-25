import numpy as np
from a3_ultrasound.data.rf_observation import template_observation_gates,observed_peak
from a3_ultrasound.data.board_timebase import envelope_anchor_positions


def pulse(t,p,a=1.,w=8.):return a*np.exp(-((t-p)/w)**2)


def test_fixed_RF_template_avoids_front_and_previous_interface_contamination():
    t=np.arange(1400);e=pulse(t,79,200)+pulse(t,268,25)+pulse(t,522,18)+pulse(t,781,70)
    result=template_observation_gates(e,79.,781.,2)
    assert result['missing']==[]
    assert [r['template_peak'] for r in result['gates']]==[268,522]
    trace_env=pulse(t,97,170)+pulse(t,262,23)+pulse(t,434,10)+pulse(t,501,14)+pulse(t,783,33)
    gate=result['gates'][1];got=observed_peak(trace_env,gate)
    assert got['valid'] and got['peak']==501 and not got['search_edge']


def test_missing_packet_is_not_replaced_with_a_fraction_center():
    t=np.arange(900);e=pulse(t,74,200)+pulse(t,432,15)+pulse(t,608,70)
    result=template_observation_gates(e,74.,608.,2)
    assert result['missing']==[1] and [r['k'] for r in result['gates']]==[2]


def test_shared_gates_do_not_overlap_and_observer_ignores_compared_targets():
    t=np.arange(1000);e=pulse(t,70,200)+pulse(t,250,20)+pulse(t,400,15)+pulse(t,550,10)+pulse(t,800,80)
    gates=template_observation_gates(e,60.,800.,3)['gates'];assert len(gates)==3
    assert all(a['right']<b['left'] for a,b in zip(gates,gates[1:]))
    peak=observed_peak(e,gates[1])['peak']
    for nominal,calibrated in ((390.,430.),(400.,400.),(100.,900.)):
        assert observed_peak(e,gates[1])['peak']==peak
        assert (peak-calibrated)-(peak-nominal)==nominal-calibrated


def test_missing_or_zero_source_is_failed_and_edge_hit_is_flagged():
    assert not observed_peak(np.zeros(500),dict(left=200,right=550))['valid']
    assert not observed_peak(np.zeros(500),dict(left=200,right=300))['valid']
    x=np.zeros(500);x[200]=4
    assert observed_peak(x,dict(left=200,right=300))['search_edge']


def test_coarse_backwall_search_override_changes_only_RF_gate():
    t=np.arange(1400);x=pulse(t,79)*np.cos(t*.3)+pulse(t,519,.4)*np.cos(t*.3)+pulse(t,781,.8)*np.cos(t*.3)
    old=envelope_anchor_positions(x,570.)[1][0];new=envelope_anchor_positions(x,800.)[1][0]
    assert abs(old-519)<=1 and abs(new-781)<=1


def test_rf_census_resume_rejects_changed_timing_context(tmp_path):
    import importlib.util,json
    from pathlib import Path
    import pytest
    path=Path(__file__).parents[1]/'scripts/a3_v152_calibrate_rf.py'
    spec=importlib.util.spec_from_file_location('calibrate_context',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=tmp_path/'catalog.json';source.write_text(json.dumps(dict(bottom_search_center_samples=570)))
    out=tmp_path/'measurement';out.mkdir();m.bind_rf_catalog(out,source,False);m.bind_rf_catalog(out,source,True)
    source.write_text(json.dumps(dict(bottom_search_center_samples=800)))
    with pytest.raises(AssertionError,match='timing context'):m.bind_rf_catalog(out,source,True)
