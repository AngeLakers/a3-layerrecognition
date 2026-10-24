import json
import numpy as np
import pytest
from a3_ultrasound.phase60 import HEADS,SEEDS,FOLDS,phase_cells,prediction_diagnostics


def test_matrix_exact_order_and_duplicate_axes_rejected():
    cfg=dict(heads=list(HEADS),seeds=list(SEEDS),folds=[{'fold':f} for f in FOLDS])
    cells=phase_cells(cfg)
    assert len(cells)==60 and cells[0]==('F1',23,'local_independent') and cells[20]==('F1',41,'local_independent')
    cfg['seeds']=[23,23,59]
    with pytest.raises(ValueError,match='exact ordered'):phase_cells(cfg)


def test_joint_map_not_marginal_threshold_and_partial_exclusion(tmp_path):
    # Joint MAP=01 while both marginal thresholds would yield 11.
    p=[.09,.34,.26,.31]
    paths=np.array([[0,0],[0,1],[1,0],[1,1]])
    row=dict(K=2,sample_id='a:1',specimen_id='a',y=[1,1],label_mask=[True,True],
             probabilities=p,marginals=(np.array(p)@paths).tolist(),map_states=[0,1])
    partial=dict(row,sample_id='a:2',y=[1,-1],label_mask=[True,False])
    file=tmp_path/'rows.jsonl';file.write_text(json.dumps(row)+'\n'+json.dumps(partial)+'\n')
    result=prediction_diagnostics(file);a=result['entities']['a']
    assert a['confusion'][3][1]==1 and a['complete']==1 and a['partial']==1 and a['support'][3]==1
    invalid=dict(row,map_states=[1,1]);file.write_text(json.dumps(invalid)+'\n')
    with pytest.raises(ValueError,match='MAP'):prediction_diagnostics(file)
