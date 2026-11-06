"""R3 D2 decision from a complete table; no RF reads, no threshold tuning."""
import math
import numpy as np


def choose_channel(rows, planned_events):
    """rows: fold,board,group,x,r,d AUCs and fit_normal_support.

    planned_events are all five validation fold-board events (TC4 only).
    K2:0 and K3:00 are checked separately; K3:01/10 separately. Missing
    classes/support/events returns INSUFFICIENT, never cancellation by default.
    """
    from a3_ultrasound.v16_protocol import template
    expected={(f['fold'],b,2 if b.startswith('TCL3-') else 3)
              for f in template()['folds'] for b in f['validation'] if b.startswith(('TCL3-','TCL4-'))}
    provided={(e['fold'],e['board'],e['K']) for e in planned_events}
    if provided!=expected or len(planned_events)!=len(expected):
        raise ValueError('D2 planned events must match the frozen five-fold TC4 validation panel.')
    needed={}
    for event in planned_events:
        for group in (('K2:0','K2:1') if event['K']==2 else ('K3:00','K3:01','K3:10')):
            needed[(event['fold'],event['board'],group)]=None
    seen=set()
    for row in rows:
        key=(row['fold'],row['board'],row['group'])
        if key in seen:raise ValueError('Duplicate D2 fold-board-group.')
        seen.add(key)
        if key in needed:needed[key]=row
    problems=[]
    for key,row in needed.items():
        if row is None or row.get('fit_normal_support',0)<200 or any(not isinstance(row.get(c),(int,float)) or not math.isfinite(row[c]) or not 0<=row[c]<=1 for c in ('x','r','d')):
            problems.append(key)
    if problems:return dict(status='INSUFFICIENT',channel=None,missing_or_unsupported=problems)
    groups={g:[r for (f,b,s),r in needed.items() if s==g] for g in ('K2:0','K2:1','K3:00','K3:01','K3:10')}
    if len(groups['K2:1'])!=5 or any(not v for v in groups.values()):
        return dict(status='INSUFFICIENT',channel=None,reason='Incomplete planned five-fold panel')
    checks={}
    for channel in ('r','d'):
        def gain(group,other='x'):return np.array([r[channel]-r[other] for r in groups[group]])
        values=[np.median(gain('K2:1'))>=.03,int((gain('K2:1')>0).sum())>=4,
                np.median(gain('K2:0'))>=-.02,np.median(gain('K3:00'))>=-.02,
                np.median(gain('K3:01'))>=0,np.median(gain('K3:10'))>=0]
        if channel=='r':values.append(np.median(gain('K2:1','d'))>=-.02)
        checks[channel]=list(map(bool,values))
    channel='residual' if all(checks['r']) else 'free_deep' if all(checks['d']) else 'none'
    return dict(status='DECIDED',channel=channel,checks=checks,
                matrix_cells=65 if channel=='none' else 95)
