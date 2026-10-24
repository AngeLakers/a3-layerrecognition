"""Read saved CPU diagnostic outputs, independently check pairs, tabulate D2 IPW."""
import argparse
import json
from pathlib import Path
import numpy as np


def quantiles(x,w):
    order=np.argsort(x,kind='stable');x=x[order];w=w[order]
    cumulative=np.cumsum(w);rank=np.array([.25,.5,.75])*(int(w.sum())-1)
    low=np.floor(rank).astype(int);high=np.ceil(rank).astype(int)
    a=x[np.searchsorted(cumulative,low,side='right')]
    b=x[np.searchsorted(cumulative,high,side='right')]
    return (a+(b-a)*(rank-low)).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True);a=p.parse_args()
    result=json.loads((a.input/'ep12_reenumeration.json').read_text(encoding='utf-8'))
    rows=[json.loads(s) for s in (a.input/'paired_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==40002 and len({r['sample_id'] for r in rows})==40002
    verified={}
    for subset in ['all','abnormal','normal']:
        selected=[r for r in rows if all(r['label_mask']) and
                  (subset=='all' or bool(any(r['y']))==(subset=='abnormal'))]
        good=[r for r in selected if r['on']==r['y'] and r['off']!=r['y']]
        bad=[r for r in selected if r['on']!=r['y'] and r['off']==r['y']]
        check=dict(on_correct_off_wrong_raw=len(good),on_wrong_off_correct_raw=len(bad),
            on_correct_off_wrong_ipw=sum(r['sample_weight'] for r in good),
            on_wrong_off_correct_ipw=sum(r['sample_weight'] for r in bad))
        assert all(result['paired']['all'][subset][k]==v for k,v in check.items())
        verified[subset]=check
    with np.load(a.input/'pre_attention.npz',allow_pickle=False) as z:
        board=z['board'];k=z['k'];weights=z['weight'];mass=z['mass']
        assert len(k)==93338 and np.allclose(mass.sum(1),1.,atol=2e-6)
        groups=[]
        for b in ['all']+sorted(set(board)):
            for j in sorted(set(k)):
                take=(k==j)&((board==b) if b!='all' else True)
                if not take.any():continue
                groups.append(dict(board=b,k=int(j),raw_n=int(take.sum()),ipw_n=int(weights[take].sum()),
                    raw_q25_median_q75=np.quantile(mass[take],[.25,.5,.75],axis=0).T.tolist(),
                    ipw_q25_median_q75=[quantiles(mass[take,q],weights[take]) for q in range(3)]))
    a.reports.mkdir(parents=True,exist_ok=True)
    (a.reports/'ep12_pair_verification.json').write_text(json.dumps(dict(status='PASS',records=40002,paired=verified),indent=2),encoding='utf-8')
    (a.reports/'v15_pre_attention_tables.json').write_text(json.dumps(dict(groups=groups,region_order=['nominal_front_0_70','upstream_window_union_excluding_front','rest_of_Pre'],M6_frozen=False),ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# D2 名义前窗注意力分布','', '原0–70名义区域，仅描述开发checkpoint；物理窗口尚未定义，因此不冻结M6。各区域与实际Pre支撑取交，互斥且和为1。',
        '分位数按q25 / 中位数 / q75列出。IPW按整数权重复制的线性分位数定义。','']
    for scope in ['raw','ipw']:
        lines+=['## '+scope,'','| 板 | k | 原始界面数 | IPW总权 | 名义前窗 | 上游窗口并集 | 其他Pre |',
                '| --- | --- | --- | --- | --- | --- | --- |']
        for g in groups:
            values=[' / '.join(f'{v:.5g}' for v in x) for x in g[scope+'_q25_median_q75']]
            lines.append(f"| {g['board']} | {g['k']} | {g['raw_n']} | {g['ipw_n']} | "+' | '.join(values)+' |')
    (a.reports/'v15_pre_attention_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',pairs_verified=True,attention_groups=len(groups))))


if __name__=='__main__':main()
