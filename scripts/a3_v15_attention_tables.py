"""Raw and IPW summaries of saved front-180 attention; no new inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from a3_v15_summarize_ep12 import quantiles


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True);a=p.parse_args()
    decision=json.loads((a.input/'M6_decision.json').read_text(encoding='utf-8'))
    with np.load(a.input/'pre_attention.npz',allow_pickle=False) as z:
        b=z['board'];k=z['k'];w=z['weight'];mass=z['mass']
        assert len(k)==93338 and np.allclose(mass.sum(1),1.,atol=2e-6)
        assert len(set(z['sample_id']))==40002
        groups=[]
        for board in ['ALL']+sorted(set(b)):
            for layer in np.unique(k):
                m=(k==layer)&((b==board) if board!='ALL' else True)
                if m.any():groups.append(dict(board=board,k=int(layer),raw_n=int(m.sum()),ipw_n=int(w[m].sum()),raw=np.quantile(mass[m],[.25,.5,.75],axis=0).T.tolist(),ipw=[quantiles(mass[m,j],w[m]) for j in range(3)]))
        nonfirst=k>=2
        median=float(np.median(mass[nonfirst,0]))
        assert median==decision['front_mass_nonfirst_raw_median']
        ipw=quantiles(mass[nonfirst,0],w[nonfirst])[1]
    result=dict(groups=groups,nonfirst_raw_median=median,nonfirst_ipw_median=ipw,enable_m6=decision['enable_m6'],region_order=['front_0_180','upstream_excluding_front','other_Pre'])
    a.reports.mkdir(exist_ok=True,parents=True)
    (a.reports/'D2_front180_tables.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# D2 独立前窗注意力统计','',f"非首界面前窗质量raw中位数={median:.9g}，IPW中位数={ipw:.9g}。预先固定raw >0.5规则：enable_m6={decision['enable_m6']}。",'','前窗[0,180]由训练板确定后固定；旧epoch12及40002验证用于开发架构决定，不作独立确认。区域与Pre支持求交并互斥，质量和为1。IPW分位数按整数复制的线性分位数定义。','']
    for scope in ('raw','ipw'):
        lines+=['## '+scope,'','| 板 | k | 原始界面数 | IPW权重 | 前窗q25/median/q75 | 上游 | 其他 |','|---|---:|---:|---:|---|---|---|']
        for g in groups:
            values=[' / '.join(f'{v:.6g}' for v in r) for r in g[scope]]
            lines.append(f"| {g['board']} | {g['k']} | {g['raw_n']} | {g['ipw_n']} | "+' | '.join(values)+' |')
    (a.reports/'D2_front180_tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',raw_median=median,ipw_median=ipw,enable_m6=decision['enable_m6'])))


if __name__=='__main__':main()
