"""Exact count/sum aggregation without keeping every prediction in RAM."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np


class BinaryStats:
    def __init__(self):self.cm=np.zeros((2,2),dtype=np.int64);self.brier=0.;self.nll=0.
    def add(self,y,p,weight=1):
        self.cm[int(y),int(p>=.5)]+=weight;p=min(1-1e-7,max(1e-7,float(p)))
        self.brier+=weight*(p-y)**2;self.nll-=weight*(y*math.log(p)+(1-y)*math.log1p(-p))
    def result(self):
        n=int(self.cm.sum());fs=[]
        for k in (0,1):
            den=int(self.cm[k].sum()+self.cm[:,k].sum())
            fs.append(2*int(self.cm[k,k])/den if den else 0.)
        return {'count':n,'macro_f1':float(np.mean(fs)) if n else None,
            'accuracy':float(np.trace(self.cm)/n) if n else None,'brier':self.brier/n if n else None,
            'marginal_nll':self.nll/n if n else None,'confusion_matrix_0_1':self.cm.tolist()}


class PredictionAccumulator:
    def __init__(self):
        self.interfaces=defaultdict(BinaryStats);self.deepest=defaultdict(BinaryStats);self.upper=defaultdict(BinaryStats)
        self.n=self.labeled=self.joint_n=self.joint_correct=0;self.joint_nll=0.;self.max_logz=0.
        self.base_errors=self.wrong_right=self.right_wrong=self.beyond=0;self.has_rho=False
        self.single_n=self.wrong_layer=0
        self.single_raw_n=self.single_to_multi=self.single_to_multi_raw=0
        self.multi_n=self.multi_raw_n=self.multi_missed=self.multi_missed_raw=0
        self.multi_joint_wrong=self.multi_joint_wrong_raw=0
        self.shadow_positive_n=self.shadow_positive_raw_n=self.shadow_recalled=self.shadow_recalled_raw=0
        self.evaluated_n=0;self.weighted=False
        self.positive_n=self.positive_correct=self.normal_n=self.normal_fp=0
        self.raw_positive_n=self.raw_positive_correct=self.raw_normal_n=self.raw_normal_fp=0
        self.c_values=[];self.gc_values=[];self.c_weights=[];self.c_saturated=[]
        self.direction={key:[0,0,0,0] for key in ('upper_correct','upper_wrong')}
    def add(self,row):
        weight=row.get('sample_weight',1)
        if type(weight) is not int or weight<1:raise ValueError('Positive integer inverse-probability weight required.')
        self.evaluated_n+=1;self.weighted|=weight!=1
        self.n+=weight;self.max_logz=max(self.max_logz,abs(row['raw_logZ']))
        y,m,p=row['y'],row['label_mask'],row['marginals']
        self.has_rho|=row['rho'] is not None
        if y is None:return
        # A known abnormal upstream truth is sufficient; unknown labels are never filled with zero.
        for k in range(1,len(y)):
            if m[k] and y[k]==1 and any(m[j] and y[j]==1 for j in range(k)):
                recalled=row['map_states'][k]==1
                self.shadow_positive_n+=weight;self.shadow_positive_raw_n+=1
                self.shadow_recalled+=weight*recalled;self.shadow_recalled_raw+=recalled
        if row.get('correction') is not None and row.get('rho') is not None:
            for k in range(1,len(y)):
                if not m[k]:continue
                c=float(row['correction'][k]);gc=float(row['gate_times_correction'][k])
                if not math.isfinite(c) or not math.isfinite(gc):raise ValueError('Nonfinite correction diagnostic.')
                self.c_values.append(c);self.gc_values.append(gc);self.c_weights.append(weight)
                self.c_saturated.append(abs(c)>=.95*row['rho'])
                if all(m[:k]):
                    key='upper_correct' if row['map_states'][:k]==y[:k] else 'upper_wrong'
                    correct=(c>0 and y[k]==1) or (c<0 and y[k]==0)
                    counts=self.direction[key]
                    counts[0]+=weight;counts[1]+=weight*correct;counts[2]+=1;counts[3]+=correct
        for k,known in enumerate(m):
            if not known:continue
            self.interfaces[str(k+1)].add(y[k],p[k],weight);self.labeled+=weight
            bw=(row['base_probabilities'][k]>=.5)!=bool(y[k]);fw=(p[k]>=.5)!=bool(y[k])
            self.base_errors+=weight*bw;self.wrong_right+=weight*(bw and not fw);self.right_wrong+=weight*(not bw and fw)
            self.beyond+=weight*bool(row['rho'] is not None and bw and abs(row['base_logodds'][k])>row['rho'])
        if m[-1]:
            self.deepest[row['specimen_id']].add(y[-1],p[-1],weight)
            if all(m[:-1]):self.upper[f'K{len(y)}:'+''.join(map(str,y[:-1]))].add(y[-1],p[-1],weight)
        if all(m):
            correct=row['map_states']==y
            if any(y):
                self.positive_n+=weight;self.positive_correct+=weight*correct
                self.raw_positive_n+=1;self.raw_positive_correct+=correct
            else:
                self.normal_n+=weight;self.normal_fp+=weight*(not correct)
                self.raw_normal_n+=1;self.raw_normal_fp+=not correct
            self.joint_n+=weight;self.joint_correct+=weight*(row['map_states']==y)
            idx=sum(v << (len(y)-k-1) for k,v in enumerate(y));self.joint_nll-=weight*row['log_probabilities'][idx]
            if sum(y)==1:
                self.single_n+=weight
                self.single_raw_n+=1
                self.single_to_multi+=weight*(sum(row['map_states'])>1)
                self.single_to_multi_raw+=sum(row['map_states'])>1
                self.wrong_layer+=weight*(sum(row['map_states'])==1 and row['map_states']!=y)
            if sum(y)>=2:
                missed=any(v==1 and row['map_states'][k]==0 for k,v in enumerate(y))
                self.multi_n+=weight;self.multi_raw_n+=1
                self.multi_missed+=weight*missed;self.multi_missed_raw+=missed
                self.multi_joint_wrong+=weight*(not correct);self.multi_joint_wrong_raw+=not correct
    def result(self):
        per={k:v.result() for k,v in self.deepest.items()}
        def quantiles(values):
            return dict(zip(('min','q25','median','q75','max'),np.quantile(values,[0,.25,.5,.75,1]).tolist())) if values else None
        return {'sample_count':self.n,'labeled_interface_count':self.labeled,
            'evaluated_sample_count':self.evaluated_n,'sampling_weighted':self.weighted,
            'metric_scope':'inverse-probability population estimate' if self.weighted else 'all evaluated rows with unit weight',
            'raw_logZ_scope':'maximum over evaluated rows only',
            'per_interface':{k:v.result() for k,v in self.interfaces.items()},'deepest_per_specimen':per,
            'deepest_specimen_macro_f1':float(np.mean([v['macro_f1'] for v in per.values()])) if per else None,
            'deepest_by_true_upper_states':{k:v.result() for k,v in self.upper.items()},
            'joint_complete_label_count':self.joint_n,'joint_map_accuracy':self.joint_correct/self.joint_n if self.joint_n else None,
            'joint_nll':self.joint_nll/self.joint_n if self.joint_n else None,
            'base_error_count':int(self.base_errors),'base_wrong_to_exact_right':int(self.wrong_right),
            'base_right_to_exact_wrong':int(self.right_wrong),'base_wrong_beyond_rho_count':int(self.beyond) if self.has_rho else None,
            'base_wrong_beyond_rho_fraction_of_errors':self.beyond/self.base_errors if self.has_rho and self.base_errors else None,
            'raw_logZ_max_abs':self.max_logz,'single_positive_truth_count':self.single_n,
            'single_positive_wrong_layer_count':self.wrong_layer,
            'single_positive_wrong_layer_rate':self.wrong_layer/self.single_n if self.single_n else None,
            'wrong_layer_definition':'exact MAP predicts exactly one positive at a different interface; denominator all complete single-positive truths',
            'threshold':.5,'macro_f1_labels':[0,1],'macro_f1_zero_division':0,
            'positive_joint_map_accuracy':self.positive_correct/self.positive_n if self.positive_n else None,
            'multiple_defect_missing_any_rate':self.multi_missed/self.multi_n if self.multi_n else None,
            'multiple_defect_joint_error_rate':self.multi_joint_wrong/self.multi_n if self.multi_n else None,
            'single_defect_to_multiple_rate':self.single_to_multi/self.single_n if self.single_n else None,
            'downstream_abnormal_recall_given_upstream_abnormal':self.shadow_recalled/self.shadow_positive_n if self.shadow_positive_n else None,
            'defect_diagnostics':{
                'multiple_missing_any':{'weighted_numerator':self.multi_missed,'weighted_denominator':self.multi_n,
                    'raw_numerator':self.multi_missed_raw,'raw_denominator':self.multi_raw_n},
                'multiple_joint_error':{'weighted_numerator':self.multi_joint_wrong,'weighted_denominator':self.multi_n,
                    'raw_numerator':self.multi_joint_wrong_raw,'raw_denominator':self.multi_raw_n},
                'single_to_multiple':{'weighted_numerator':self.single_to_multi,'weighted_denominator':self.single_n,
                    'raw_numerator':self.single_to_multi_raw,'raw_denominator':self.single_raw_n},
                'downstream_recall_upstream_abnormal':{'weighted_numerator':self.shadow_recalled,'weighted_denominator':self.shadow_positive_n,
                    'raw_numerator':self.shadow_recalled_raw,'raw_denominator':self.shadow_positive_raw_n}},
            'normal_joint_false_positive_rate':self.normal_fp/self.normal_n if self.normal_n else None,
            'joint_subsets':{'positive':{'weighted_count':self.positive_n,'weighted_correct':self.positive_correct,
                'raw_count':self.raw_positive_n,'raw_correct':self.raw_positive_correct},
                'normal':{'weighted_count':self.normal_n,'weighted_false_positive':self.normal_fp,
                'raw_count':self.raw_normal_n,'raw_false_positive':self.raw_normal_fp}},
            'correction_saturation_fraction_MAP_path':float(np.mean(self.c_saturated)) if self.c_saturated else None,
            'correction_saturation_fraction_MAP_path_weighted':float(np.average(self.c_saturated,weights=self.c_weights)) if self.c_saturated else None,
            'correction_quantiles':quantiles(self.c_values),
            'gate_times_correction_quantiles':quantiles(self.gc_values),
            'correction_quantile_scope':'exact raw evaluated reliable non-first MAP-path interfaces; numpy linear quantiles',
            'correction_direction':{key:{'weighted_count':v[0],'weighted_correct':v[1],
                'raw_count':v[2],'raw_correct':v[3],'accuracy':v[1]/v[0] if v[0] else None,
                'raw_accuracy':v[3]/v[2] if v[2] else None} for key,v in self.direction.items()}}
