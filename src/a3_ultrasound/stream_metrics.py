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
        self.evaluated_n=0;self.weighted=False
    def add(self,row):
        weight=row.get('sample_weight',1)
        if type(weight) is not int or weight<1:raise ValueError('Positive integer inverse-probability weight required.')
        self.evaluated_n+=1;self.weighted|=weight!=1
        self.n+=weight;self.max_logz=max(self.max_logz,abs(row['raw_logZ']))
        y,m,p=row['y'],row['label_mask'],row['marginals']
        self.has_rho|=row['rho'] is not None
        if y is None:return
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
            self.joint_n+=weight;self.joint_correct+=weight*(row['map_states']==y)
            idx=sum(v << (len(y)-k-1) for k,v in enumerate(y));self.joint_nll-=weight*row['log_probabilities'][idx]
            if sum(y)==1:
                self.single_n+=weight
                self.wrong_layer+=weight*(sum(row['map_states'])==1 and row['map_states']!=y)
    def result(self):
        per={k:v.result() for k,v in self.deepest.items()}
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
            'correction_quantiles':'deferred to saved prediction rows; not approximated online'}
