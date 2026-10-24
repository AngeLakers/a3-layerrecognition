"""V1.4 validation-only guarded selection shared by dense/stream/cache runners."""
from dataclasses import dataclass, asdict
import math


@dataclass
class GuardedSelector:
    min_epochs: int = 15
    patience: int = 10
    min_delta: float = .002
    normal_fpr_limit: float = .01
    guarded: dict | None = None
    fallback: dict | None = None
    stale: int = 0
    progress_anchor: float | None = None

    def __post_init__(self):
        if type(self.min_epochs) is not int or self.min_epochs < 0 or type(self.patience) is not int or self.patience < 1:
            raise ValueError('Invalid minimum epochs or patience.')
        if not math.isfinite(self.min_delta) or not 0 <= self.min_delta < 1:
            raise ValueError('Invalid min_delta.')
        if not math.isfinite(self.normal_fpr_limit) or not 0 <= self.normal_fpr_limit <= 1:
            raise ValueError('Invalid normal FPR limit.')

    def _better(self, candidate, current):
        # Inclusive +/- delta band; 1e-12 only protects its floating endpoint.
        # NLL tie replacement does not reset stale (progress_anchor is separate).
        return current is None or candidate['score'] > current['score'] + self.min_delta or (
            abs(candidate['score']-current['score']) <= self.min_delta + 1e-12 and candidate['nll'] < current['nll'])

    def update(self, epoch, metrics):
        score=metrics.get('positive_joint_map_accuracy')
        fpr=metrics.get('normal_joint_false_positive_rate')
        nll=metrics.get('joint_nll')
        if any(v is None or not math.isfinite(v) for v in (score,fpr,nll)):
            raise ValueError('Selection requires nonempty complete positive and normal validation subsets and finite NLL.')
        candidate=dict(epoch=epoch,score=score,nll=nll,normal_fpr=fpr)
        fallback_updated=self._better(candidate,self.fallback)
        if fallback_updated:self.fallback=candidate.copy()
        admissible=fpr <= self.normal_fpr_limit
        guarded_updated=admissible and self._better(candidate,self.guarded)
        significant=admissible and (self.progress_anchor is None or score > self.progress_anchor+self.min_delta)
        if significant:self.progress_anchor=score
        if guarded_updated:self.guarded=candidate.copy()
        if epoch <= self.min_epochs or significant:self.stale=0
        else:self.stale+=1
        return dict(guarded_updated=guarded_updated,fallback_updated=fallback_updated,
                    stop=epoch>self.min_epochs and self.stale>=self.patience,stale=self.stale,
                    guard_violated=self.guarded is None)

    def state_dict(self):
        return asdict(self)
