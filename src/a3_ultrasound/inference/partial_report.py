"""Minimum expected cost reporting, independently of training labels."""
from functools import lru_cache
from itertools import product
import math
import torch


@lru_cache(maxsize=12)
def _table(k, delay, exchange):
    if not 1 <= k <= 6:
        raise ValueError("Exact reporting exceeds the current enumerator implementation limit.")
    states = torch.tensor(list(product((0, 1), repeat=k)))
    actions = torch.tensor(sorted(product((0, 1, -1), repeat=k),
                                  key=lambda a: (a.count(-1), tuple(2 if x == -1 else x for x in a))))
    y, a = states[:, None], actions[None, :]
    fn = ((y == 1) & (a == 0)).sum(-1)
    fp = ((y == 0) & (a == 1)).sum(-1)
    pending = (actions == -1).sum(-1)
    cost = fn + fp + exchange * ((fn == fp) & (fn > 0)) + delay * pending
    return actions, cost.float(), pending.gt(0)


def partial_report(probabilities, k, *, point_fee, delay=.25, exchange=1., locked=None, frozen=None):
    """One batch of equal effective K; -1 means undecided, also denotes free slots in locked."""
    if not all(math.isfinite(x) and x >= 0 for x in (point_fee, delay, exchange)):
        raise ValueError("Report costs must be finite and nonnegative.")
    actions, cost, review = _table(k, delay, exchange)
    p = probabilities.float()
    if p.ndim != 2 or p.size(1) != 2**k or not torch.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected finite state probabilities in binary lexical order.")
    if not torch.allclose(p.sum(1), torch.ones_like(p[:, 0]), atol=1e-5):
        raise ValueError("State probabilities must sum to one.")
    actions = actions.to(p.device)
    risk = p @ cost.to(p.device) + point_fee * review.to(p.device)
    if locked is not None:
        if locked.shape != (len(p), k) or not ((locked >= -1) & (locked <= 1) & (locked == locked.long())).all():
            raise ValueError("Invalid locked action.")
        fixed = locked >= 0 if frozen is None else frozen
        if fixed.shape != locked.shape or fixed.dtype != torch.bool:
            raise ValueError("Invalid frozen-slot mask")
        incompatible = (fixed[:, None] & (locked[:, None] != actions)).any(-1)
        risk = risk.masked_fill(incompatible, float("inf"))
    idx = risk.argmin(1)
    return actions[idx], risk.gather(1, idx[:, None]).flatten()


def state_probabilities(logits, temperature=1.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    z = logits.float() / temperature
    mass = torch.ones_like(z[:, :1])
    for p in z.sigmoid().unbind(1):
        mass = torch.stack((mass * (1-p[:, None]), mass * p[:, None]), -1).flatten(1)
    return mass

