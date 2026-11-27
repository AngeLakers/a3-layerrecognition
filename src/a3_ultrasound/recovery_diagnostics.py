"""Additional observations for restoring the F4 development baseline."""

from collections import defaultdict

import numpy as np


class BoardLayerDiagnostics:
    def __init__(self, board):
        self.board = board
        self.layers = defaultdict(list)
        self.missed_in_group = defaultdict(int)
        self.extra_in_group = defaultdict(int)

    def add(self, row):
        if row["specimen_id"] != self.board or row["y"] is None:
            return
        complete_abnormal = all(row["label_mask"]) and any(row["y"])
        for layer, (truth, known, logit, action) in enumerate(
            zip(row["y"], row["label_mask"], row["base_logodds"], row["map_states"]), 1
        ):
            if known:
                weight = row["sample_weight"]
                self.layers[layer].append((logit, truth, action, weight))
                if complete_abnormal and truth != action:
                    errors = self.missed_in_group if truth else self.extra_in_group
                    errors[layer] += weight

    def result(self):
        return {
            "board": self.board,
            "layers": {
                str(layer): layer_metrics(rows) | {
                    "full_abnormal_group_missed_weight": self.missed_in_group[layer],
                    "full_abnormal_group_extra_weight": self.extra_in_group[layer],
                }
                for layer, rows in self.layers.items()
            },
        }


def layer_metrics(rows):
    score, truth, action, weight = np.asarray(rows, dtype=np.float64).T
    positive = truth == 1
    _, inverse = np.unique(score, return_inverse=True)
    pos = np.bincount(inverse, weights=weight * positive)
    neg = np.bincount(inverse, weights=weight * ~positive)
    pairs = pos.sum() * neg.sum()
    result = {"weighted_auroc": float(pos @ (neg.cumsum() - .5 * neg) / pairs) if pairs else None}
    for name, mask in (("positive", positive), ("negative", ~positive)):
        values, weights = score[mask], weight[mask]
        count = float(weights.sum())
        group = {"count": int(mask.sum()), "weight": int(count), "reported_positive": None, "logit_quantiles": None}
        if count:
            order = np.argsort(values)
            cdf = weights[order].cumsum()
            group.update(
                reported_positive=float(weights @ action[mask] / count),
                logit_quantiles=values[order[np.searchsorted(cdf, count * np.array([.1, .5, .9]))]].tolist(),
            )
        result[name] = group
    return result


def write_recovery_checkpoint(output, filename, model, record, config_sha256):
    """Write the current weights before they can be replaced by the selected model."""
    import torch

    path = output / filename
    temp = path.with_suffix(".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": record["epoch"],
        "updates_total": record["updates_total"],
        "lr_epoch_start": record["lr_epoch_start"],
        "lr_last": record["lr_last"],
        "config_sha256": config_sha256,
        "validation": record["validation"],
    }, temp)
    temp.replace(path)
