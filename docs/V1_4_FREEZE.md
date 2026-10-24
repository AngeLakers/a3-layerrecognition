# V1.4 frozen execution contract

The release tag is `v1.4-freeze`; the executable template is
[`configs/experiment/v1_4_freeze.json`](../configs/experiment/v1_4_freeze.json).
The template is closed to real training and contains path/commit placeholders.
An operator prepares a separate runtime config after resolving the release commit.
Source code and training rules must not be adjusted in response to run scores.

Only F1/seed23 Proposed rho3, Vanilla Chain and Local Independent are registered.
Every run initializes a fresh GN8/position2 model; historical rho1, Shadow and
D1/D2 cache entrypoints remain available for provenance, outside this run list.
No old weights are accepted. Outer evaluation is rejected even if an operator
changes its permission flag in a copy of this config.

B8 uses max40/min15/patience10/min_delta=.002. The existing guarded selector
maintains a separate progress anchor, with no stale count through epoch15 and
no patience reset for an NLL-only tie replacement. Admissible epochs have IPW
normal complete-joint FPR <=.02. `best_guarded.pt` and `best_fallback.pt` are
maintained independently; if no epoch qualifies, no guarded weight is invented
and the fallback result is marked diagnostic-only. Saturation, correction
direction and the former 1% alarm do not screen out these runs. Numerical/OOM
failures and the existing operator/ownership/watchdog/disk/timeout protections
remain failures; B8 is the planned early-stopping rule.
The frozen full-data job has a 24-hour operational wall limit, while historical
short diagnostics retain 8 hours; this avoids using the short-job limit to
truncate the declared forty-epoch budget. The 20-minute no-progress guard remains.

B5 uses every reliable abnormal training sample and resamples 3x normals per
epoch. The forty frozen files contain the final order, with no second shuffle;
all three Heads use exactly the same files and SHA256 values. The validation
panel is fixed. The training runner saves epoch order, optimizer and RNG states;
automatic recovery/retraining is not supported.

All Heads record IPW and raw numerator/denominator diagnostics:

- `multiple_defect_missing_any_rate`: among complete-label truths with >=2
  abnormal interfaces, at least one true abnormal interface is MAP-predicted normal.
- `multiple_defect_joint_error_rate`: any complete-vector MAP error in that
  subset, including extra positives; a secondary diagnostic.
- `single_defect_to_multiple_rate`: among complete-label single-defect truths,
  MAP predicts more than one defect.
- `downstream_abnormal_recall_given_upstream_abnormal`: current reliable truth
  is abnormal and at least one reliable upstream truth is abnormal; denominator
  is interface positions, weighted by their sample's IPW. Unknown truth is never
  filled with normal. These diagnostics never enter selection or early stopping.

Only Proposed writes `best_guarded_diagnostics.npz` and
`best_fallback_diagnostics.npz` when the corresponding checkpoint updates.
Each row is a valid non-first interface on the inferred joint-MAP path, with
`g`, `c`, `gc`, sample/specimen identity, zero-based interface index, K, IPW,
current truth/MAP, full upstream truths, valid/reliable masks and upstream group.
Unknown/padded truths are -1. Group 1 means any known upstream abnormal;
group 0 requires all upstream truths known normal; -1 is undetermined.
The Unicode metadata binds epoch, checkpoint name, commit and config SHA256.
Load with `numpy.load(path, allow_pickle=False)`; no object arrays are used.

Every `summary.json` contains `commit_hash` and `config_sha256` of the actual
runtime config bytes, along with both best records and the stopping reason.
In Git checkouts the commit is read from Git. An immutable source export carries
`SOURCE_REVISION.json`, generated from the release tag, and its supervisor
verifies a file manifest before starting a job. The public template cannot start
without explicit runtime permission and a matching source commit.

Synthetic unit tests validate the implementation; they are not research results.
Subgroup recall differences are diagnostic and alone do not prove a causal
claim that no true defects were suppressed.
