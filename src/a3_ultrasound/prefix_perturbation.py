"""Deterministic head-independent attempts; the substituted predictions are head-specific."""
from __future__ import annotations
import hashlib
import math
import torch
from a3_ultrasound.models.losses import validate_complete_training_batch
from a3_ultrasound.speed import fast_structure,cpu_batch,options


def replacement_mask(interface_mask, *, probability, seed, epoch, step):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('prefix replacement probability must be in [0, 1].')
    if epoch < 1 or step < 0:
        raise ValueError('Expected epoch >= 1 and step >= 0.')
    # Explicit stable tuple mapping; never consume global/dropout/shuffle RNG.
    key = f'a3-prefix-v1:{seed}:{epoch}:{step}'.encode('ascii')
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
    generator = torch.Generator(device='cpu').manual_seed(value)
    mask=cpu_batch().interface_mask if fast_structure() else interface_mask.cpu()
    return ((torch.rand(interface_mask.shape, generator=generator) < probability)&mask).to(interface_mask.device)


def training_conditional(model, batch, *, probability=0., seed=23, epoch=1, step=0):
    validate_complete_training_batch(batch)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('prefix replacement probability must be in [0, 1].')
    if model.head_type in ("state_conditioned_gru", "waveform_only_gru") and probability != 0:
        raise ValueError("State-conditioned GRU training requires true prefixes.")
    if probability == 0 or model.head_type not in ('vanilla_chain', 'proposed_v15'):
        return model(batch, batch.y).conditional, dict(attempted=0, changed=0, eligible=0,
                                                      attempt_mask_sha256=None)
    encoded = model.encode(batch)
    # Detach the prediction-producing Head pass, not the reusable encoder graph.
    with torch.no_grad():
        predicted = (model.conditional(encoded, batch.y).logits > 0).long()
    attempt = replacement_mask(batch.interface_mask, probability=probability,
                               seed=seed, epoch=epoch, step=step)
    mixed = torch.where(attempt, predicted, batch.y)
    output = model.conditional(encoded, mixed)
    # Count all valid slots to match the declared independent Bernoulli policy.
    # The last slot's replacement cannot affect an earlier conditional.
    if fast_structure() and options().coalesced_finite:
        host_attempt=replacement_mask(cpu_batch().interface_mask,probability=probability,seed=seed,epoch=epoch,step=step)
        stats=dict(attempted=int(host_attempt.sum()),changed=(attempt & (predicted!=batch.y)).sum(),
            eligible=int(cpu_batch().interface_mask.sum()),attempt_mask_sha256=hashlib.sha256(host_attempt.numpy().tobytes()).hexdigest())
    else:
        stats = dict(attempted=int(attempt.sum()), changed=int((attempt & (predicted != batch.y)).sum()),
                     eligible=int(batch.interface_mask.sum()),
                     attempt_mask_sha256=hashlib.sha256(attempt.cpu().numpy().tobytes()).hexdigest())
    return output, stats
