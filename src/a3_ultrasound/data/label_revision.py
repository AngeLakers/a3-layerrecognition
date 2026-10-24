"""Design-interior admission at fixed, previously verified spatial registration."""
import numpy as np


def admit_design_interiors(reliable_y, reliable_mask, design_y, design_mask, quality):
    y = np.asarray(reliable_y)
    mask = np.asarray(reliable_mask, dtype=bool)
    design = np.asarray(design_y)
    known_design = np.asarray(design_mask, dtype=bool)
    quality = np.asarray(quality, dtype=bool)
    if y.ndim != 2 or any(a.shape != y.shape for a in (mask, design, known_design)) or quality.shape != (len(y),):
        raise ValueError('Inconsistent label arrays.')
    if not np.isin(y[mask], (0, 1)).all() or not np.isin(design[known_design], (0, 1)).all():
        raise ValueError('Known labels must be binary.')
    if np.any(mask & known_design & (y != design)):
        raise ValueError('Reliable labels conflict with the fixed design.')
    admitted = known_design & (design == 1) & quality[:, None] & ~mask
    revised_mask = (mask | admitted) & known_design
    revised_y = np.where(revised_mask, design, -1).astype(np.int8)
    if np.any(mask & ~revised_mask):
        raise ValueError('Existing source-reliable labels cannot be removed.')
    return revised_y, revised_mask, admitted
