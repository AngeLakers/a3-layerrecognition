import numpy as np
import pytest
from a3_ultrasound.data.label_revision import admit_design_interiors


def test_overlap_admitted_but_unknown_design_quality_and_outer_buffer_stay_unknown():
    y = np.array([[-1, -1], [0, -1], [-1, -1], [-1, -1], [1, 0]])
    mask = y >= 0
    design = np.array([[1, 1], [0, 0], [1, 1], [1, -1], [1, 0]])
    design_mask = design >= 0
    quality = np.array([True, True, False, True, True])
    result, known, admitted = admit_design_interiors(y, mask, design, design_mask, quality)
    np.testing.assert_array_equal(result, [[1, 1], [0, -1], [-1, -1], [1, -1], [1, 0]])
    assert admitted.sum() == 3
    assert np.all(result[~known] == -1)
    np.testing.assert_array_equal(result[mask], y[mask])


def test_conflicting_design_cannot_silently_rewrite_existing_ground_truth():
    with pytest.raises(ValueError, match='conflict'):
        admit_design_interiors(np.array([[0, 1]]), np.ones((1, 2), bool),
                               np.array([[1, 1]]), np.ones((1, 2), bool), np.ones(1, bool))
