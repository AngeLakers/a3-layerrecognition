"""Regression checks for the authorized interior-only 2 mm exception."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from tc4_small_defect_labels import relax
from tc4_label_pool import classify


def test_small_circle_boundary_is_included_but_outside_buffer_is_unchanged():
    small = dict(front_x=20, front_y=20, size=2, interface=1, front_vertices=[])
    large = dict(front_x=40, front_y=40, size=4, interface=2, front_vertices=[])
    third = dict(front_x=60, front_y=60, size=4, interface=3, front_vertices=[])
    pts = np.array([[20, 20], [21, 20], [21.01, 20], [41.9, 40], [1, 1]])
    slots = [small, large, third]
    _, _, old, _, _ = classify(pts, slots, .8, 2)
    new, admitted, _ = relax(pts, slots, old, 2)
    assert new[1, 0] and admitted[1, 0]
    assert not new[2, 0]
    np.testing.assert_array_equal(new[:, 1:], old[:, 1:])
    assert not new[-1].any()


def test_small_square_includes_corners_without_expanding_its_outline():
    slot = dict(front_x=20, front_y=20, size=2, interface=1,
                front_vertices=[[19, 19], [19, 21], [21, 21], [21, 19]])
    pts = np.array([[21, 21], [21.001, 21], [20, 20]])
    new, admitted, _ = relax(pts, [slot], np.zeros((3, 3), dtype=bool), 2)
    np.testing.assert_array_equal(new[:, 0], [True, False, True])
    np.testing.assert_array_equal(new, admitted)
