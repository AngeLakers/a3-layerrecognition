"""Predeclared propagation-order controls."""

from __future__ import annotations

from itertools import permutations


def physical_and_control_orders(interface_count: int) -> tuple[tuple[int, ...], ...]:
    """Return all required orders, preserving the physical forward order first."""

    if interface_count not in (2, 3):
        raise ValueError("Order controls are only defined for K=2 or K=3.")
    forward = tuple(range(interface_count))
    remaining = tuple(order for order in permutations(forward) if order != forward)
    return (forward, *remaining)
