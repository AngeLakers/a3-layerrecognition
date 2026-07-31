"""Analytic head-capacity accounting for PCC and the matched-joint baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True)
class HeadCapacity:
    """Parameter-count decomposition for one structured prediction head."""

    responsibility_projector: int
    local_base: int
    structured_decoder: int
    total: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityMatch:
    """Label-free matched-joint width selected against one PCC reference."""

    joint_hidden_dim: int
    pcc: HeadCapacity
    matched_joint: HeadCapacity
    absolute_difference: int
    relative_difference: float
    tolerance_relative: float

    @property
    def within_tolerance(self) -> bool:
        return self.relative_difference <= self.tolerance_relative


def _positive_integer(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _linear_parameters(in_features: int, out_features: int) -> int:
    return in_features * out_features + out_features


def responsibility_projector_parameter_count(
    *,
    local_dim: int,
    context_dim: int,
    query_dim: int,
    token_dim: int,
) -> int:
    """Count seven bias-free source projections, three biases, and three LNs."""

    local_dim = _positive_integer("local_dim", local_dim)
    context_dim = _positive_integer("context_dim", context_dim)
    query_dim = _positive_integer("query_dim", query_dim)
    token_dim = _positive_integer("token_dim", token_dim)
    source_projection_weights = token_dim * (
        2 * local_dim + 2 * context_dim + 3 * query_dim
    )
    token_biases = 3 * token_dim
    layer_norm_affine = 6 * token_dim
    return source_projection_weights + token_biases + layer_norm_affine


def pcc_head_parameter_count(
    *,
    feature_dim: int,
    query_dim: int,
    token_dim: int,
    prefix_dim: int,
    hidden_dim: int,
    max_interfaces: int = 3,
) -> HeadCapacity:
    """Count the implemented projector plus :class:`PCCConditionalStep`."""

    feature_dim = _positive_integer("feature_dim", feature_dim)
    query_dim = _positive_integer("query_dim", query_dim)
    token_dim = _positive_integer("token_dim", token_dim)
    prefix_dim = _positive_integer("prefix_dim", prefix_dim)
    hidden_dim = _positive_integer("hidden_dim", hidden_dim)
    max_interfaces = _positive_integer("max_interfaces", max_interfaces)

    projector = responsibility_projector_parameter_count(
        local_dim=feature_dim,
        context_dim=feature_dim,
        query_dim=query_dim,
        token_dim=token_dim,
    )
    local_base = _linear_parameters(token_dim, 1)
    prefix_embedding = 2 * prefix_dim
    prefix_input_dim = max_interfaces * (query_dim + prefix_dim + 1)
    prefix_network = _linear_parameters(prefix_input_dim, hidden_dim) + _linear_parameters(
        hidden_dim, prefix_dim
    )
    correction = _linear_parameters(token_dim + prefix_dim, hidden_dim) + _linear_parameters(
        hidden_dim, 1
    )
    gate = _linear_parameters(token_dim, hidden_dim) + _linear_parameters(hidden_dim, 1)
    structured_decoder = prefix_embedding + prefix_network + correction + gate
    total = projector + local_base + structured_decoder
    return HeadCapacity(
        responsibility_projector=projector,
        local_base=local_base,
        structured_decoder=structured_decoder,
        total=total,
    )


def matched_joint_head_parameter_count(
    *,
    feature_dim: int,
    query_dim: int,
    token_dim: int,
    fusion_dim: int,
    joint_hidden_dim: int,
    supported_k: tuple[int, ...] = (2, 3),
) -> HeadCapacity:
    """Count the frozen matched-joint decoder without implementing M4 training."""

    feature_dim = _positive_integer("feature_dim", feature_dim)
    query_dim = _positive_integer("query_dim", query_dim)
    token_dim = _positive_integer("token_dim", token_dim)
    fusion_dim = _positive_integer("fusion_dim", fusion_dim)
    joint_hidden_dim = _positive_integer("joint_hidden_dim", joint_hidden_dim)
    if not supported_k or any(
        not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in supported_k
    ):
        raise ValueError("supported_k must contain positive integers.")
    if len(set(supported_k)) != len(supported_k):
        raise ValueError("supported_k must not contain duplicates.")

    projector = responsibility_projector_parameter_count(
        local_dim=feature_dim,
        context_dim=feature_dim,
        query_dim=query_dim,
        token_dim=token_dim,
    )
    local_base = _linear_parameters(token_dim, 1)
    fusion = _linear_parameters(3 * token_dim, fusion_dim)
    joint_branches = sum(
        _linear_parameters(k * fusion_dim, joint_hidden_dim)
        + _linear_parameters(joint_hidden_dim, 2**k)
        for k in supported_k
    )
    structured_decoder = fusion + joint_branches
    total = projector + local_base + structured_decoder
    return HeadCapacity(
        responsibility_projector=projector,
        local_base=local_base,
        structured_decoder=structured_decoder,
        total=total,
    )


def select_matched_joint_hidden_width(
    *,
    feature_dim: int,
    query_dim: int,
    token_dim: int,
    prefix_dim: int,
    pcc_hidden_dim: int,
    fusion_dim: int,
    minimum: int = 1,
    maximum: int = 512,
    tolerance_relative: float = 0.10,
    max_interfaces: int = 3,
    supported_k: tuple[int, ...] = (2, 3),
) -> CapacityMatch:
    """Choose width only by parameter distance; no labels or scores are inputs."""

    minimum = _positive_integer("minimum", minimum)
    maximum = _positive_integer("maximum", maximum)
    if minimum > maximum:
        raise ValueError("minimum must be <= maximum.")
    if not isinstance(tolerance_relative, (int, float)) or isinstance(
        tolerance_relative, bool
    ):
        raise ValueError("tolerance_relative must be numeric.")
    tolerance_relative = float(tolerance_relative)
    if not isfinite(tolerance_relative) or tolerance_relative < 0.0:
        raise ValueError("tolerance_relative must be finite and nonnegative.")

    pcc = pcc_head_parameter_count(
        feature_dim=feature_dim,
        query_dim=query_dim,
        token_dim=token_dim,
        prefix_dim=prefix_dim,
        hidden_dim=pcc_hidden_dim,
        max_interfaces=max_interfaces,
    )
    candidates: list[tuple[int, float, int, HeadCapacity]] = []
    for width in range(minimum, maximum + 1):
        matched = matched_joint_head_parameter_count(
            feature_dim=feature_dim,
            query_dim=query_dim,
            token_dim=token_dim,
            fusion_dim=fusion_dim,
            joint_hidden_dim=width,
            supported_k=supported_k,
        )
        absolute_difference = abs(matched.total - pcc.total)
        relative_difference = absolute_difference / pcc.total
        candidates.append((absolute_difference, relative_difference, width, matched))
    absolute_difference, relative_difference, width, matched = min(
        candidates,
        key=lambda item: (item[0], item[2]),
    )
    return CapacityMatch(
        joint_hidden_dim=width,
        pcc=pcc,
        matched_joint=matched,
        absolute_difference=absolute_difference,
        relative_difference=relative_difference,
        tolerance_relative=tolerance_relative,
    )


__all__ = [
    "CapacityMatch",
    "HeadCapacity",
    "matched_joint_head_parameter_count",
    "pcc_head_parameter_count",
    "responsibility_projector_parameter_count",
    "select_matched_joint_hidden_width",
]
