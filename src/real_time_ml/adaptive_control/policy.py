"""Permissive, explicitly adaptive-control-only policy over a pluggable model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import ControlProfile


@dataclass(frozen=True)
class ControlEstimate:
    relaxation: float
    discomfort: float
    model_variant: str
    active_modalities: list[str]
    raw: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None


class CandidateModel(Protocol):
    def predict_current(self, features: dict[str, float], condition: str, coverage: dict[str, float]) -> ControlEstimate: ...

    def predict_candidates(
        self,
        features: dict[str, float],
        current_condition: str,
        candidates: list[str],
        current: ControlEstimate,
    ) -> dict[str, ControlEstimate]: ...


@dataclass(frozen=True)
class ControlDecision:
    action: str
    current_condition: str
    target_condition: str
    estimate: ControlEstimate | None
    candidate_utilities: dict[str, float]
    utility_delta: float | None
    reasons: list[str]


class AdaptiveControlPolicy:
    def __init__(
        self,
        profile: ControlProfile,
        *,
        relaxation_weight: float,
        discomfort_weight: float,
        hysteresis: float,
        extreme_discomfort_limit: float,
    ) -> None:
        self.profile = profile
        self.relaxation_weight = relaxation_weight
        self.discomfort_weight = discomfort_weight
        self.hysteresis = hysteresis
        self.extreme_discomfort_limit = extreme_discomfort_limit

    def utility(self, estimate: ControlEstimate) -> float:
        return self.relaxation_weight * estimate.relaxation + self.discomfort_weight * estimate.discomfort

    def decide(
        self,
        model: CandidateModel,
        features: dict[str, float],
        current_condition: str,
        coverage: dict[str, float],
    ) -> ControlDecision:
        estimate = model.predict_current(features, current_condition, coverage)
        current_utility = self.utility(estimate)
        utilities = {current_condition: current_utility}
        if estimate.discomfort >= self.extreme_discomfort_limit:
            return ControlDecision(
                action="failsafe",
                current_condition=current_condition,
                target_condition=current_condition,
                estimate=estimate,
                candidate_utilities=utilities,
                utility_delta=None,
                reasons=["extreme_predicted_discomfort"],
            )
        candidates = self.profile.adjacent(current_condition)
        candidate_estimates = model.predict_candidates(features, current_condition, candidates, estimate)
        for condition, candidate_estimate in candidate_estimates.items():
            utilities[condition] = self.utility(candidate_estimate)
        target, target_utility = max(utilities.items(), key=lambda item: (item[1], item[0]))
        delta = target_utility - current_utility
        if target != current_condition and delta >= self.hysteresis:
            target_estimate = candidate_estimates.get(target)
            reason = "adaptive_utility_step"
            if target_estimate is not None:
                mode = str(target_estimate.raw.get("candidate_policy_mode", ""))
                if mode == "stable_probe":
                    reason = "stable_probe"
                elif mode == "calm_exploration":
                    reason = "calm_exploration"
                elif mode == "stress_recovery":
                    reason = "stress_recovery_step"
            return ControlDecision(
                action="apply",
                current_condition=current_condition,
                target_condition=target,
                estimate=estimate,
                candidate_utilities=utilities,
                utility_delta=delta,
                reasons=[reason],
            )
        return ControlDecision(
            action="hold",
            current_condition=current_condition,
            target_condition=current_condition,
            estimate=estimate,
            candidate_utilities=utilities,
            utility_delta=delta,
            reasons=["stable_no_better_neighbor"],
        )
