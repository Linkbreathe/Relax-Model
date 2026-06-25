"""Adaptive-control-only configuration, deliberately separate from frozen research settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ADAPTIVE_CONTROL_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "adaptive-control.yaml"


@dataclass(frozen=True)
class AdaptiveControlSettings:
    source: Path
    profile_path: Path
    registry_path: Path
    default_bundle: str
    listen_host: str
    unity_to_python_port: int
    python_send_host: str
    python_to_unity_port: int
    sensor_hz: float
    status_hz: float
    command_expiry_ms: int
    command_timeout_seconds: float
    min_modality_coverage: float
    min_active_modalities: int
    consecutive_low_modality_windows: int
    transition_seconds: float
    initial_condition: str
    utility_relaxation_weight: float
    utility_discomfort_weight: float
    utility_hysteresis: float
    extreme_discomfort_limit: float

    def bundle_manifest(self, bundle_id: str | None = None) -> Path:
        selected = bundle_id or self.default_bundle
        candidate = self.registry_path / selected / "manifest.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Adaptive control model bundle manifest not found: {candidate}")
        return candidate


def _resolve(source: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (source.parent / candidate).resolve()


def load_adaptive_control_settings(path: str | Path | None = None) -> AdaptiveControlSettings:
    source = Path(path).resolve() if path else DEFAULT_ADAPTIVE_CONTROL_CONFIG.resolve()
    payload: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    network = payload.get("network", {})
    runtime = payload.get("runtime", {})
    policy = payload.get("policy", {})
    models = payload.get("models", {})
    settings = AdaptiveControlSettings(
        source=source,
        profile_path=_resolve(source, str(payload["profile_path"])),
        registry_path=_resolve(source, str(models["registry_path"])),
        default_bundle=str(models["default_bundle"]),
        listen_host=str(network.get("listen_host", "127.0.0.1")),
        unity_to_python_port=int(network.get("unity_to_python_port", 5055)),
        python_send_host=str(network.get("python_send_host", "127.0.0.1")),
        python_to_unity_port=int(network.get("python_to_unity_port", 5056)),
        sensor_hz=float(network.get("sensor_hz", 10.0)),
        status_hz=float(network.get("status_hz", 1.0)),
        command_expiry_ms=int(runtime.get("command_expiry_ms", 9500)),
        command_timeout_seconds=float(runtime.get("command_timeout_seconds", 25.0)),
        min_modality_coverage=float(policy.get("min_modality_coverage", 0.2)),
        min_active_modalities=int(policy.get("min_active_modalities", 2)),
        consecutive_low_modality_windows=int(policy.get("consecutive_low_modality_windows", 3)),
        transition_seconds=float(runtime.get("transition_seconds", 2.0)),
        initial_condition=str(runtime.get("initial_condition", "C5")),
        utility_relaxation_weight=float(policy.get("utility_relaxation_weight", 0.85)),
        utility_discomfort_weight=float(policy.get("utility_discomfort_weight", -0.15)),
        utility_hysteresis=float(policy.get("utility_hysteresis", 0.002)),
        extreme_discomfort_limit=float(policy.get("extreme_discomfort_limit", 0.95)),
    )
    if settings.sensor_hz <= 0 or settings.status_hz <= 0:
        raise ValueError("Adaptive control sensor_hz and status_hz must be positive")
    if settings.transition_seconds <= 0 or settings.command_expiry_ms <= 0:
        raise ValueError("Adaptive control transition_seconds and command_expiry_ms must be positive")
    if not 0 <= settings.min_modality_coverage <= 1:
        raise ValueError("Adaptive control min_modality_coverage must be in [0, 1]")
    return settings
