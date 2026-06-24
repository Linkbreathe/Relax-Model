from __future__ import annotations


TARGETS = ["relaxation", "discomfort"]
CONTEXT_EXACT = {"intensity", "frequency", "intensity_index", "frequency_index", "condition_index", "presentation_position"}


def columns_for_group(columns: list[str], group: str) -> list[str]:
    eeg = [name for name in columns if name.startswith("eeg_")]
    behavior = [name for name in columns if name.startswith(("ecg_", "head_", "eye_"))]
    context = [name for name in columns if name.startswith("video_") or name in CONTEXT_EXACT]
    mapping = {
        "context_only": context,
        "no_eeg": behavior + context,
        "eeg_only": eeg,
        "user_all": eeg + behavior,
        "fused": eeg + behavior + context,
        "full": eeg + behavior + context,
        "behavior_only": [name for name in behavior if name.startswith(("head_", "eye_"))],
    }
    if group not in mapping:
        raise ValueError(f"Unknown feature group: {group}")
    return sorted(set(mapping[group]))
