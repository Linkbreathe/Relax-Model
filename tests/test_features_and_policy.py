from __future__ import annotations

import numpy as np

from real_time_ml.features.physio import StreamingPhysioProcessor, peak_f1
from real_time_ml.modeling.train import predict_state
from real_time_ml.policy.recommender import SafetyPolicy, adjacent_conditions


def synthetic_physio(sample_rate: float = 500.0) -> np.ndarray:
    time = np.arange(int(sample_rate * 10.0)) / sample_rate
    values = np.zeros((len(time), 9), dtype=float)
    values[:, 0] = np.arange(len(time))
    for index, frequency in zip(range(1, 5), (8, 10, 12, 15)):
        values[:, index] = 20 * np.sin(2 * np.pi * frequency * time)
    ecg = np.zeros(len(time))
    for second in np.arange(0.5, 10.0, 1.0):
        center = int(second * sample_rate)
        ecg[center : center + 5] = [0, 500, 1000, 500, 0]
    values[:, 7] = ecg
    return values


def processor() -> StreamingPhysioProcessor:
    return StreamingPhysioProcessor(
        sample_rate=500.0, eeg_columns=[1, 2, 3, 4], ecg_columns=[7, 8], counter_column=0,
        bands={"delta": [1, 4], "theta": [4, 8], "alpha": [8, 13], "beta": [13, 30], "gamma": [30, 45]},
    )


def test_offline_replay_and_realtime_window_features_are_identical():
    samples = synthetic_physio()
    offline, offline_qc = processor().process_window(samples)
    realtime, realtime_qc = processor().process_window(np.vstack([samples[:2500], samples[2500:]]))
    assert offline.keys() == realtime.keys()
    for name in offline:
        assert np.allclose(offline[name], realtime[name], equal_nan=True, rtol=1e-12, atol=1e-12)
    assert offline_qc == realtime_qc


def test_peak_f1_tolerance():
    reference = np.arange(1.0, 10.0)
    assert peak_f1(reference, reference + 0.05, 0.1) == 1.0
    assert peak_f1(reference, reference + 0.2, 0.1) == 0.0


def test_adjacent_policy_never_jumps_diagonally():
    assert set(adjacent_conditions("C5")) == {"C2", "C4", "C5", "C6", "C8"}
    assert set(adjacent_conditions("C1")) == {"C1", "C2", "C4"}


def test_policy_holds_when_model_is_experimental():
    policy = SafetyPolicy("1.0", [0.08, 0.16, 0.25], [0.12, 0.26, 0.41])
    result = policy.recommend(
        unix_time_ms=20_000, cycle_index=0, window_start_ms=10_000, current_condition="C5",
        state={"relaxation": 0.4, "discomfort": 0.2},
        intervals={name: [0.3, 0.6] for name in ("relaxation", "discomfort")},
        modality_coverage={"head": 1.0}, model_deployable=False,
        predict_candidate=lambda _: {"relaxation": 0.9, "discomfort": 0.0},
    )
    assert result.action == "hold"
    assert "state_model_not_better_than_baselines" in result.reasons


def test_prediction_coerces_empty_csv_values_to_missing():
    class Estimator:
        def predict(self, frame):
            assert frame["feature"].isna().all()
            return np.asarray([[0.7, 0.1]])

    result = predict_state(
        {"estimator": Estimator(), "feature_columns": ["feature"], "targets": ["relaxation", "discomfort"]},
        {"feature": ""},
    )
    assert result == {"relaxation": 0.7, "discomfort": 0.1}
