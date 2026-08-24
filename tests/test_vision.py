import pytest

from vision_quality_gate.vision import evaluate, evaluate_thresholds, extract_features, make_dataset


def test_image_generation_is_deterministic():
    first = make_dataset(12, seed=3)
    second = make_dataset(12, seed=3)
    assert (first[0] == second[0]).all()
    assert (first[1] == second[1]).all()


def test_feature_shape_is_stable():
    images, _ = make_dataset(10, seed=2)
    features = extract_features(images)
    assert features.shape[0] == 10
    assert features.shape[1] == 40


def test_quality_gate_reports_coverage_and_abstention():
    metrics = evaluate(seed=3)
    assert 0 <= metrics["coverage"] <= 1
    assert 0 <= metrics["abstention_rate"] <= 1
    assert metrics["coverage"] + metrics["abstention_rate"] == 1


def test_threshold_sweep_exposes_tradeoff_rows():
    rows = evaluate_thresholds(seed=3, thresholds=(.50, .70))
    assert [row["confidence_threshold"] for row in rows] == [.50, .70]
    assert all(0 <= row["coverage"] <= 1 for row in rows)


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError):
        evaluate(seed=3, confidence_threshold=1.0)
