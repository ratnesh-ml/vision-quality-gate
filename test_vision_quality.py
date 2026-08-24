from vision_quality import evaluate, extract_features, make_dataset


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
