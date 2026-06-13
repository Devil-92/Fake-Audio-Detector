import numpy as np

from audio_detection.metrics import compute_metrics, equal_error_rate


def test_equal_error_rate_is_bounded():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    eer = equal_error_rate(y_true, y_score)

    assert 0.0 <= eer <= 1.0


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_score)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["per_class_accuracy"]["Genuine"] == 1.0
    assert metrics["per_class_accuracy"]["Deepfake"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
