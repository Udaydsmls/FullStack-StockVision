import numpy as np

from stockvision.evaluation import compute_metrics


def test_perfect_prediction_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(y, y)
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.directional_accuracy == 1.0


def test_directional_accuracy_detects_inversion():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([4.0, 3.0, 2.0, 1.0])
    metrics = compute_metrics(y, pred)
    assert metrics.directional_accuracy == 0.0


def test_handles_zero_targets_in_mape():
    y = np.array([0.0, 1.0, 2.0])
    pred = np.array([0.1, 1.1, 1.9])
    metrics = compute_metrics(y, pred)
    assert metrics.mae > 0
    assert metrics.mape == metrics.mape  # not NaN
