import numpy as np

from stockvision.evaluation import compute_metrics


def test_perfect_prediction_has_zero_error():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["directional_accuracy"] == 1.0


def test_inverted_prediction_gets_every_direction_wrong():
    metrics = compute_metrics([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert metrics["directional_accuracy"] == 0.0


def test_zero_targets_do_not_break_mape():
    metrics = compute_metrics([0.0, 1.0, 2.0], [0.1, 1.1, 1.9])
    assert metrics["mae"] > 0
    assert not np.isnan(metrics["mape"])


def test_empty_input_returns_zeros():
    assert compute_metrics([], [])["n"] == 0
