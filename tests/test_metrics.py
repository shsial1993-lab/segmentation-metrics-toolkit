import numpy as np

from src.metrics import evaluate


def test_binary_metrics() -> None:
    reference = np.array([[1, 1], [0, 0]])
    prediction = np.array([[1, 0], [1, 0]])
    metrics = evaluate(prediction, reference)
    assert metrics['foreground_iou'] == 1 / 3
    assert metrics['tp'] == 1
    assert metrics['fp'] == 1
    assert metrics['fn'] == 1
