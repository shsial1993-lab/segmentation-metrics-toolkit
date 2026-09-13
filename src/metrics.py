from __future__ import annotations

from typing import Any

import numpy as np


def _binary(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError('masks must be two-dimensional')
    return array > 0


def evaluate(prediction: Any, reference: Any, eps: float = 1e-12) -> dict[str, float | int]:
    pred = _binary(prediction)
    ref = _binary(reference)
    if pred.shape != ref.shape:
        raise ValueError('prediction and reference shapes must match')

    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    tn = int(np.logical_and(~pred, ~ref).sum())
    foreground_union = tp + fp + fn
    background_union = tn + fp + fn
    foreground_iou = (tp / foreground_union) if foreground_union else float('nan')
    background_iou = (tn / background_union) if background_union else float('nan')
    valid_ious = [value for value in (foreground_iou, background_iou) if not np.isnan(value)]
    return {
        'foreground_iou': float(foreground_iou),
        'background_iou': float(background_iou),
        'miou': float(np.mean(valid_ious)) if valid_ious else float('nan'),
        'dice': float((2 * tp) / max(2 * tp + fp + fn, eps)),
        'precision': float(tp / max(tp + fp, eps)),
        'recall': float(tp / max(tp + fn, eps)),
        'pixel_accuracy': float((tp + tn) / max(pred.size, 1)),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
    }


def error_map(prediction: Any, reference: Any) -> np.ndarray:
    pred = _binary(prediction)
    ref = _binary(reference)
    if pred.shape != ref.shape:
        raise ValueError('prediction and reference shapes must match')
    result = np.zeros((*pred.shape, 3), dtype=np.uint8)
    result[np.logical_and(pred, ref)] = (0, 255, 0)
    result[np.logical_and(pred, ~ref)] = (255, 0, 0)
    result[np.logical_and(~pred, ref)] = (0, 0, 255)
    return result
