# Segmentation Metrics Toolkit

A dependency-light evaluator for binary masks. It reports foreground IoU, background
IoU, mean IoU, Dice, precision, recall, pixel accuracy, confusion counts, and a
color-coded error map.

## Evaluation rules

- mIoU is the arithmetic mean of foreground and background IoU.
- Zero-union classes are excluded from the mean.
- The reference mask is independent ground truth, never the model output.
- Masks are interpreted as white/foreground and black/background.

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.cli predicted.png reference.png --errors errors.png
~~~

The error map uses green for true positive, red for false positive, blue for false
negative, and black for true negative.

## Structure

- src/metrics.py — NumPy metric functions
- src/cli.py — PNG command-line interface
- tests/test_metrics.py — deterministic unit tests

## License

Apache-2.0
