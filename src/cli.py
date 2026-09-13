from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from .metrics import error_map, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate two binary PNG masks.')
    parser.add_argument('prediction')
    parser.add_argument('reference')
    parser.add_argument('--errors', default='error_map.png')
    args = parser.parse_args()
    prediction = np.asarray(Image.open(args.prediction).convert('L'))
    reference = np.asarray(Image.open(args.reference).convert('L'))
    print(json.dumps(evaluate(prediction, reference), indent=2, allow_nan=True))
    Image.fromarray(error_map(prediction, reference)).save(args.errors)


if __name__ == '__main__':
    main()
