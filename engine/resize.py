"""
resize(): the engine's entry point, called by engine/main.py.

analyze() labels the source's elements, layout() arranges them on the
target canvas, finalize() enforces the hard rules (see README.md).
"""

import traceback
from typing import Any

from engine.analyze import analyze
from engine.finalize import finalize
from engine.layout import layout, scaled_copy


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """The source design ({"canvas", "elements"}) re-laid-out for
    target_canvas ({"width", "height"}): the same elements, each id once,
    with new positions, sizes and font sizes."""

    W, H = int(target_canvas["width"]), int(target_canvas["height"])
    try:
        placed = layout(analyze(source), W, H)
    except Exception:
        # Never crash on a weird design: fall back to fitting the whole
        # source uniformly inside the target (no distortion), centred.
        traceback.print_exc()
        placed = _uniform_fit(source, W, H)
    return {"canvas": {"width": W, "height": H},
            "elements": finalize(source, placed, W, H)}


def _uniform_fit(source: dict[str, Any], W: int, H: int) -> list[dict]:
    W0, H0 = source["canvas"]["width"], source["canvas"]["height"]
    s = min(W / W0, H / H0)
    dx, dy = (W - W0 * s) / 2, (H - H0 * s) / 2
    return [scaled_copy(e, 0, 0, s, dx, dy) for e in source["elements"]]
