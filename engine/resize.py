"""
THIS IS THE FILE YOU IMPLEMENT.

Everything else in this repo (input loading, output writing, rendering,
the Dockerfile) is plumbing so you can focus on exactly one function:
`resize()`. See RULES.md and the design schema for the full contract.
"""

import traceback
from typing import Any

from engine.analyze import analyze
from engine.finalize import finalize
from engine.layout import layout, scaled_copy


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    """
    Args:
        source: {"canvas": {"width": int, "height": int}, "elements": [...]}
                 — the design at its original size. See
                 chameleon_design_schema_v0.9.json for the full element spec
                 (text / image / shape, position, size, rotation, opacity,
                 z_index, group_id, and type-specific fields).
        target_canvas: {"width": int, "height": int} — the shape you're
                 resizing into.

    Returns:
        A design dict with the SAME shape as `source`:
        {"canvas": target_canvas, "elements": [...]}

        Hard requirements (see RULES.md for the full list):
        - Every element `id` in `source` must appear exactly once in your
          output. Do not add, remove, or invent elements.
        - Do not regenerate or redraw any asset's pixel content — you are
          repositioning and resizing the elements you were given, nothing
          more. This is a disqualifying rule, not a style preference.
        - Non-background image elements should be kept in proportion
          (matching the source aspect ratio within ~5%) unless they are a
          full-bleed / `fit: "cover"` element, which is expected to crop.
        - `font_size` may change to help text fit. `font_family` and
          `content` should not, for this stage of scoring (see RULES.md,
          "Adaptive typography and copy").
        - Nothing should end up outside `target_canvas`'s bounds.

    This is where your actual approach goes: constraint solving, a learned
    layout model, heuristics, whatever you choose. Latency is not a scoring
    concern (see RULES.md) — spend your effort on quality, not speed.
    """

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
