"""
Last line of defence: whatever layout produced, make the output obey the
hard rules — every source id exactly once, original order and z_index,
nothing off-canvas, images and circles in proportion, sane font sizes.
"""

import math
from typing import Any

from engine.analyze import rotated_size


def extent(el: dict) -> tuple[float, float]:
    """(width, height) the element covers on canvas, rotation included."""
    return rotated_size(el["width"], el["height"], el.get("rotation"))


def _keeps_aspect(src: dict) -> bool:
    if src["type"] == "image":
        return src.get("fit") != "cover"
    return src["type"] == "shape" and src.get("shape_type") == "ellipse"


def finalize(source: dict[str, Any], placed: list[dict], W: int, H: int) -> list[dict]:
    by_id = {}
    for e in placed:
        by_id.setdefault(e["id"], e)
    out = []
    for src in source["elements"]:
        el = dict(by_id.get(src["id"], src))
        # fields the engine must never change
        for k in ("id", "type", "content", "font_family", "asset_id", "z_index",
                  "group_id", "shape_type", "fit"):
            if k in src:
                el[k] = src[k]
        for k in ("x", "y", "width", "height"):
            v = el.get(k)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                el[k] = 0.0
        el["width"], el["height"] = max(0.0, el["width"]), max(0.0, el["height"])

        if _keeps_aspect(src) and src["height"] > 0 and src["width"] > 0 and el["height"] > 0:
            ratio = src["width"] / src["height"]
            if abs((el["width"] / el["height"]) / ratio - 1) > 0.01:
                cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
                if el["width"] / el["height"] > ratio:
                    el["width"] = el["height"] * ratio
                else:
                    el["height"] = el["width"] / ratio
                el["x"], el["y"] = cx - el["width"] / 2, cy - el["height"] / 2

        # Too big for the canvas (rotated bounds included): shrink about the
        # centre (uniformly when the element must keep its shape, and scale
        # text with it).
        ew, eh = extent(el)
        if ew > W or eh > H:
            s = min(W / max(ew, 1e-9), H / max(eh, 1e-9))
            cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
            if _keeps_aspect(src) or el["type"] == "text":
                el["width"] *= s
                el["height"] *= s
                if el["type"] == "text":
                    el["font_size"] = el["font_size"] * s
            elif el.get("rotation"):
                el["width"], el["height"] = el["width"] * s, el["height"] * s
            else:
                el["width"], el["height"] = min(el["width"], W), min(el["height"], H)
            el["x"], el["y"] = cx - el["width"] / 2, cy - el["height"] / 2

        # keep the centre where box and rotated bounds both stay on canvas
        ew, eh = extent(el)
        cx = min(max(el["x"] + el["width"] / 2, ew / 2), W - ew / 2)
        cy = min(max(el["y"] + el["height"] / 2, eh / 2), H - eh / 2)
        el["x"], el["y"] = cx - el["width"] / 2, cy - el["height"] / 2
        el["x"] = min(max(el["x"], 0.0), W - el["width"])
        el["y"] = min(max(el["y"], 0.0), H - el["height"])

        if el["type"] == "text":
            el["font_size"] = round(max(1.0, float(el.get("font_size", src["font_size"]))), 2)
        for k in ("x", "y", "width", "height"):
            el[k] = round(el[k], 2)
        # rounding must not push anything over the edge
        el["width"] = min(el["width"], W - el["x"])
        el["height"] = min(el["height"], H - el["y"])
        if "corner_radius" in el:
            el["corner_radius"] = round(min(el["corner_radius"],
                                            0.5 * min(el["width"], el["height"])), 2)
        out.append(el)
    return out
