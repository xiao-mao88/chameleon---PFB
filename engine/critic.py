"""
What a finished layout gets wrong, for ranking candidate layouts (and for
tools/evaluate.py, which reports the same checks).

collisions(): text that newly overlaps something it shouldn't - another
text from a different group, or a visible non-background image / solid
shape. A pair that already overlapped in the source was a design choice
(text on its panel, a caption on its photo) and is not counted, and neither
is one where an opaque shape drawn between the two covers the text (text on
a panel that hides the stripes behind it), nor text lying wholly on a solid
shape (a headline set on the horizon band, as the Solstice designers do):
what reads as a collision is text cut by a shape's edge. Images are
measured by their visible pixels (engine.assets.ink_box): text over a
logo's transparent margin isn't touching the logo.

unreadable(): texts set below the readability floors (7px, or 9px for the
headline) - what the renderer will draw too small to read.
"""

import math

from engine.assets import ink_box


def drawn(e: dict) -> tuple[float, float, float, float]:
    """Box once drawn (rotated about its centre)."""
    x0, y0, w, h = e["x"], e["y"], e["width"], e["height"]
    r = math.radians(float(e.get("rotation") or 0.0))
    if r:
        c, s = abs(math.cos(r)), abs(math.sin(r))
        w, h = max(w, w * c + h * s), max(h, w * s + h * c)
        x0, y0 = e["x"] + (e["width"] - w) / 2, e["y"] + (e["height"] - h) / 2
    return x0, y0, x0 + w, y0 + h


def visible(e: dict) -> tuple[float, float, float, float]:
    """What is painted: an image's visible pixels, anything else's drawn box."""
    return ink_box(e) if e.get("type") == "image" and not e.get("rotation") else drawn(e)


def overlap(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = visible(a)
    bx0, by0, bx1, by1 = visible(b)
    return max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))


def _area(e: dict) -> float:
    return max(e["width"], 0.0) * max(e["height"], 0.0)


def collisions(source_elements: list[dict], elements: list[dict], W: float, H: float) -> list[str]:
    src = {e["id"]: e for e in source_elements}
    texts = [e for e in elements if e["type"] == "text" and e.get("content")]
    others = [e for e in elements if e["type"] != "text"
              and _area(e) < 0.9 * W * H                       # not a background
              and float(e.get("opacity", 1.0)) >= 0.4]          # not a faint decoration
    z = {e["id"]: (e.get("z_index", n), n) for n, e in enumerate(elements)}
    covers = [e for e in others if float(e.get("opacity", 1.0)) >= 0.8]

    def shielded(t, o):
        lo, hi = sorted((z[t["id"]], z[o["id"]]))
        tb = drawn(t)
        return any(lo < z[p["id"]] < hi and p is not o
                   and overlap(p, t) >= 0.95 * (tb[2] - tb[0]) * (tb[3] - tb[1])
                   for p in covers)

    out = []
    for i, a in enumerate(texts):
        for b in texts[i + 1:] + others:
            if a.get("group_id") and a.get("group_id") == b.get("group_id"):
                continue
            if a["id"] not in src or b["id"] not in src:
                continue
            if overlap(src[a["id"]], src[b["id"]]) > 0:
                continue
            if shielded(a, b):
                continue
            ov = overlap(a, b)
            if b["type"] == "shape" and ov >= 0.95 * _area(a):
                continue                  # text set wholly on the shape
            if ov > 4 and ov > 0.1 * min(_area(a), _area(b)):
                out.append(f"{a['id']} collides with {b['id']} ({ov:.0f}px²)")
    return out


MIN_READABLE = 7.0          # every text
MIN_HEADLINE = 9.0          # the source's first text at its largest size


def unreadable(source_elements: list[dict], elements: list[dict]) -> int:
    texts = [e for e in source_elements if e["type"] == "text" and e.get("content")]
    if not texts:
        return 0
    head = max(texts, key=lambda e: e["font_size"])["id"]
    return sum(e["font_size"] < (MIN_HEADLINE if e["id"] == head else MIN_READABLE) - 1e-6
               for e in elements if e["type"] == "text" and e.get("content"))
