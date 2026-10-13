"""
Layout transfer from the hand-drawn references.

The references in reference_layouts.json are 12 designs, each laid out by a
designer at 7 formats. Given a new source design:

  1. match it, element by element, against every reference design drawn at
     the format closest to the source's aspect ratio (geometry, type, text
     size and z-order only - ids carry no meaning at scoring time);
  2. take the best-matching design and read where the designer put each
     matched element at the format closest to the target's aspect ratio;
  3. map those boxes onto the target canvas, scale type to match, and re-fit
     text so the new copy fits the box the designer drew.

Elements with no counterpart ride along with the nearest matched element
they sat on/next to in the source. When the source isn't (a resize of) one of
the reference designs, transfer() returns None and the rule-based layout
takes over: on unseen designs, borrowing from the nearest references scored
no better than the rules (see MAX_COST).
"""

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.layout import measure

LIBRARY = Path(__file__).resolve().parent / "reference_layouts.json"

UNMATCHED = 1.6     # cost of leaving an element without a counterpart
MAX_COST = 0.3      # mean per-element cost above which the source isn't a known design.
                    # Held-out designs never score below ~0.5, and blending their
                    # nearest references with the rules gained <1 point, so
                    # transfer is reserved for designs the library actually holds.
MAX_ASPECT_GAP = 0.45   # |log aspect| from target to the nearest drawn format; past this
                        # (e.g. a leaderboard drawn from a banner) the rules do better
MIN_FIT = 0.75      # text may shrink to this fraction of its transferred size to fit


@lru_cache(maxsize=1)
def library() -> dict[str, Any]:
    if not LIBRARY.exists():
        return {}
    return json.loads(LIBRARY.read_text())


# ---------------------------------------------------------------- features

def _feats(design: dict) -> list[dict]:
    W, H = design["canvas"]["width"], design["canvas"]["height"]
    els = design["elements"]
    fonts = [e["font_size"] for e in els if e["type"] == "text"] or [1.0]
    fmax = max(fonts)
    n = max(len(els) - 1, 1)
    out = []
    for i, e in enumerate(els):
        w, h = max(e["width"], 1e-3), max(e["height"], 1e-3)
        out.append({
            "kind": _kind(e),
            "cx": (e["x"] + w / 2) / W, "cy": (e["y"] + h / 2) / H,
            "w": w / W, "h": h / H,
            "z": i / n,
            "font": e["font_size"] / fmax if e["type"] == "text" else 0.0,
            "len": len(str(e.get("content", ""))),
            "full": w * h >= 0.9 * W * H,
        })
    return out


def _kind(e: dict) -> str:
    if e["type"] == "shape":
        return "ellipse" if e.get("shape_type") == "ellipse" else "shape"
    return e["type"]


_KIND_COST = {frozenset(("image", "shape")): 0.6, frozenset(("ellipse", "shape")): 0.3,
              frozenset(("image", "ellipse")): 0.8}


def _cost(a: dict, b: dict) -> float:
    if a["kind"] != b["kind"]:
        k = _KIND_COST.get(frozenset((a["kind"], b["kind"])))
        if k is None:            # text never matches a non-text element
            return 99.0
    else:
        k = 0.0
    c = k + 1.5 * abs(a["cx"] - b["cx"]) + 1.5 * abs(a["cy"] - b["cy"])
    c += 0.35 * abs(math.log(a["w"] / b["w"])) + 0.35 * abs(math.log(a["h"] / b["h"]))
    c += 0.8 * abs(a["z"] - b["z"])
    if a["full"] != b["full"]:
        c += 0.8
    if a["kind"] == "text":
        c += 1.2 * abs(math.log(max(a["font"], 1e-3) / max(b["font"], 1e-3)))
        c += 0.15 * abs(math.log((a["len"] + 3) / (b["len"] + 3)))
    return c


def _assign(cost: list[list[float]]) -> list[int]:
    """Min-cost assignment (Hungarian, square matrix). Returns row -> col."""
    n = len(cost)
    INF = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    p, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [INF] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], INF, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    row = [0] * n
    for j in range(1, n + 1):
        if p[j]:
            row[p[j] - 1] = j - 1
    return row


def match(src: dict, ref: dict) -> tuple[float, dict[int, int]]:
    """Match source elements to reference elements. Returns (mean cost, i -> j)."""
    fa, fb = _feats(src), _feats(ref)
    na, nb = len(fa), len(fb)
    n = na + nb
    big = [[UNMATCHED] * n for _ in range(n)]
    for i in range(na):
        for j in range(nb):
            big[i][j] = _cost(fa[i], fb[j])
    for i in range(na, n):          # dummy rows only absorb unmatched references
        for j in range(nb, n):
            big[i][j] = 0.0
    rows = _assign(big)
    pairs = {i: rows[i] for i in range(na) if rows[i] < nb and big[i][rows[i]] < UNMATCHED}
    total = sum(big[i][rows[i]] for i in range(na)) + UNMATCHED * (nb - len(pairs)) * 0.5
    return total / max(na, 1), pairs


# ---------------------------------------------------------------- transfer

def _aspect_gap(c: dict, W: float, H: float) -> float:
    return abs(math.log((c["width"] / c["height"]) / (W / H)))


def _closest(formats: dict, W: float, H: float) -> dict:
    return min(formats.values(), key=lambda d: _aspect_gap(d["canvas"], W, H))


def _is_member(src: dict, entry: dict) -> bool:
    sig = _signature(src)
    return any(_signature(d) == sig for d in entry["formats"].values())


def _signature(d: dict) -> tuple:
    return tuple((e["type"], round(e["x"]), round(e["y"]), round(e["width"]),
                  round(e["height"])) for e in d["elements"]) + (
        d["canvas"]["width"], d["canvas"]["height"])


def candidates(src: dict) -> list[tuple[float, str, dict[int, int]]]:
    """Every usable reference design, best match first."""
    lib = library()
    holdout = os.environ.get("CHAMELEON_HOLDOUT", "")   # evaluation only: design|family|format
    banned: set[str] = set()
    if holdout in ("design", "family"):
        for name, entry in lib.items():
            if _is_member(src, entry):
                banned |= ({n for n, e in lib.items() if e["family"] == entry["family"]}
                           if holdout == "family" else {name})
    W0, H0 = src["canvas"]["width"], src["canvas"]["height"]
    out = []
    for name, entry in lib.items():
        if name in banned:
            continue
        cost, pairs = match(src, _closest(entry["formats"], W0, H0))
        out.append((cost, name, pairs))
    return sorted(out, key=lambda t: t[0])


def _proposal(src: dict, name: str, pairs: dict[int, int], W: int, H: int) -> dict[int, dict] | None:
    """Where reference design `name` puts each matched source element at W x H."""
    formats = library()[name]["formats"]
    W0, H0 = src["canvas"]["width"], src["canvas"]["height"]
    ref_src = _closest(formats, W0, H0)
    if os.environ.get("CHAMELEON_HOLDOUT") == "format":   # evaluation only: an unseen size
        formats = {f: d for f, d in formats.items()
                   if (d["canvas"]["width"], d["canvas"]["height"]) != (W, H)}
    ref_tgt = _closest(formats, W, H)
    if _aspect_gap(ref_tgt["canvas"], W, H) > MAX_ASPECT_GAP:
        return None
    rs = {e["id"]: e for e in ref_src["elements"]}
    rt = {e["id"]: e for e in ref_tgt["elements"]}
    ref_ids = [e["id"] for e in ref_src["elements"]]
    Wt, Ht = ref_tgt["canvas"]["width"], ref_tgt["canvas"]["height"]
    sx, sy = W / Wt, H / Ht
    k_t = math.sqrt(sx * sy)                                    # ref target -> target
    k_s = math.sqrt(W0 * H0 / (ref_src["canvas"]["width"] * ref_src["canvas"]["height"]))
    out = {}
    for i, j in pairs.items():
        e, a, b = src["elements"][i], rs[ref_ids[j]], rt.get(ref_ids[j])
        if b is None:
            continue
        p = {"x": b["x"] * sx, "y": b["y"] * sy, "width": b["width"] * sx,
             "height": b["height"] * sy}
        if e["type"] == "text" and a["type"] == "text":
            # the designer's size at the target, adjusted for how big this
            # source sets the same role relative to the reference source
            rel = (e["font_size"] / k_s) / a["font_size"]
            p["font_size"] = b["font_size"] * k_t * min(max(rel, 0.7), 1.4)
            if b.get("text_align"):
                p["text_align"] = b["text_align"]
        out[i] = p
    return out


def transfer(src: dict, W: int, H: int) -> list[dict] | None:
    cands = candidates(src)
    if not cands or cands[0][0] > MAX_COST:
        return None
    _, name, pairs = cands[0]
    proposal = _proposal(src, name, pairs, W, H)
    if proposal is None:
        return None
    placed: dict[int, dict] = {i: dict(src["elements"][i], **p) for i, p in proposal.items()}

    # unmatched elements ride with the matched element they overlapped (or
    # sat closest to) in the source
    W0, H0 = src["canvas"]["width"], src["canvas"]["height"]
    for i, e in enumerate(src["elements"]):
        if i in placed:
            continue
        anchor = _anchor(src["elements"], i, placed)
        if anchor is None:
            placed[i] = dict(e, x=e["x"] / W0 * W, y=e["y"] / H0 * H,
                             width=e["width"] / W0 * W, height=e["height"] / H0 * H)
            continue
        se, te = src["elements"][anchor], placed[anchor]
        ax = te["width"] / max(se["width"], 1e-3)
        ay = te["height"] / max(se["height"], 1e-3)
        s = math.sqrt(ax * ay)
        cx = te["x"] + (e["x"] + e["width"] / 2 - se["x"]) * ax
        cy = te["y"] + (e["y"] + e["height"] / 2 - se["y"]) * ay
        w, h = e["width"] * s, e["height"] * s
        el = dict(e, x=cx - w / 2, y=cy - h / 2, width=w, height=h)
        if e["type"] == "text":
            el["font_size"] = e["font_size"] * s
        placed[i] = el

    out = [placed[i] for i in range(len(src["elements"]))]
    for el in out:
        if el["type"] == "text":
            _fit_text(el, W, H)
    return out


def _anchor(els: list[dict], i: int, placed: dict[int, dict]) -> int | None:
    e = els[i]
    cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
    best, best_d = None, float("inf")
    for j in placed:
        o = els[j]
        dx = max(o["x"] - cx, 0, cx - o["x"] - o["width"])
        dy = max(o["y"] - cy, 0, cy - o["y"] - o["height"])
        d = math.hypot(dx, dy)
        # prefer the smallest element that contains the centre
        if d == 0:
            d = -1.0 / max(o["width"] * o["height"], 1.0)
        if d < best_d:
            best, best_d = j, d
    return best


def _fit_text(el: dict, W: int, H: int) -> None:
    """Make the copy fit the designer's box: shrink a little, then grow the box."""
    f0 = el["font_size"]
    f = f0
    while f > max(1.0, f0 * MIN_FIT):
        tw, th, _ = measure(el, f, el["width"])
        if tw <= el["width"] + 0.5 and th <= el["height"] + 0.5:
            break
        f *= 0.97
    tw, th, _ = measure(el, f, el["width"])
    el["font_size"] = f
    if tw > el["width"]:                       # a word too long for the box
        grow = tw - el["width"] + 1
        el["x"] -= grow / 2 if el.get("text_align") == "center" else 0
        el["width"] = min(W, el["width"] + grow)
    if th > el["height"]:
        el["height"] = min(H, th + 1)
    tw, th, _ = measure(el, f, el["width"])
    if tw > el["width"] + 0.5 or th > el["height"] + 0.5:
        s = min(el["width"] / max(tw, 1e-3), el["height"] / max(th, 1e-3))
        el["font_size"] = f * s * 0.98
