"""
Work out what each element *is* from geometry and style alone (ids carry no
meaning at scoring time).

Every source element lands in exactly one of these buckets:

  background  covers (almost) the whole canvas -> stretched full-bleed
  edge        a non-background shape touching canvas edges (horizon band,
              scrim, vertical stripes) -> stays glued to those edges
  deco        faint shapes (opacity < 0.4) -> free-floating, re-positioned
              proportionally; if they sit on the hero visual they ride with it
  visual      the hero cluster: big images/shapes plus everything rigid that
              overlaps them (badges, captions, rings) -> scaled as one rigid
              block so intentional layering survives
  container   a big shape holding several copy items (a text panel) ->
              re-wrapped around wherever that copy ends up
  corner      small rigid items hugging a canvas corner (sale badge, ribbon,
              watermark) -> pinned to the same corner
  flow        everything else, in reading order: logo, eyebrow, headline,
              body, CTA… -> re-flowed into columns
"""

import math
from dataclasses import dataclass, field
from typing import Any

Box = tuple[float, float, float, float]  # x0, y0, x1, y1

DECO_OPACITY = 0.4
SEED_AREA = 0.02        # min fraction of canvas for an element to seed the hero visual
CLUSTER_PAD = 0.02      # boxes inflated by this * long side when clustering
CORNER_DIST = 0.09      # corner items sit within this fraction of two edges
CORNER_MAX_AREA = 0.06
EDGE_TOL = 0.01


def box_of(el: dict) -> Box:
    return el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"]


def rotated_size(w: float, h: float, rotation) -> tuple[float, float]:
    """Size a w x h box covers once rotated about its centre (the renderer
    rotates that way): the larger of the box and its rotated bounds."""
    r = math.radians(float(rotation or 0.0))
    if not r:
        return w, h
    c, s = abs(math.cos(r)), abs(math.sin(r))
    return max(w, w * c + h * s), max(h, w * s + h * c)


def headline_id(elements: list[dict]) -> str | None:
    """The headline: the first non-empty text at the largest font size."""
    texts = [e for e in elements if e["type"] == "text" and e.get("content")]
    return max(texts, key=lambda e: e["font_size"])["id"] if texts else None


def drawn_box(el: dict) -> Box:
    """What the element covers once drawn: its box, grown to the rotated box's
    bounds when it is rotated."""
    x0, y0, x1, y1 = box_of(el)
    if not el.get("rotation"):
        return x0, y0, x1, y1
    rw, rh = rotated_size(x1 - x0, y1 - y0, el["rotation"])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2


def union(boxes) -> Box:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlaps(a: Box, b: Box, pad: float = 0.0) -> bool:
    return (a[0] - pad < b[2] and b[0] - pad < a[2]
            and a[1] - pad < b[3] and b[1] - pad < a[3])


def contains(outer: Box, inner: Box, tol: float = 1.0) -> bool:
    return (inner[0] >= outer[0] - tol and inner[1] >= outer[1] - tol
            and inner[2] <= outer[2] + tol and inner[3] <= outer[3] + tol)


@dataclass
class Unit:
    """One or more elements that move together."""
    els: list[dict]
    reflow: bool = False        # a lone text element: font/width re-flowed
    row: int | None = None      # units sharing a row id sat side by side (badge row)
    primary: bool = False       # holds the headline (first text at the largest size)
    box: Box = field(init=False)

    def __post_init__(self):
        self.box = union(drawn_box(e) for e in self.els)

    @property
    def w(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def has_text(self) -> bool:
        return any(e["type"] == "text" for e in self.els)

    @property
    def is_labelled_shape(self) -> bool:
        """One shape with one text sitting inside it (pill, badge, tag) -
        re-wrappable around its label, whatever the label's length."""
        shapes = [e for e in self.els if e["type"] == "shape"]
        texts = [e for e in self.els if e["type"] == "text"]
        return (len(self.els) == 2 and len(shapes) == 1 and len(texts) == 1
                and shapes[0].get("shape_type") != "ellipse"
                and contains(box_of(shapes[0]), box_of(texts[0]),
                             tol=0.25 * texts[0]["height"]))

    @property
    def is_button(self) -> bool:
        """A pill/box shape with a single short label on it (CTA-like)."""
        shapes = [e for e in self.els if e["type"] == "shape"]
        texts = [e for e in self.els if e["type"] == "text"]
        return (len(self.els) == 2 and len(shapes) == 1 and len(texts) == 1
                and shapes[0]["width"] >= 1.5 * shapes[0]["height"]
                and len(texts[0].get("content", "")) <= 32)


@dataclass
class Analysis:
    canvas: tuple[int, int]
    background: list[dict]
    edge: list[dict]
    deco: list[dict]             # free decorations
    visual: Unit | None          # hero cluster (includes attached decos)
    visual_parts: list[Unit]     # text-bearing units the hero pulled in (badges, captions)
    container: Unit | None
    container_children: set[str]
    corner: list[tuple[Unit, str]]   # (unit, "tl"|"tr"|"bl"|"br")
    flow: list[Unit]             # reading order
    centered: bool               # copy was centred on the canvas
    max_font: float              # largest source font size
    elements: list[dict] = field(default_factory=list)   # the source elements, as given


def _opacity(el: dict) -> float:
    return float(el.get("opacity", 1.0))


def _touching(el: dict, W: float, H: float) -> set[str]:
    x0, y0, x1, y1 = box_of(el)
    tx, ty = EDGE_TOL * W, EDGE_TOL * H
    s = set()
    if x0 <= tx: s.add("l")
    if x1 >= W - tx: s.add("r")
    if y0 <= ty: s.add("t")
    if y1 >= H - ty: s.add("b")
    return s


def analyze(source: dict[str, Any]) -> Analysis:
    W, H = source["canvas"]["width"], source["canvas"]["height"]
    short = min(W, H)
    els = list(source["elements"])

    background, rest = [], []
    for e in els:
        a = area(box_of(e))
        clipped = area((max(0, e["x"]), max(0, e["y"]),
                        min(W, e["x"] + e["width"]), min(H, e["y"] + e["height"])))
        if e["type"] != "text" and clipped >= 0.9 * W * H and a <= 1.6 * W * H:
            background.append(e)
        else:
            rest.append(e)

    # Groups first: elements sharing group_id always move together.
    units: list[Unit] = []
    by_group: dict[str, list[dict]] = {}
    for e in rest:
        if e.get("group_id"):
            by_group.setdefault(e["group_id"], []).append(e)
    seen = set()
    for e in rest:
        g = e.get("group_id")
        if g:
            if g not in seen:
                seen.add(g)
                units.append(Unit(by_group[g]))
        else:
            units.append(Unit([e], reflow=e["type"] == "text"))

    # A lone text sitting inside a small ungrouped shape is that shape's label.
    lone_text = [u for u in units if u.reflow]
    for u in list(units):
        if len(u.els) != 1 or u.els[0]["type"] != "shape" or _opacity(u.els[0]) < DECO_OPACITY:
            continue
        inside = [t for t in lone_text if t in units and contains(u.box, t.box)]
        if len(inside) == 1 and area(u.box) < 0.05 * W * H:
            t = inside[0]
            units.remove(t)
            units.remove(u)
            units.append(Unit(u.els + t.els))

    # Container: a big lone shape fully holding 2+ copy items (a text
    # panel). Checked before edges: a panel may touch one edge, but a band
    # spanning the full width/height (scrim, horizon) stays an edge.
    def lone_shape(u):
        return len(u.els) == 1 and u.els[0]["type"] == "shape"

    container, children = None, []
    for u in sorted(units, key=lambda u: -area(u.box)):
        if not lone_shape(u) or area(u.box) < 0.05 * W * H:
            continue
        touch = _touching(u.els[0], W, H)
        if {"l", "r"} <= touch or {"t", "b"} <= touch:
            continue
        inner = [v for v in units if v is not u and contains(u.box, v.box)]
        if len(inner) >= 2 and sum(v.has_text for v in inner) >= 2:
            container, children = u, inner
            break

    edge, deco, others = [], [], []
    for u in units:
        if u is container:
            continue
        if lone_shape(u) and u not in children:
            e = u.els[0]
            if _touching(e, W, H) and area(u.box) >= 0.01 * W * H:
                edge.append(e)
                continue
            if _opacity(e) < DECO_OPACITY:
                deco.append(e)
                continue
        others.append(u)

    # Hero visual: connected component of big non-text items (+ anything
    # rigid that overlaps it), outside the container.
    pad = CLUSTER_PAD * max(W, H)
    candidates = [u for u in others if u not in children]
    seeds = [u for u in candidates
             if not u.has_text and area(u.box) >= SEED_AREA * W * H]
    visual = None
    if seeds:
        comps: list[list[Unit]] = []
        for s in sorted(seeds, key=lambda u: -area(u.box)):
            hit = [c for c in comps if any(overlaps(s.box, v.box, pad) for v in c)]
            merged = [s] + [v for c in hit for v in c]
            comps = [c for c in comps if c not in hit] + [merged]
        core = max(comps, key=lambda c: sum(area(v.box) for v in c))
        # Pull in rigid (non-reflow) items that overlap the core.
        changed = True
        while changed:
            changed = False
            for u in candidates:
                if u in core or u.reflow or u in seeds and u not in core:
                    continue
                if any(overlaps(u.box, v.box) for v in core):
                    core.append(u)
                    changed = True
        visual = core

    visual_core = None
    if visual:
        visual_core = union(u.box for u in visual)
        attached = [d for d in deco
                    if contains(visual_core, box_of(d), tol=0.15 * short)
                    or overlaps(visual_core, box_of(d)) and
                    visual_core[0] <= (d["x"] + d["width"] / 2) <= visual_core[2] and
                    visual_core[1] <= (d["y"] + d["height"] / 2) <= visual_core[3]]
        deco = [d for d in deco if d not in attached]
        visual_unit = Unit([e for u in visual for e in u.els] + attached)
        visual_parts = [u for u in visual if u.has_text]
        others = [u for u in others if u not in visual]
    else:
        visual_unit = None
        visual_parts = []

    _tag_rows(others)
    flow = xy_order(others, W, H)

    # Corner items: small rigid things hugging a corner, not in the
    # container and not part of a row of similar items.
    corner = []
    for u in list(flow):
        if (u.reflow or u.row is not None or u in children
                or area(u.box) > CORNER_MAX_AREA * W * H):
            continue
        near_l = u.box[0] <= CORNER_DIST * W
        near_r = u.box[2] >= (1 - CORNER_DIST) * W
        near_t = u.box[1] <= CORNER_DIST * H
        near_b = u.box[3] >= (1 - CORNER_DIST) * H
        # text-bearing items only pin to top corners: at the bottom they're
        # usually the CTA, which belongs to the copy
        v = "t" if near_t else "b" if near_b and not u.has_text else None
        h = "l" if near_l else "r" if near_r else None
        if v and h:
            corner.append((u, v + h))
            flow.remove(u)

    head = headline_id(els)
    for u in flow + [u for u, _ in corner] + visual_parts:
        u.primary = head is not None and any(e["id"] == head for e in u.els)

    texts = [e for u in flow if u.reflow for e in u.els]
    centered = bool(texts) and sum(e.get("text_align") == "center" for e in texts) > len(texts) / 2

    return Analysis(
        canvas=(W, H), background=background, edge=edge, deco=deco,
        visual=visual_unit, visual_parts=visual_parts,
        container=container,
        container_children={e["id"] for u in children for e in u.els},
        corner=corner, flow=flow, centered=centered,
        max_font=max((e["font_size"] for e in els if e["type"] == "text"), default=0.0),
        elements=els)


def _gaps(units: list[Unit], axis: int) -> list[tuple[float, float]]:
    """Empty bands along `axis` (0 = x, 1 = y) between the units' projections."""
    spans = sorted((u.box[axis], u.box[axis + 2]) for u in units)
    gaps, end = [], spans[0][1]
    for a0, a1 in spans[1:]:
        if a0 > end:
            gaps.append((end, a0))
        end = max(end, a1)
    return gaps


def _split(units: list[Unit], axis: int, cut: tuple[float, float]):
    mid = (cut[0] + cut[1]) / 2
    lo = [u for u in units if (u.box[axis] + u.box[axis + 2]) / 2 < mid]
    return lo, [u for u in units if u not in lo]


_row_ids = iter(range(10**9))


def xy_order(units: list[Unit], W: float, H: float) -> list[Unit]:
    order = _xy_order(units, W, H)
    # pull pre-tagged row members together at the first member's position
    out, done = [], set()
    for u in order:
        if u.row is None:
            out.append(u)
        elif u.row not in done:
            done.add(u.row)
            out += sorted((v for v in order if v.row == u.row), key=lambda v: v.box[0])
    return out


def _xy_order(units: list[Unit], W: float, H: float) -> list[Unit]:
    """Reading order by recursive XY-cut: split on the widest empty band
    (rows before columns on ties), recurse, read top-to-bottom and
    left-to-right. Leaves of a column split made only of similar rigid
    items (badges, chips) are tagged as one row."""
    if len(units) <= 1:
        return list(units)
    gy = max(_gaps(units, 1), key=lambda g: g[1] - g[0], default=None)
    gx = max(_gaps(units, 0), key=lambda g: g[1] - g[0], default=None)
    wy = (gy[1] - gy[0]) / H if gy else -1
    wx = (gx[1] - gx[0]) / W if gx else -1
    if wy < 0 and wx < 0:
        # overlapping everything: fall back to top-left first
        return sorted(units, key=lambda u: (u.box[1], u.box[0]))
    if wy >= wx * 0.8:
        a, b = _split(units, 1, gy)
        return _xy_order(a, W, H) + _xy_order(b, W, H)
    parts, rest = [], units
    while True:
        g = max(_gaps(rest, 0), key=lambda g: g[1] - g[0], default=None) if len(rest) > 1 else None
        if not g:
            parts.append(rest)
            break
        a, rest = _split(rest, 0, min(_gaps(rest, 0)))  # peel off leftmost column
        parts.append(a)
    singles = [p[0] for p in parts if len(p) == 1]
    if (len(parts) >= 2 and len(singles) == len(parts)
            and all(not u.reflow for u in singles)
            and max(u.h for u in singles) <= 1.6 * min(u.h for u in singles)):
        rid = next(_row_ids)
        for u in singles:
            u.row = rid
        return sorted(singles, key=lambda u: u.box[0])
    out = []
    for p in parts:
        out += _xy_order(p, W, H)
    return out


def _signature(u: Unit) -> tuple:
    return tuple(sorted((e["type"], e.get("shape_type", "")) for e in u.els))


def _tag_rows(units: list[Unit]) -> None:
    """Rigid look-alikes (same parts, same height, same top) side by side
    are one row — badge strips, chip lists — even if the XY-cut can't
    isolate them."""
    pool = [u for u in units if not u.reflow and len(u.els) > 1]
    while pool:
        u = pool.pop(0)
        mates = [v for v in pool if _signature(v) == _signature(u)
                 and abs(v.h - u.h) <= 0.2 * max(u.h, v.h)
                 and abs(v.box[1] - u.box[1]) <= 0.25 * max(u.h, v.h)]
        if mates:
            rid = next(_row_ids)
            for v in [u] + mates:
                v.row = rid
                if v in pool:
                    pool.remove(v)


def rows_of(units: list[Unit]) -> list[list[Unit]]:
    """Group consecutive units that share a row tag."""
    out: list[list[Unit]] = []
    for u in units:
        if out and u.row is not None and out[-1][-1].row == u.row:
            out[-1].append(u)
        else:
            out.append([u])
    return out
