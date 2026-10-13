"""
Place analysed units on the target canvas.

Two arrangements, picked by target shape (and, for near-square targets, by
which one leaves the copy bigger and the hero visual less shrunk):

  vertical    one centred column: hero visual and copy in source reading
              order (story, portrait, skyscraper)
  horizontal  hero visual on one side, copy column(s) beside it; on wide
              banners buttons get their own right-hand column, on thin
              strips small images (logo) get their own left-hand column

Copy is fitted by one scale factor per column, found by binary search:
every font and rigid item starts at its "natural" size (source size *
sqrt(target area / source area) — the rule that best matches the hand-drawn
references) and the whole column shrinks together until it fits.
"""

import math
from dataclasses import dataclass

from engine.analyze import Analysis, Unit, area, box_of, rows_of, union

try:
    from engine.textmeasure import measure_text
except Exception:  # renderer/skia unavailable: rough estimate instead
    measure_text = None

GAP_CAP = 0.06          # max source gap between copy items, as fraction of source height
MIN_GAP = 0.012         # min gap between copy items, as fraction of target short side
VISUAL_GAP = 0.04       # gap between hero visual and copy, fraction of short side
MIN_FONT = 9.0          # smallest font any reference uses; fitting won't go below it if it can help it


def font_scale(k: float, rel: float) -> float:
    """How much a font scales when the canvas area scales by k**2.

    Fitted on the hand-drawn references (504 pairs): big type tracks the
    canvas almost linearly, small type (relative to the largest font in the
    design, `rel` <= 1) much less so — small canvases flatten the hierarchy,
    big ones stretch it. Median error ~11% vs ~15% for plain k.
    """
    rel = min(max(rel, 1e-3), 1.0)
    return math.exp(-0.039) * k ** (0.858 + 0.222 * math.log(rel))


def _texts(unit) -> list[dict]:
    return [e for e in unit.els if e["type"] == "text"]


# --------------------------------------------------------------------------
# Element transforms
# --------------------------------------------------------------------------

def scaled_copy(el: dict, x0: float, y0: float, s: float, X: float, Y: float) -> dict:
    """Copy of `el` scaled by s about (x0, y0), with that point moved to (X, Y)."""
    out = dict(el)
    out["x"] = X + (el["x"] - x0) * s
    out["y"] = Y + (el["y"] - y0) * s
    out["width"] = el["width"] * s
    out["height"] = el["height"] * s
    for k in ("corner_radius", "stroke_width", "letter_spacing"):
        if isinstance(el.get(k), (int, float)) and k != "letter_spacing":
            out[k] = el[k] * s
    if el["type"] == "text":
        out["font_size"] = max(1.0, el["font_size"] * s)
    return out


def place_rigid(unit: Unit, X: float, Y: float, s: float) -> list[dict]:
    return [scaled_copy(e, unit.box[0], unit.box[1], s, X, Y) for e in unit.els]


def measure(el: dict, font: float, width: float) -> tuple[float, float, int]:
    """(text width, text height, lines) for el at this font size and box width."""
    if measure_text is not None:
        try:
            m = measure_text(el, font, width)
            return m["text_width"], m["text_height"], m["lines"]
        except Exception:
            pass
    # Fallback: average glyph ~0.55em, greedy wrap by words.
    lh = float(el.get("line_height", 1.2))
    words = str(el.get("content", "")).split(" ")
    lines, cur, widest = 1, 0.0, 0.0
    for w in words:
        ww = 0.55 * font * (len(w) + 1)
        if cur and cur + ww > width:
            widest, lines, cur = max(widest, cur), lines + 1, ww
        else:
            cur += ww
    return max(widest, cur), lines * font * lh, lines


# --------------------------------------------------------------------------
# Copy items
# --------------------------------------------------------------------------

class TextItem:
    """A lone text element: font scaled, box re-wrapped to the column."""

    def __init__(self, unit: Unit, k: float, align: str | None, max_fs: float):
        self.unit, self.el = unit, unit.els[0]
        fs = self.el["font_size"]
        self.pref = fs * font_scale(k, fs / max_fs)
        self.min_f = min(1.0, MIN_FONT / max(self.pref, 1e-6))
        self.align = align
        self._cache: dict = {}

    def _fit(self, f: float, cw: float):
        key = (round(f, 4), round(cw, 1))
        if key not in self._cache:
            font = max(1.0, self.pref * f)
            tw, th, _ = measure(self.el, font, cw)
            if tw > cw:                       # one unbreakable word too wide
                font = max(1.0, font * cw / tw * 0.98)
                tw, th, _ = measure(self.el, font, cw)
            w = min(cw, tw + 0.02 * font + 2)
            # re-measure a hair narrower than the final box: output rounding
            # must never flip a line break at render time
            tw2, th2, _ = measure(self.el, font, w - 0.5)
            if tw2 > w - 0.5 or th2 > th + 0.5:
                w = cw + 1.0
                tw2, th2, _ = measure(self.el, font, w - 0.5)
            self._cache[key] = (font, w, th2 + 0.15 * font)
        return self._cache[key]

    def size(self, f, cw):
        _, w, h = self._fit(f, cw)
        return w, h

    def place(self, x, y, f, cw):
        font, w, h = self._fit(f, cw)
        el = dict(self.el)
        el.update(x=x, y=y, width=w, height=h, font_size=font)
        if self.align:
            el["text_align"] = self.align
        return [el]


class RigidItem:
    """Images, groups, buttons: uniformly scaled, never distorted."""

    def __init__(self, unit: Unit, k: float, max_fs: float | None = None):
        self.unit = unit
        self.k, floor = _rigid_scale(unit.els, k, max_fs)
        self.min_f = floor / self.k

    def scale(self, f, cw):
        return max(1e-3, min(self.k * f, cw / max(self.unit.w, 1e-6)))

    def size(self, f, cw):
        s = self.scale(f, cw)
        return self.unit.w * s, self.unit.h * s

    def place(self, x, y, f, cw):
        return place_rigid(self.unit, x, y, self.scale(f, cw))


class LineItem:
    """Several units that sat side by side in the source (badge rows)."""

    def __init__(self, units: list[Unit], k: float, max_fs: float):
        self.units = units
        self.k, floor = _rigid_scale([e for u in units for e in u.els], k, max_fs)
        self.min_f = floor / self.k
        self.box = union(u.box for u in units)

    def scale(self, f, cw):
        w = self.box[2] - self.box[0]
        return max(1e-3, min(self.k * f, cw / max(w, 1e-6)))

    def size(self, f, cw):
        s = self.scale(f, cw)
        return (self.box[2] - self.box[0]) * s, (self.box[3] - self.box[1]) * s

    def place(self, x, y, f, cw):
        s = self.scale(f, cw)
        out = []
        for u in self.units:
            out += place_rigid(u, x + (u.box[0] - self.box[0]) * s,
                               y + (u.box[1] - self.box[1]) * s, s)
        return out


def _rigid_scale(els, k, max_fs):
    """(preferred scale, floor scale) for a rigid unit. Units with text
    scale like their largest label would; images like headline-sized type."""
    texts = [e for e in els if e["type"] == "text"]
    if not texts or not max_fs:
        return font_scale(k, 1.0), 0.0
    big = max(e["font_size"] for e in texts)
    small = min(e["font_size"] for e in texts)
    pref = font_scale(k, big / max_fs)
    return pref, min(pref, MIN_FONT / max(small, 1e-6))


class VisualItem(RigidItem):
    """The hero cluster: scaled as one rigid block, no font floor."""

    def __init__(self, unit: Unit, k: float, core):
        self.unit, self.k, self.min_f = unit, k, 0.0
        self.core = core


def make_items(units: list[Unit], k: float, align: str | None, max_fs: float):
    items = []
    for line in rows_of(units):
        if len(line) > 1:
            items.append(LineItem(line, k, max_fs))
        elif line[0].reflow:
            items.append(TextItem(line[0], k, align, max_fs))
        else:
            items.append(RigidItem(line[0], k, max_fs))
    return items


def _item_box(item):
    if isinstance(item, LineItem):
        return item.box
    return item.unit.box


def source_gaps(items, src_h: float) -> list[float]:
    gaps = []
    for a, b in zip(items, items[1:]):
        g = _item_box(b)[1] - _item_box(a)[3]
        gaps.append(min(max(g, 0.0), GAP_CAP * src_h))
    return gaps


# --------------------------------------------------------------------------
# Column stacking
# --------------------------------------------------------------------------

@dataclass
class Column:
    items: list
    gaps: list[float]           # source gaps between items
    k: float
    min_gap: float

    def heights(self, f, cw):
        sizes = [it.size(f, cw) for it in self.items]
        gaps = [max(g * self.k * f, self.min_gap * min(1.0, f * 2)) for g in self.gaps]
        return sizes, gaps

    def total(self, f, cw):
        sizes, gaps = self.heights(f, cw)
        return sum(h for _, h in sizes) + sum(gaps)

    def width(self, f, cw):
        return max((it.size(f, cw)[0] for it in self.items), default=0.0)

    def solve(self, cw, ch, f_max=1.0):
        """Largest f <= f_max whose stacked height fits in ch.

        Everything shrinks together (hierarchy kept). We first try not to go
        below the point where the smallest text hits MIN_FONT; only if the
        copy can't fit even there do we go smaller.
        """
        if not self.items or self.total(f_max, cw) <= ch:
            return f_max
        f_floor = min(f_max, max(it.min_f for it in self.items))
        lo, hi = (f_floor, f_max) if self.total(f_floor, cw) <= ch else (0.0, f_floor)
        for _ in range(28):
            mid = (lo + hi) / 2
            if self.total(mid, cw) <= ch:
                lo = mid
            else:
                hi = mid
        return max(lo, 1e-3)

    def place(self, x, y, cw, ch, f, align, valign="center"):
        sizes, gaps = self.heights(f, cw)
        total = sum(h for _, h in sizes) + sum(gaps)
        cy = y + (ch - total) / 2 if valign == "center" else y
        out = []
        for i, (it, (w, h)) in enumerate(zip(self.items, sizes)):
            ix = x + (cw - w) / 2 if align == "center" else x
            out += it.place(ix, cy, f, cw)
            cy += h + (gaps[i] if i < len(gaps) else 0)
        return out


# --------------------------------------------------------------------------
# Arrangements
# --------------------------------------------------------------------------

@dataclass
class Plan:
    elements: list[dict]
    quality: float


def _k(a: Analysis, W, H):
    W0, H0 = a.canvas
    return math.sqrt((W * H) / (W0 * H0))


def margin(W, H):
    return max(4.0, 0.05 * min(W, H))


def place_corners(a: Analysis, W, H, k):
    """Pin corner units; return elements + the vertical band each corner uses."""
    m = margin(W, H)
    out, top, bottom = [], 0.0, 0.0
    for u, where in a.corner:
        s = min(k, 0.22 * min(W, H) / max(u.h, 1e-6), 0.35 * W / max(u.w, 1e-6))
        w, h = u.w * s, u.h * s
        x = m if where[1] == "l" else W - m - w
        y = m if where[0] == "t" else H - m - h
        out += place_rigid(u, x, y, s)
        if where[0] == "t":
            top = max(top, h + m)
        else:
            bottom = max(bottom, h + m)
    return out, top, bottom


def vertical(a: Analysis, W, H) -> Plan:
    W0, H0 = a.canvas
    k = _k(a, W, H)
    m = margin(W, H)
    corner_els, top, bottom = place_corners(a, W, H, k)
    x0, y0, cw = m, max(m, top + 0.5 * m), W - 2 * m
    ch = H - y0 - max(m, bottom + 0.5 * m)
    align = "center" if a.centered else None

    # Visual inserted into the reading order where it sat in the source.
    before = [u for u in a.flow if a.visual and u.cy < a.visual.cy]
    if a.visual and a.flow:
        # side-by-side source: the visual leads if it was on the left
        fb = union(u.box for u in a.flow)
        vb = a.visual.box
        if min(vb[3], fb[3]) - max(vb[1], fb[1]) > 0.5 * min(vb[3] - vb[1], fb[3] - fb[1]):
            before = [] if a.visual.cx <= (fb[0] + fb[2]) / 2 else list(a.flow)
    after = [u for u in a.flow if u not in before]
    items_b = make_items(before, k, align, a.max_font)
    items_a = make_items(after, k, align, a.max_font)
    vis = VisualItem(a.visual, k * 1.15, a.visual_core) if a.visual else None
    items = items_b + ([vis] if vis else []) + items_a
    gaps = source_gaps(items, H0)
    if vis:
        vg = VISUAL_GAP * min(W, H) / k   # in source units; scaled by k*f later
        if items_b:
            gaps[len(items_b) - 1] = max(gaps[len(items_b) - 1], vg)
        if items_a:
            gaps[len(items_b)] = max(gaps[len(items_b)], vg)
    col = Column(items, gaps, k, MIN_GAP * min(W, H))
    f = col.solve(cw, ch)
    col_align = "center" if (a.centered or vis) else "left"
    els = col.place(x0, y0, cw, ch, f, col_align if a.centered else "left")
    if not a.centered and vis:
        # left-aligned copy, but centre the hero visual across the canvas
        vis_els = {e["id"] for e in vis.unit.els}
        vx = [e for e in els if e["id"] in vis_els]
        if vx:
            bx = union(box_of(e) for e in vx)
            dx = (W - (bx[2] - bx[0])) / 2 - bx[0]
            for e in vx:
                e["x"] += dx
    quality = _quality(a, f, els, W, H)
    return Plan(els + corner_els, quality)


def horizontal(a: Analysis, W, H) -> Plan:
    W0, H0 = a.canvas
    k = _k(a, W, H)
    m = margin(W, H)
    r = W / H
    strip = r >= 6 or H <= 150
    corner_units = []
    if strip:
        corner_units = [u for u, _ in a.corner]
        corner_els, top, bottom = [], 0.0, 0.0
    else:
        corner_els, top, bottom = place_corners(a, W, H, k)
    els: list[dict] = []
    flow = a.flow + corner_units

    # Hero visual on the side it sat on in the source; if the source
    # stacked it above/below the copy, it leads (left) when it came first.
    vis_left = True
    if a.visual is not None and flow:
        fb = union(u.box for u in flow)
        vb = a.visual.box
        dx = (a.visual.cx - (fb[0] + fb[2]) / 2) / W0
        dy = (a.visual.cy - (fb[1] + fb[3]) / 2) / H0
        vis_left = dx <= 0 if abs(dx) >= abs(dy) else dy <= 0
    left, right = m, W - m
    if a.visual is not None:
        frac = 0.6 if r < 4 else 0.3
        vh = H - 2 * m if not strip else H - 2 * max(2.0, 0.08 * H)
        s = min(vh / a.visual.h, frac * (W - 2 * m) / a.visual.w, 1.6 * k)
        vw = a.visual.w * s
        vy = (H - a.visual.h * s) / 2
        vx = left if vis_left else right - vw
        els += place_rigid(a.visual, vx, vy, s)
        gap = VISUAL_GAP * min(W, H) + m * 0.5
        if vis_left:
            left = vx + vw + gap
        else:
            right = vx - gap

    # Columns: [lead images (strip only)] [main copy] [buttons (wide only)]
    lead, main, action = [], [], []
    first_button = next((i for i, u in enumerate(flow) if r >= 2.5 and u.is_button), None)
    for i, u in enumerate(flow):
        if first_button is not None and i >= first_button and first_button >= 2:
            action.append(u)          # CTA and whatever followed it
        elif first_button == i:
            action.append(u)
        elif strip and not u.has_text and area(u.box) < 0.1 * W0 * H0:
            lead.append(u)
        else:
            main.append(u)
    if not main and action:
        main, action = action, []

    text_align = "left"
    ch_top = max(m, top + 0.5 * m) if not strip else max(2.0, 0.1 * H)
    ch_bot = max(m, bottom + 0.5 * m) if not strip else max(2.0, 0.1 * H)
    ch = H - ch_top - ch_bot
    gap = max(m, 0.03 * W) if strip else m

    def column(units, align):
        items = make_items(units, k, align, a.max_font)
        return Column(items, source_gaps(items, H0), k, MIN_GAP * min(W, H))

    quality_fs = []
    cols = []
    for units, share in ((lead, 0.18), (action, 0.3)):
        if units:
            c = column(units, None)
            cw = share * (right - left)
            f = c.solve(cw, ch, 1.0)
            w = min(cw, c.width(f, cw))
            cols.append((c, w, f, units is lead))
    lead_cols = [c for c in cols if c[3]]
    act_cols = [c for c in cols if not c[3]]
    x = left
    for c, w, f, _ in lead_cols:
        els += c.place(x, ch_top, w, ch, f, "left")
        x += w + gap
        quality_fs.append(f)
    xr = right
    for c, w, f, _ in act_cols:
        els += c.place(xr - w, ch_top, w, ch, f, "center")
        xr -= w + gap
        quality_fs.append(f)
    if main:
        c = column(main, text_align)
        cw = max(10.0, xr - x)
        # keep a readable measure on wide formats (references: ~30-45%)
        cw = min(cw, (0.42 if r >= 2 else 0.5) * W)
        f = c.solve(cw, ch, 1.0)
        w = c.width(f, cw)
        els += c.place(x, ch_top, max(w, 1.0) if w < cw else cw, ch, f, "left")
        quality_fs.insert(0, f)
    f_main = min(quality_fs) if quality_fs else 1.0
    return Plan(els + corner_els, _quality(a, f_main, els, W, H))


def _quality(a: Analysis, f: float, els, W, H) -> float:
    """Prefer layouts where copy keeps its natural size and the hero visual
    keeps roughly its share of the canvas."""
    q = min(f, 1.0)
    if a.visual is not None:
        W0, H0 = a.canvas
        ids = {e["id"] for e in a.visual.els}
        placed = [e for e in els if e["id"] in ids]
        if placed:
            share = area(union(box_of(e) for e in placed)) / (W * H)
            src_share = area(a.visual.box) / (W0 * H0)
            q = 0.6 * q + 0.4 * min(1.0, share / max(src_share, 1e-6))
    return q


def place_free(a: Analysis, W, H, k) -> list[dict]:
    """Background, edge-glued shapes and free decorations."""
    W0, H0 = a.canvas
    sx, sy = W / W0, H / H0
    out = []
    for e in a.background:
        out.append(dict(e, x=0, y=0, width=W, height=H))
    for e in a.edge:
        out.append(_edge_place(e, W0, H0, W, H, k))
    for e in a.deco:
        s = k
        if e.get("shape_type") == "ellipse" or e["type"] == "image":
            s = min(k, 0.9 * min(W, H) / max(e["width"], e["height"], 1e-6))
            w, h = e["width"] * s, e["height"] * s
        else:
            w, h = e["width"] * sx, e["height"] * sy
        cx = (e["x"] + e["width"] / 2) * sx
        cy = (e["y"] + e["height"] / 2) * sy
        el = dict(e, x=cx - w / 2, y=cy - h / 2, width=w, height=h)
        if "corner_radius" in e:
            el["corner_radius"] = e["corner_radius"] * min(w / max(e["width"], 1e-6),
                                                           h / max(e["height"], 1e-6))
        out.append(el)
    return out


def _edge_place(e, W0, H0, W, H, k):
    tol_x, tol_y = 0.01 * W0, 0.01 * H0
    x0, y0, x1, y1 = box_of(e)
    l, r = x0 <= tol_x, x1 >= W0 - tol_x
    t, b = y0 <= tol_y, y1 >= H0 - tol_y

    def axis(a0, a1, L0, L, lo, hi):
        if lo and hi:
            return 0.0, L
        if lo:
            return 0.0, (a1 / L0) * L
        if hi:
            ln = (L0 - a0) / L0 * L
            return L - ln, ln
        c = (a0 + a1) / 2 / L0 * L
        ln = (a1 - a0) * k
        return c - ln / 2, ln

    x, w = axis(x0, x1, W0, W, l, r)
    y, h = axis(y0, y1, H0, H, t, b)
    return dict(e, x=x, y=y, width=w, height=h)


def wrap_container(a: Analysis, placed: dict[str, dict], W, H, k) -> dict | None:
    if a.container is None:
        return None
    src = a.container.els[0]
    kids = [placed[i] for i in a.container_children if i in placed]
    if not kids:
        return None
    kb = union(box_of(e) for e in kids)
    src_kids = union(box_of(e) for e in a.container.els[1:]) if len(a.container.els) > 1 else None
    pad = 0.04 * min(W, H) + 2
    x0, y0 = max(0.0, kb[0] - pad), max(0.0, kb[1] - pad)
    x1, y1 = min(float(W), kb[2] + pad), min(float(H), kb[3] + pad)
    el = dict(src, x=x0, y=y0, width=x1 - x0, height=y1 - y0)
    if "corner_radius" in src:
        el["corner_radius"] = min(src["corner_radius"] * k, 0.5 * min(x1 - x0, y1 - y0))
    return el


def layout(a: Analysis, W: int, H: int) -> list[dict]:
    r = W / H
    if r < 0.9:
        plan = vertical(a, W, H)
    elif r > 1.25 or a.visual is None:
        plan = horizontal(a, W, H) if r > 1.25 else vertical(a, W, H)
    else:
        # Near-square: centred designs stack, left-aligned designs go
        # side by side (what the reference designers did).
        W0, H0 = a.canvas
        if W0 / H0 <= 1.3:
            plan = vertical(a, W, H) if a.centered else horizontal(a, W, H)
        else:
            # wide sources often left-align even centred designs: compare
            v, h = vertical(a, W, H), horizontal(a, W, H)
            plan = v if v.quality >= h.quality else h
    k = _k(a, W, H)
    out = place_free(a, W, H, k) + plan.elements
    placed = {e["id"]: e for e in out}
    cont = wrap_container(a, placed, W, H, k)
    if cont is not None:
        out.append(cont)
    return out
