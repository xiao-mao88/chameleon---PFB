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
references) and the whole column shrinks together until it fits. Text never
shrinks below a readability floor while the column can still fit by
shrinking something else: small type stops at the floor and the headline,
visuals and gaps keep shrinking. The floor is relaxed (MIN_FONT, then
MIN_READABLE, then none) only when the copy can't fit even then. Buttons
that would get too small to read are re-wrapped around their label instead
of being scaled as a block.
"""

import math
from math import comb
from dataclasses import dataclass, replace
from itertools import combinations

from engine.analyze import Analysis, Unit, area, box_of, rows_of, union, xy_order

try:
    from engine.textmeasure import measure_text
except Exception:  # renderer/skia unavailable: rough estimate instead
    measure_text = None

GAP_CAP = 0.06          # max source gap between copy items, as fraction of source height
MIN_GAP = 0.012         # min gap between copy items, as fraction of target short side
VISUAL_GAP = 0.04       # gap between hero visual and copy, fraction of short side
MIN_FONT = 9.0          # smallest font any reference uses; fitting won't go below it if it can help it
MIN_READABLE = 7.0      # last floor before text is allowed to become unreadable
FLOORS = (MIN_FONT, MIN_READABLE, 0.0)


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
        if isinstance(el.get(k), (int, float)):
            out[k] = el[k] * s
    if el["type"] == "text":
        out["font_size"] = max(1.0, el["font_size"] * s)
    return out


def place_rigid(unit: Unit, X: float, Y: float, s: float) -> list[dict]:
    return [scaled_copy(e, unit.box[0], unit.box[1], s, X, Y) for e in unit.els]


def tracked(el: dict, font: float, t: float = 1.0) -> dict:
    """el with letter_spacing scaled to `font` (tracking is set relative to
    the source font size), times t."""
    ls = el.get("letter_spacing")
    if not ls:
        return el
    return dict(el, letter_spacing=ls * t * font / max(el["font_size"], 1e-6))


TRACKING_STEPS = (1.0, 0.75, 0.5, 0.25, 0.0)


def fit_tracking(el: dict, font: float, width: float):
    """Largest tracking step whose drawn lines fit `width`. The renderer
    breaks lines on untracked widths and then draws them wider, so wrapped
    tracked text can't fit at any font size; loosen the tracking instead
    (short tracked labels on one line keep it all). Returns (el, measure)."""
    for t in TRACKING_STEPS:
        e = tracked(el, font, t)
        m = measure(e, font, width)
        if m[0] <= width - 0.5:
            return e, m
    return e, m


def is_tracked(el: dict) -> bool:
    return bool(el.get("letter_spacing")) or bool(el.get("capitalize"))


def measure(el: dict, font: float, width: float) -> tuple[float, float, int]:
    """(text width, text height, lines) for el at this font size and box width.
    letter_spacing is taken as given - pass tracked(el, font) for a new size."""
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
        self.primary = unit.primary            # the headline keeps MIN_FONT
        self.align = align
        self._cache: dict = {}
        self._styled: dict = {}     # (font, width) -> el with fitted tracking
        r = math.radians(float(self.el.get("rotation") or 0.0))
        self.rot = (abs(math.cos(r)), abs(math.sin(r))) if r else None

    def _drawn(self, w, h):
        """Bounds of a w x h box once rotated about its centre."""
        if self.rot is None:
            return w, h
        c, s = self.rot
        return max(w, w * c + h * s), max(h, w * s + h * c)

    def _fit(self, f: float, cw: float, floor: float = 0.0):
        if self.rot is None:
            return self._fit_box(f, cw, floor)
        # rotated: of the boxes whose rotated bounds fit the column width,
        # the one that is least tall once drawn (columns stack vertically)
        key = ("rot", round(f, 4), round(cw, 1), floor)
        if key not in self._cache:
            best = None
            for i in range(12):
                fit = self._fit_box(f, cw * (0.15 + 0.85 * i / 11), floor)
                dw, dh = self._drawn(fit[1], fit[2])
                # keep the requested size (a narrow box shrinks long words)
                rank = (-round(fit[0], 2), dh)
                if dw <= cw and (best is None or rank < best[0]):
                    best = (rank, fit)
            self._cache[key] = best[1] if best else self._fit_box(f, 0.15 * cw, floor)
        return self._cache[key]

    def _fit_box(self, f: float, cw: float, floor: float = 0.0):
        key = (round(f, 4), round(cw, 1), floor)
        if key not in self._cache:
            if floor and self.primary:
                floor = max(floor, MIN_FONT)
            font = max(1.0, self.pref * f, floor)
            if is_tracked(self.el):
                self._cache[key] = self._fit_tracked(font, cw)
                return self._cache[key]
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

    def _fit_tracked(self, font: float, cw: float):
        """Letter-spaced or capitalised text: the renderer breaks lines on
        plain widths but draws them wider, so keep the box at the full
        column width and shrink the font until the drawn lines fit it."""
        w = cw
        for _ in range(12):
            font = math.ceil(font * 100) / 100      # finalize rounds font_size
            el, (tw, th, _) = fit_tracking(self.el, font, w)
            if tw <= w - 0.5 or font <= 1.0:
                break
            font = max(1.0, font * max(0.85, (w - 0.5) / tw * 0.98))
        self._styled[(font, w)] = el
        return font, w, th + 0.15 * font

    def size(self, f, cw, floor=0.0):
        _, w, h = self._fit(f, cw, floor)
        return self._drawn(w, h)

    def font_ok(self, f, cw, floor):
        need = max(floor, MIN_FONT) if floor and self.primary else floor
        return self._fit(f, cw, floor)[0] >= need - 1e-6

    def place(self, x, y, f, cw, floor=0.0):
        font, w, h = self._fit(f, cw, floor)
        el = dict(self._styled.get((font, w), self.el))
        dw, dh = self._drawn(w, h)       # (x, y) is where the drawn bounds start
        el.update(x=x + (dw - w) / 2, y=y + (dh - h) / 2, width=w, height=h, font_size=font)
        if self.align:
            el["text_align"] = self.align
        return [el]


class RigidItem:
    """Images, groups, buttons: uniformly scaled, never distorted."""

    def __init__(self, unit: Unit, k: float, max_fs: float | None = None):
        self.unit = unit
        self.k, floor = _rigid_scale(unit.els, k, max_fs)
        self.min_f = floor / self.k
        self.small = _smallest_font(unit.els)

    def scale(self, f, cw, floor=0.0):
        s = self.k * f
        if self.small and floor:
            s = max(s, floor / self.small)
        return max(1e-3, min(s, cw / max(self.unit.w, 1e-6)))

    def size(self, f, cw, floor=0.0):
        s = self.scale(f, cw, floor)
        return self.unit.w * s, self.unit.h * s

    def font_ok(self, f, cw, floor):
        return not self.small or self.small * self.scale(f, cw, floor) >= floor - 1e-6

    def place(self, x, y, f, cw, floor=0.0):
        return place_rigid(self.unit, x, y, self.scale(f, cw, floor))


class ButtonItem(RigidItem):
    """A CTA-like pill. Scaled as a block while that keeps its label at the
    floor; past that, the pill is re-wrapped around a readable label (a
    rectangle may change shape, the copy may not shrink further)."""

    def __init__(self, unit: Unit, k: float, max_fs: float | None = None,
                 compact: bool = False):
        super().__init__(unit, k, max_fs)
        self.compact = compact      # always re-wrap (pills in a wrapped row)
        self._styled: dict = {}     # reflow key -> label with fitted tracking
        self.shape = next(e for e in unit.els if e["type"] == "shape")
        self.label = next(e for e in unit.els if e["type"] == "text")
        sh, lb = self.shape, self.label
        fs = lb["font_size"]
        tw, th, _ = measure(lb, fs, lb["width"])
        if tw > lb["width"] + 1 or th > lb["height"] + 1:
            # the label already spills out of its pill in the source (long or
            # translated copy): always re-wrap it rather than scale the spill
            self.compact = True
            tw, th = min(tw, lb["width"]), min(th, lb["height"])
        # padding around the label as a fraction of its font size
        self.pad_x = min(max((sh["width"] - tw) / 2 / fs, 0.4), 1.2)
        self.pad_y = min(max((sh["height"] - th) / 2 / fs, 0.25), 0.8)
        self._cache: dict = {}

    def _reflow(self, f, cw, floor):
        rigid = self.k * f
        if not self.compact and (not floor or
                                 self.label["font_size"] * self.scale(f, cw, floor) >= floor - 1e-6):
            return None
        key = (round(f, 4), round(cw, 1), floor)
        if key not in self._cache:
            font = max(floor, self.label["font_size"] * rigid, 1.0)
            px, py = self.pad_x * font, self.pad_y * font
            inner = max(cw - 2 * px, 1.0)
            lb, (tw, th, _) = fit_tracking(self.label, font, inner)
            for _ in range(8):                # a word (or tracking) too wide
                if tw <= inner - 0.5 or font <= 1.0:
                    break
                font = max(1.0, font * max(0.85, (inner - 0.5) / tw * 0.98))
                px, py = self.pad_x * font, self.pad_y * font
                inner = max(cw - 2 * px, 1.0)
                lb, (tw, th, _) = fit_tracking(self.label, font, inner)
            font = math.ceil(font * 100) / 100   # finalize rounds font_size
            lb, (tw, th, _) = fit_tracking(self.label, font, inner)
            lw = inner if is_tracked(lb) else min(inner, tw + 2)
            # same guard as TextItem: rounding must not flip a line break
            tw2, th2, _ = measure(lb, font, lw - 0.5)
            if tw2 > lw - 0.5 or th2 > th + 0.5:
                lw = inner
                tw2, th2, _ = measure(lb, font, lw - 0.5)
            self._cache[key] = (font, px, py, lw, th2 + 0.15 * font)
            self._styled[key] = lb
        return self._cache[key]

    def font_ok(self, f, cw, floor):
        r = self._reflow(f, cw, floor)
        return super().font_ok(f, cw, floor) if r is None else r[0] >= floor - 1e-6

    def size(self, f, cw, floor=0.0):
        r = self._reflow(f, cw, floor)
        if r is None:
            return super().size(f, cw, floor)
        _, px, py, lw, lh = r
        return lw + 2 * px, lh + 2 * py

    def place(self, x, y, f, cw, floor=0.0):
        r = self._reflow(f, cw, floor)
        if r is None:
            return super().place(x, y, f, cw, floor)
        font, px, py, lw, lh = r
        w, h = lw + 2 * px, lh + 2 * py
        out = []
        for e in self.unit.els:
            el = dict(e)
            if e is self.shape:
                el.update(x=x, y=y, width=w, height=h)
                if isinstance(e.get("corner_radius"), (int, float)):
                    ratio = e["corner_radius"] / max(min(e["width"], e["height"]), 1e-6)
                    el["corner_radius"] = min(ratio * min(w, h), 0.5 * min(w, h))
                if isinstance(e.get("stroke_width"), (int, float)):
                    el["stroke_width"] = e["stroke_width"] * font / self.label["font_size"]
            else:
                el = dict(self._styled[(round(f, 4), round(cw, 1), floor)])
                el.update(x=x + px, y=y + py, width=lw, height=lh, font_size=font,
                          text_align="center")
            out.append(el)
        return out


class LineItem:
    """Several units that sat side by side in the source (badge rows).
    Kept as one row while that keeps the labels readable; on narrow columns
    the row wraps onto as few extra lines as needed (centred lines)."""

    def __init__(self, units: list[Unit], k: float, max_fs: float):
        self.units = units
        self.k, floor = _rigid_scale([e for u in units for e in u.els], k, max_fs)
        self.min_f = floor / self.k
        self.box = union(u.box for u in units)
        self.small = _smallest_font([e for u in units for e in u.els])
        xs = sorted(units, key=lambda u: u.box[0])
        gaps = [max(0.0, b.box[0] - a.box[2]) for a, b in zip(xs, xs[1:])]
        self.gap = sum(gaps) / len(gaps) if gaps else 0.0
        self.line_gap = max(self.gap, 0.25 * max(u.h for u in units))
        # when the row can't stay whole, its units are packed one by one:
        # pills re-wrapped around a readable label, anything else rigid
        self.pills = [ButtonItem(u, k, max_fs, compact=True) if u.is_labelled_shape
                      else RigidItem(u, k, max_fs) for u in units]
        # some label already spills out of its pill in the source
        self.spilled = any(u.is_labelled_shape and ButtonItem(u, k, max_fs).compact
                           for u in units)
        self.all_pills = all(u.is_labelled_shape for u in units)

    def _lines(self, per):
        return [self.units[i:i + per] for i in range(0, len(self.units), per)]

    def _extent(self, per):
        """(width, height) in source units with `per` units on each line."""
        if per >= len(self.units):
            return self.box[2] - self.box[0], self.box[3] - self.box[1]
        lines = self._lines(per)
        w = max(sum(u.w for u in ln) + self.gap * (len(ln) - 1) for ln in lines)
        h = sum(max(u.h for u in ln) for ln in lines) + self.line_gap * (len(lines) - 1)
        return w, h

    def _fit(self, f, cw, floor):
        """(units per line, scale); 0 units per line = re-wrapped pills."""
        s = self.k * f
        if self.small and floor:
            s = max(s, floor / self.small)
        if self.spilled:
            return 0, s
        w, _ = self._extent(len(self.units))
        if w * s <= cw:
            return len(self.units), max(1e-3, s)
        if self.all_pills and floor:
            return 0, s                 # wrap as compact pills
        for per in range(len(self.units) - 1, 0, -1):
            w, _ = self._extent(per)
            if w * s <= cw:
                return per, max(1e-3, s)
        if floor:
            return 0, s                 # mixed row too wide even one per line
        w, _ = self._extent(1)
        return 1, max(1e-3, min(s, cw / max(w, 1e-6)))

    def _pill_lines(self, f, cw, floor):
        """Greedy-pack the re-wrapped pills into centred lines."""
        g = 0.6 * floor if floor else max(self.gap * self.k * f, 2.0)
        lines, cur, cur_w = [], [], 0.0
        for b in self.pills:
            w, h = b.size(f, cw, floor)
            if cur and cur_w + g + w > cw:
                lines.append(cur)
                cur, cur_w = [], 0.0
            cur_w += (g if cur else 0.0) + w
            cur.append((b, w, h))
        lines.append(cur)
        return lines, g

    def _pill_extent(self, f, cw, floor):
        lines, g = self._pill_lines(f, cw, floor)
        w = max(sum(w for _, w, _ in ln) + g * (len(ln) - 1) for ln in lines)
        h = sum(max(h for _, _, h in ln) for ln in lines) + g * (len(lines) - 1)
        return w, h

    def scale(self, f, cw, floor=0.0):
        return self._fit(f, cw, floor)[1]

    def size(self, f, cw, floor=0.0):
        per, s = self._fit(f, cw, floor)
        if per == 0:
            return self._pill_extent(f, cw, floor)
        w, h = self._extent(per)
        return w * s, h * s

    def font_ok(self, f, cw, floor):
        per, s = self._fit(f, cw, floor)
        if per == 0:
            return all(b.font_ok(f, cw, floor) for b in self.pills)
        return not self.small or self.small * s >= floor - 1e-6

    def place(self, x, y, f, cw, floor=0.0):
        per, s = self._fit(f, cw, floor)
        out = []
        if per == 0:
            lines, g = self._pill_lines(f, cw, floor)
            total_w, _ = self._pill_extent(f, cw, floor)
            for ln in lines:
                lw = sum(w for _, w, _ in ln) + g * (len(ln) - 1)
                lh = max(h for _, _, h in ln)
                ux = x + (total_w - lw) / 2
                for b, w, h in ln:
                    out += b.place(ux, y + (lh - h) / 2, f, cw, floor)
                    ux += w + g
                y += lh + g
            return out
        if per >= len(self.units):
            for u in self.units:
                out += place_rigid(u, x + (u.box[0] - self.box[0]) * s,
                                   y + (u.box[1] - self.box[1]) * s, s)
            return out
        total_w, _ = self._extent(per)
        for ln in self._lines(per):
            lw = sum(u.w for u in ln) + self.gap * (len(ln) - 1)
            ux = x + (total_w - lw) / 2 * s
            lh = max(u.h for u in ln)
            for u in ln:
                out += place_rigid(u, ux, y + (lh - u.h) / 2 * s, s)
                ux += (u.w + self.gap) * s
            y += (lh + self.line_gap) * s
        return out


def _smallest_font(els) -> float | None:
    return min((e["font_size"] for e in els if e["type"] == "text"), default=None)


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
        self.small = None


def make_items(units: list[Unit], k: float, align: str | None, max_fs: float):
    items = []
    for line in rows_of(units):
        if len(line) > 1:
            items.append(LineItem(line, k, max_fs))
        elif line[0].reflow:
            items.append(TextItem(line[0], k, align, max_fs))
        elif line[0].is_labelled_shape:
            items.append(ButtonItem(line[0], k, max_fs))
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
    floor: float = 0.0          # px: text in this column stays at or above it

    def heights(self, f, cw):
        sizes = [it.size(f, cw, self.floor) for it in self.items]
        gaps = [max(g * self.k * f, self.min_gap * min(1.0, f * 2)) for g in self.gaps]
        return sizes, gaps

    def total(self, f, cw):
        sizes, gaps = self.heights(f, cw)
        return sum(h for _, h in sizes) + sum(gaps)

    def width(self, f, cw):
        return max((it.size(f, cw, self.floor)[0] for it in self.items), default=0.0)

    def solve(self, cw, ch, f_max=1.0, floors=FLOORS):
        """Largest f <= f_max whose stacked height fits in ch.

        Everything shrinks together (hierarchy kept) while no text is at its
        floor. Text that reaches the floor stops there and the rest keeps
        shrinking. Floors are tried strictest first (FLOORS); the first one
        the copy can fit under is kept in self.floor.
        """
        if not self.items:
            self.floor = floors[0]
            return f_max
        def ok(f):
            return self.total(f, cw) <= ch and (
                not self.floor or all(it.font_ok(f, cw, self.floor) for it in self.items))

        for floor in floors:
            self.floor = floor
            if ok(f_max):
                return f_max
            if floor and ok(1e-3):
                break
        if self.floor == 0.0:
            # no floor: the original rule - keep the smallest text at
            # MIN_FONT if that alone is enough, else shrink everything
            f_floor = min(f_max, max(it.min_f for it in self.items))
            lo, hi = (f_floor, f_max) if self.total(f_floor, cw) <= ch else (0.0, f_floor)
        else:
            lo, hi = 1e-3, f_max
        for _ in range(28):
            mid = (lo + hi) / 2
            if self.total(mid, cw) <= ch:
                lo = mid
            else:
                hi = mid
        return max(lo, 1e-3)

    def min_width(self, ch, floor, avail):
        """Narrowest width at which this column fits in ch with text kept at
        `floor` (None if it doesn't fit even at `avail`)."""
        self.floor = floor

        def fits(w):
            return (self.total(1e-3, w) <= ch
                    and all(it.font_ok(1e-3, w, floor) for it in self.items))

        if not fits(avail):
            return None
        lo, hi = 1.0, avail
        for _ in range(14):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def place(self, x, y, cw, ch, f, align, valign="center"):
        sizes, gaps = self.heights(f, cw)
        total = sum(h for _, h in sizes) + sum(gaps)
        cy = y + (ch - total) / 2 if valign == "center" else y
        out = []
        for i, (it, (w, h)) in enumerate(zip(self.items, sizes)):
            ix = x + (cw - w) / 2 if align == "center" else x
            out += it.place(ix, cy, f, cw, self.floor)
            cy += h + (gaps[i] if i < len(gaps) else 0)
        return out


# --------------------------------------------------------------------------
# Arrangements
# --------------------------------------------------------------------------

@dataclass
class Plan:
    elements: list[dict]
    quality: float
    floor: float = FLOORS[0]    # strictest readability floor all copy fits under


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
        small = _smallest_font(u.els)
        button = None
        b = ButtonItem(u, k) if u.is_labelled_shape else None
        if b is not None and b.compact:
            # its label spilled out of the pill in the source: re-wrap it at
            # its natural size (or the floor), within the corner's share
            floor = MIN_FONT if small * s < MIN_FONT else 0.0
            cw = W - 2 * m if floor else max(0.35 * W, min(W - 2 * m, u.w * s))
            f = s / max(b.k, 1e-6)
            button = (b, floor, f, cw)
            w, h = b.size(f, cw, floor)
        elif small and small * s < MIN_FONT:
            # too small to read at its corner share: grow it (a pill is
            # re-wrapped around a readable label) up to the canvas width
            cw = W - 2 * m
            if b is not None:
                for floor in FLOORS[:-1]:
                    if b.font_ok(1e-3, cw, floor):
                        button = (b, floor, 1e-3, cw)
                        w, h = b.size(1e-3, cw, floor)
                        break
            if button is None:
                s = max(s, min(MIN_FONT / small, cw / max(u.w, 1e-6),
                               0.5 * H / max(u.h, 1e-6)))
                w, h = u.w * s, u.h * s
        x = m if where[1] == "l" else W - m - w
        y = m if where[0] == "t" else H - m - h
        if button is not None:
            bi, floor, f, cw = button
            out += bi.place(x, y, f, cw, floor)
        else:
            out += place_rigid(u, x, y, s)
        if where[0] == "t":
            top = max(top, h + m)
        else:
            bottom = max(bottom, h + m)
    return out, top, bottom


def vertical(a: Analysis, W, H, fold_corners: bool = False) -> Plan:
    W0, H0 = a.canvas
    if fold_corners and a.corner:
        # small canvas: corner items join the copy instead of reserving
        # bands above and below it
        a = replace(a, corner=[], flow=xy_order(a.flow + [u for u, _ in a.corner], W0, H0))
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
    return Plan(els + corner_els, quality, col.floor)


def horizontal(a: Analysis, W, H, vis_frac: float | None = None, max_cols: int = 4) -> Plan:
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
        frac = vis_frac or (0.6 if r < 4 else 0.3)
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
    # a button inside a text panel stays in the panel's column (strips
    # excepted: there everything needs its own column)
    in_panel = a.container_children if a.container and not strip else set()
    first_button = next((i for i, u in enumerate(flow) if r >= 2.5 and u.is_button
                         and not any(e["id"] in in_panel for e in u.els)), None)
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

    copy_start = len(els)
    floors_used = []
    quality_fs = []
    cols = []
    for units, share in ((lead, 0.18), (action, 0.3)):
        if units:
            c = column(units, None)
            cw = share * (right - left)
            f = c.solve(cw, ch, 1.0)
            floors_used.append(c.floor)
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
        floors_used.append(c.floor)
        w = c.width(f, cw)
        els += c.place(x, ch_top, max(w, 1.0) if w < cw else cw, ch, f, "left")
        quality_fs.insert(0, f)
    f_main = min(quality_fs) if quality_fs else 1.0
    floor = min(floors_used, default=FLOORS[0])
    if floor < MIN_FONT:
        # copy only fits as unreadable type: spread it over more columns
        units = [u for u in flow if u in lead or u in main or u in action]
        alt = strip_columns(units, left, right, ch_top, ch, gap, column, max_cols)
        if alt is not None and alt[0] > floor:
            els = els[:copy_start] + alt[2]
            floor, f_main = alt[0], alt[1]
    return Plan(els + corner_els, _quality(a, f_main, els, W, H), floor)


EXHAUSTIVE_SPLITS = 2000   # above this many ways to split the copy, use the DP


def strip_columns(units, x0, x1, y, ch, gap, column, max_cols=4):
    """Lay reading-ordered units out as up to max_cols side-by-side columns
    on a thin strip, choosing the split that keeps text at the strictest
    readability floor, then the largest type. Returns (floor, f, elements)
    or None if no split keeps text above the last floor.

    Splits are contiguous runs of the reading order. Few units: every split
    is tried. Many units: for each column count, a DP finds the split with
    the least total width (O(n^2 * columns) instead of combinatorial)."""
    n_units = len(units)
    cols_cache: dict = {}
    mw_cache: dict = {}

    def col(i, j):
        if (i, j) not in cols_cache:
            cols_cache[(i, j)] = column(units[i:j], None)
        return cols_cache[(i, j)]

    def mw(i, j, floor, avail):
        key = (i, j, floor, avail)
        if key not in mw_cache:
            mw_cache[key] = col(i, j).min_width(ch, floor, avail)
        return mw_cache[key]

    def splits(n, floor, avail):
        """Candidate splits into n columns, as bounds tuples."""
        if sum(comb(n_units - 1, k) for k in range(min(max_cols, n_units))) <= EXHAUSTIVE_SPLITS:
            for cuts in combinations(range(1, n_units), n - 1):
                yield (0,) + cuts + (n_units,)
            return
        INF = float("inf")
        # best[k][j]: least summed width putting units[:j] in k columns
        best = [[INF] * (n_units + 1) for _ in range(n + 1)]
        back = [[0] * (n_units + 1) for _ in range(n + 1)]
        best[0][0] = 0.0
        for k in range(1, n + 1):
            for i in range(k - 1, n_units):
                if best[k - 1][i] == INF:
                    continue
                for j in range(i + 1, n_units - (n - k) + 1):
                    w = mw(i, j, floor, avail)
                    if w is None:
                        break           # longer runs only get taller
                    if best[k - 1][i] + w < best[k][j]:
                        best[k][j], back[k][j] = best[k - 1][i] + w, i
        if best[n][n_units] == INF:
            return
        bounds, j = [n_units], n_units
        for k in range(n, 0, -1):
            j = back[k][j]
            bounds.append(j)
        yield tuple(reversed(bounds))

    best = None
    for n in range(1, min(max_cols, n_units) + 1):
        avail = x1 - x0 - gap * (n - 1)
        if avail <= 0:
            break
        for floor in FLOORS[:-1]:
            if best is not None and floor < best[0][0]:
                break
            found = False
            for bounds in splits(n, floor, avail):
                pairs = list(zip(bounds, bounds[1:]))
                mins = []
                for i, j in pairs:
                    w = mw(i, j, floor, avail)
                    if w is None:
                        break
                    mins.append(w)
                if len(mins) < n or sum(mins) > avail:
                    continue
                found = True
                cols = [col(i, j) for i, j in pairs]
                extra = (avail - sum(mins)) / n
                widths = [w + extra for w in mins]
                fs = [c.solve(w, ch, 1.0, floors=(floor,)) for c, w in zip(cols, widths)]
                key = (floor, min(fs), -n)
                if best is None or key > best[0]:
                    best = (key, cols, widths, fs)
            if found:
                break
    if best is None:
        return None
    (floor, f_min, _), cols, widths, fs = best
    used = []
    for c, w, f in zip(cols, widths, fs):
        c.floor = floor
        used.append(min(w, c.width(f, w)))
    # spread the slack between columns so the strip reads evenly
    slack = (x1 - x0) - sum(used)
    g = slack / (len(cols) - 1) if len(cols) > 1 else 0.0
    x = x0 if len(cols) > 1 else x0 + 0.0
    out = []
    for c, w, f in zip(cols, used, fs):
        c.floor = floor
        out += c.place(x, y, w, ch, f, "left")
        x += w + g
    return floor, f_min, out


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


def detach(a: Analysis, parts: list[Unit]) -> Analysis:
    """Move these text-bearing units off the hero visual into the copy
    (reading order kept), so they are sized on their own."""
    ids = {e["id"] for u in parts for e in u.els}
    rest = [e for e in a.visual.els if e["id"] not in ids]
    if not parts or not any(e["type"] != "text" for e in rest):
        return a
    visual = Unit(rest)
    visual.kind = "visual"
    W0, H0 = a.canvas
    return replace(a, visual=visual, visual_parts=[u for u in a.visual_parts if u not in parts],
                   flow=xy_order(a.flow + parts, W0, H0))


def layout(a: Analysis, W: int, H: int) -> list[dict]:
    """Badges/captions riding on the hero visual are scaled with it. If that
    leaves their text unreadable, lay out again with them in the copy."""
    out = _layout(a, W, H)
    if a.visual is None or not a.visual_parts:
        return out
    fonts = {e["id"]: e.get("font_size", 0) for e in out if e["type"] == "text"}
    small = [u for u in a.visual_parts
             if any(fonts.get(e["id"], 0) < MIN_READABLE for e in u.els if e["type"] == "text")]
    if not small:
        return out
    b = detach(a, small)
    if b is a:
        return out
    alt = _layout(b, W, H)

    def unreadable(els):
        return sum(e["type"] == "text" and e["font_size"] < MIN_READABLE for e in els)

    return alt if unreadable(alt) < unreadable(out) else out


def _layout(a: Analysis, W: int, H: int) -> list[dict]:
    r = W / H
    if r < 0.9:
        plan = vertical(a, W, H)
        if plan.floor < MIN_FONT and a.corner:
            alt = vertical(a, W, H, fold_corners=True)
            if (alt.floor, alt.quality) > (plan.floor, plan.quality):
                plan = alt
    elif r > 1.25 or a.visual is None:
        plan = horizontal(a, W, H) if r > 1.25 else vertical(a, W, H)
        if r > 1.25 and plan.floor < MIN_FONT:
            # a strip too small for the copy: give the hero visual less of
            # the width and allow more columns before accepting tiny type
            for frac in (0.18, 0.1):
                alt = horizontal(a, W, H, vis_frac=frac, max_cols=6)
                if (alt.floor, alt.quality) > (plan.floor, plan.quality):
                    plan = alt
                if plan.floor >= MIN_FONT:
                    break
    else:
        # Near-square: centred designs stack, left-aligned designs go
        # side by side (what the reference designers did).
        W0, H0 = a.canvas
        if W0 / H0 <= 1.3:
            plan = vertical(a, W, H) if a.centered else horizontal(a, W, H)
            if plan.floor < MIN_FONT:
                # the usual arrangement can't keep the copy readable: try the others
                alts = [horizontal(a, W, H) if a.centered else vertical(a, W, H)]
                if a.corner:
                    alts.append(vertical(a, W, H, fold_corners=True))
                for alt in alts:
                    if (alt.floor, alt.quality) > (plan.floor, plan.quality):
                        plan = alt
        else:
            # wide sources often left-align even centred designs: compare
            v, h = vertical(a, W, H), horizontal(a, W, H)
            plan = v if v.quality >= h.quality else h
    if plan.floor < MIN_FONT:
        # the usual arrangement can't keep the copy readable: try them all
        alts = [vertical(a, W, H), horizontal(a, W, H)]
        if a.corner:
            alts.append(vertical(a, W, H, fold_corners=True))
        for alt in alts:
            if (alt.floor, alt.quality) > (plan.floor, plan.quality):
                plan = alt
    k = _k(a, W, H)
    out = place_free(a, W, H, k) + plan.elements
    placed = {e["id"]: e for e in out}
    cont = wrap_container(a, placed, W, H, k)
    if cont is not None:
        out.append(cont)
    return out
