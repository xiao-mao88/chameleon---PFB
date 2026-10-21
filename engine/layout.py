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
visuals and gaps keep shrinking. The floor is relaxed step by step (LADDER)
only when the copy can't fit even then. Buttons
that would get too small to read are re-wrapped around their label instead
of being scaled as a block.
"""

import math
from math import comb
from dataclasses import dataclass, field, replace
from itertools import combinations

from engine.analyze import Analysis, Unit, area, box_of, rotated_size, rows_of, union, xy_order
from engine.critic import MIN_HEADLINE, MIN_READABLE, collisions, unreadable, visible
from engine.textmeasure import measure_text

GAP_CAP = 0.06          # max source gap between copy items, as fraction of source height
MIN_GAP = 0.012         # min gap between copy items, as fraction of target short side
VISUAL_GAP = 0.04       # gap between hero visual and copy, fraction of short side
SIDE_OFFSET = 0.2       # visual/copy centres this far apart sideways (fraction of width) keep their sides
MIN_FONT = MIN_HEADLINE  # 9px: smallest font any reference uses; fitting won't go below it if it can help it


@dataclass(frozen=True, order=True)
class Floor:
    """The least font size text may shrink to, per role (Unit.role).
    Floors compare by strictness: the stricter one is the greater."""
    strictness: int
    headline: float = field(compare=False)
    action: float = field(compare=False)     # a standalone button's label
    text: float = field(compare=False)       # everything else

    def of(self, role: str) -> float:
        return getattr(self, role)

    def __bool__(self) -> bool:              # False only for "no floor"
        return self.headline > 0


def _ladder(*levels) -> tuple[Floor, ...]:
    return tuple(Floor(len(levels) - i, *lv) for i, lv in enumerate(levels))


# Floors tried strictest first: copy takes the first one it fits under.
LADDER = _ladder(
    (MIN_FONT, MIN_FONT, MIN_FONT),
    (MIN_FONT, MIN_READABLE, MIN_READABLE),
    # not all copy can be readable: the rest gives way first so the headline
    # and the call to action still read, then the headline down to 7px
    (MIN_FONT, MIN_READABLE, 5.0),
    (MIN_FONT, MIN_READABLE, 4.0),
    (MIN_FONT, MIN_READABLE, 3.0),
    (MIN_READABLE, MIN_READABLE, 3.0),
    (MIN_READABLE, MIN_READABLE, 2.0),
    (MIN_READABLE, MIN_READABLE, 1.5),        # the rest is unreadable by now anyway
    (0.0, 0.0, 0.0),                          # nothing fits: everything shrinks together
)
STRICTEST, NO_FLOOR = LADDER[0], LADDER[-1]


def font_scale(k: float, rel: float) -> float:
    """How much a font scales when the canvas area scales by k**2.

    Fitted on the hand-drawn references (504 pairs): big type tracks the
    canvas almost linearly, small type (relative to the largest font in the
    design, `rel` <= 1) much less so — small canvases flatten the hierarchy,
    big ones stretch it. Median error ~11% vs ~15% for plain k.
    """
    rel = min(max(rel, 1e-3), 1.0)
    return math.exp(-0.039) * k ** (0.858 + 0.222 * math.log(rel))


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
TRACKING_CAPS = (1.0, 0.5, 0.0)  # a column that can't fit tries tighter tracking before a lower floor


def fit_tracking(el: dict, font: float, width: float, cap: float = 1.0):
    """Largest tracking step (at most `cap`) whose drawn lines fit `width`.
    The renderer breaks lines on untracked widths and then draws them wider,
    so wrapped tracked text can't fit at any font size; loosen the tracking
    instead (short tracked labels on one line keep it all). Returns
    (el, measure)."""
    for t in (t for t in TRACKING_STEPS if t <= cap):
        e = tracked(el, font, t)
        m = measure(e, font, width)
        if m[0] <= width - 0.5:
            return e, m
    return e, m


def is_tracked(el: dict) -> bool:
    return bool(el.get("letter_spacing"))


def measure(el: dict, font: float, width: float) -> tuple[float, float, int]:
    """(text width, text height, lines) for el at this font size and box width.
    letter_spacing is taken as given - pass tracked(el, font) for a new size."""
    m = measure_text(el, font, width)
    return m["text_width"], m["text_height"], m["lines"]


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
        self.role = unit.role
        self.align = align
        self._cache: dict = {}
        self._styled: dict = {}     # (font, width, cap) -> el with fitted tracking
        self.track_cap = 1.0        # most tracking this item may keep (set by Column.solve)
        self.rot = self.el.get("rotation") or None

    def _drawn(self, w, h):
        """Bounds of a w x h box once rotated about its centre."""
        return rotated_size(w, h, self.rot) if self.rot else (w, h)

    def _fit(self, f: float, cw: float, floor: Floor = NO_FLOOR):
        if self.rot is None:
            return self._fit_box(f, cw, floor)
        # rotated: of the boxes whose rotated bounds fit the column width,
        # the one that is least tall once drawn (columns stack vertically)
        key = ("rot", round(f, 4), round(cw, 1), floor, self.track_cap)
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

    def _fit_box(self, f: float, cw: float, floor: Floor = NO_FLOOR):
        key = (round(f, 4), round(cw, 1), floor, self.track_cap)
        if key not in self._cache:
            need = floor.of(self.role)
            font = max(1.0, self.pref * f, need)
            if is_tracked(self.el):
                self._cache[key] = self._fit_tracked(font, cw)
                return self._cache[key]
            tw, th, _ = measure(self.el, font, cw)
            if tw > cw:                       # one unbreakable word too wide
                font = max(1.0, font * cw / tw * 0.98)
                if font < need and measure(self.el, need, cw)[0] <= cw:
                    font = need                   # the shrink overshot the floor
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
            el, (tw, th, _) = fit_tracking(self.el, font, w, self.track_cap)
            if tw <= w - 0.5 or font <= 1.0:
                break
            font = max(1.0, font * max(0.85, (w - 0.5) / tw * 0.98))
        self._styled[(font, w, self.track_cap)] = el
        return font, w, th + 0.15 * font

    def size(self, f, cw, floor=NO_FLOOR):
        _, w, h = self._fit(f, cw, floor)
        return self._drawn(w, h)

    def font_ok(self, f, cw, floor):
        return self._fit(f, cw, floor)[0] >= floor.of(self.role) - 1e-6

    def place(self, x, y, f, cw, floor=NO_FLOOR):
        font, w, h = self._fit(f, cw, floor)
        el = dict(self._styled.get((font, w, self.track_cap), self.el))
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
        self.role = unit.role

    def scale(self, f, cw, floor=NO_FLOOR):
        s = self.k * f
        need = floor.of(self.role)
        if self.small and need:
            s = max(s, need / self.small)
        return max(1e-3, min(s, cw / max(self.unit.w, 1e-6)))

    def size(self, f, cw, floor=NO_FLOOR):
        s = self.scale(f, cw, floor)
        return self.unit.w * s, self.unit.h * s

    def font_ok(self, f, cw, floor):
        return (not self.small
                or self.small * self.scale(f, cw, floor) >= floor.of(self.role) - 1e-6)

    def place(self, x, y, f, cw, floor=NO_FLOOR):
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
        self.track_cap = 1.0        # most tracking the label may keep (set by Column.solve)
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
        need = floor.of(self.role)
        if not self.compact and (not need or
                                 self.label["font_size"] * self.scale(f, cw, floor) >= need - 1e-6):
            return None
        key = (round(f, 4), round(cw, 1), floor, self.track_cap)
        if key not in self._cache:
            font = max(need, self.label["font_size"] * rigid, 1.0)
            px, py = self.pad_x * font, self.pad_y * font
            inner = max(cw - 2 * px, 1.0)
            lb, (tw, th, _) = fit_tracking(self.label, font, inner, self.track_cap)
            for _ in range(8):                # a word (or tracking) too wide
                if tw <= inner - 0.5 or font <= 1.0:
                    break
                font = max(1.0, font * max(0.85, (inner - 0.5) / tw * 0.98))
                px, py = self.pad_x * font, self.pad_y * font
                inner = max(cw - 2 * px, 1.0)
                lb, (tw, th, _) = fit_tracking(self.label, font, inner, self.track_cap)
            font = math.ceil(font * 100) / 100   # finalize rounds font_size
            lb, (tw, th, _) = fit_tracking(self.label, font, inner, self.track_cap)
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
        return super().font_ok(f, cw, floor) if r is None else r[0] >= floor.of(self.role) - 1e-6

    def size(self, f, cw, floor=NO_FLOOR):
        r = self._reflow(f, cw, floor)
        if r is None:
            return super().size(f, cw, floor)
        _, px, py, lw, lh = r
        return lw + 2 * px, lh + 2 * py

    def place(self, x, y, f, cw, floor=NO_FLOOR):
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
                el = dict(self._styled[(round(f, 4), round(cw, 1), floor, self.track_cap)])
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
            s = max(s, floor.text / self.small)
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
        g = 0.6 * floor.text if floor else max(self.gap * self.k * f, 2.0)
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

    def size(self, f, cw, floor=NO_FLOOR):
        per, s = self._fit(f, cw, floor)
        if per == 0:
            return self._pill_extent(f, cw, floor)
        w, h = self._extent(per)
        return w * s, h * s

    def font_ok(self, f, cw, floor):
        per, s = self._fit(f, cw, floor)
        if per == 0:
            return all(b.font_ok(f, cw, floor) for b in self.pills)
        return not self.small or self.small * s >= floor.text - 1e-6

    def place(self, x, y, f, cw, floor=NO_FLOOR):
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

    def __init__(self, unit: Unit, k: float):
        self.unit, self.k, self.min_f = unit, k, 0.0
        self.small, self.role = None, "text"


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


def ink_insets(item) -> tuple[float, float]:
    """(top, bottom) transparent margins of a rigid item, as fractions of its
    height: images whose visible pixels stop short of their box. Re-flowed
    text and re-wrapped rows have none."""
    if isinstance(item, (TextItem, LineItem)):
        return 0.0, 0.0
    b = _item_box(item)
    h = b[3] - b[1]
    if h <= 0:
        return 0.0, 0.0
    ib = union(visible(e) for e in item.unit.els)
    return max(0.0, (ib[1] - b[1]) / h), max(0.0, (b[3] - ib[3]) / h)


def left_inset(item) -> float:
    """An item's transparent left margin, as a fraction of its width: a logo
    (or a row led by one) whose pixels start well inside its box."""
    if isinstance(item, TextItem):
        return 0.0
    b = _item_box(item)
    w = b[2] - b[0]
    if w <= 0:
        return 0.0
    units = item.units if isinstance(item, LineItem) else [item.unit]
    return max(0.0, (union(visible(e) for u in units for e in u.els)[0] - b[0]) / w)


def source_gaps(items, src_h: float) -> list[float]:
    """Source gaps between consecutive items, measured between what is
    visible (an image's transparent margins don't count as space)."""
    gaps = []
    for a, b in zip(items, items[1:]):
        ta, ba = ink_insets(a)
        tb, _ = ink_insets(b)
        ha = _item_box(a)[3] - _item_box(a)[1]
        hb = _item_box(b)[3] - _item_box(b)[1]
        g = (_item_box(b)[1] + tb * hb) - (_item_box(a)[3] - ba * ha)
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
    floor: Floor = NO_FLOOR     # text in this column stays at or above it
    insets: list | None = None  # per item (top, bottom) transparent margins

    def heights(self, f, cw):
        """Item sizes and the gaps between them. A gap is the space between
        visible content: boxes may overlap by an image's transparent margin."""
        sizes = [it.size(f, cw, self.floor) for it in self.items]
        gaps = [max(g * self.k * f, self.min_gap * min(1.0, f * 2)) for g in self.gaps]
        if self.insets is None:
            self.insets = [ink_insets(it) for it in self.items]
        for i in range(len(gaps)):
            gaps[i] -= (self.insets[i][1] * sizes[i][1]
                        + self.insets[i + 1][0] * sizes[i + 1][1])
        return sizes, gaps

    def total(self, f, cw):
        sizes, gaps = self.heights(f, cw)
        return sum(h for _, h in sizes) + sum(gaps)

    def width(self, f, cw):
        return max((it.size(f, cw, self.floor)[0] for it in self.items), default=0.0)

    def solve(self, cw, ch, f_max=1.0, floors=LADDER):
        """Largest f <= f_max whose stacked height fits in ch.

        Everything shrinks together (hierarchy kept) while no text is at its
        floor. Text that reaches the floor stops there and the rest keeps
        shrinking. Floors are tried strictest first (LADDER); the first one
        the copy can fit under is kept in self.floor.
        """
        if not self.items:
            self.floor = floors[0]
            return f_max
        def ok(f):
            return self.total(f, cw) <= ch and (
                not self.floor or all(it.font_ok(f, cw, self.floor) for it in self.items))

        caps = TRACKING_CAPS if self._tracked() else (1.0,)
        done = False
        for floor in floors:
            self.floor = floor
            for cap in caps:            # tighter tracking before a lower floor
                self._set_cap(cap)
                if ok(f_max):
                    return f_max
                if floor and ok(1e-3):
                    done = True
                    break
            if done:
                break
            if floor:
                self._set_cap(1.0)
        if not self.floor:
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

    def _carriers(self):
        """Items whose tracking can be tightened (pills in a badge row too)."""
        for it in self.items:
            if hasattr(it, "track_cap"):
                yield it
            for b in getattr(it, "pills", None) or ():
                if hasattr(b, "track_cap"):
                    yield b

    def _tracked(self) -> bool:
        return any(is_tracked(getattr(it, "el", None) or getattr(it, "label", None) or {})
                   for it in self._carriers())

    def _set_cap(self, cap: float) -> None:
        for it in self._carriers():
            it.track_cap = cap

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
            if align == "center":
                ix = x + (cw - w) / 2
            else:   # left: line up what is visible with the text's edge
                ix = max(0.0, x - left_inset(it) * w)
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
    floor: Floor = STRICTEST    # strictest readability floor all copy fits under


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
            floor = STRICTEST if small * s < MIN_FONT else NO_FLOOR
            cw = W - 2 * m if floor else max(0.35 * W, min(W - 2 * m, u.w * s))
            f = s / max(b.k, 1e-6)
            button = (b, floor, f, cw)
            w, h = b.size(f, cw, floor)
        elif small and small * s < MIN_FONT:
            # too small to read at its corner share: grow it (a pill is
            # re-wrapped around a readable label) up to the canvas width
            cw = W - 2 * m
            if b is not None:
                for floor in LADDER[:2]:      # corner badges never go below 7px
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
    x0, y0, cw, ch = clear_of_bands(a, W, H, k, m, x0, y0, cw, ch)
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
    vis = VisualItem(a.visual, k * 1.15) if a.visual else None
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


def clear_of_bands(a: Analysis, W, H, k, m, x0, y0, cw, ch):
    """Narrow the copy column so it doesn't run across a full-height side
    panel that no copy sat on in the source, as long as at least 60% of its
    width is left. (Full-width bands such as a horizon are handled by
    snap_edges, which moves the band's edge instead of the copy.)"""
    W0, H0 = a.canvas
    copy = [e for u in a.flow for e in u.els]
    for band in a.edge:
        if any(area(_inter(box_of(e), box_of(band))) > 0 for e in copy):
            continue                      # copy sat on this band: it may stay on it
        b = _edge_place(band, W0, H0, W, H, k)
        bx0, by0, bx1, by1 = box_of(b)
        full_h = by0 <= 0.5 and by1 >= H - 0.5
        full_w = bx0 <= 0.5 and bx1 >= W - 0.5
        if full_h and not full_w:
            if bx0 <= 0.5 and x0 < bx1 < x0 + cw:            # left panel
                nx = bx1 + m
                if x0 + cw - nx >= 0.6 * cw:
                    cw, x0 = x0 + cw - nx, nx
            elif bx1 >= W - 0.5 and x0 < bx0 < x0 + cw:      # right panel
                if bx0 - m - x0 >= 0.6 * cw:
                    cw = bx0 - m - x0
    return x0, y0, cw, ch


def horizontal(a: Analysis, W, H, vis_frac: float | None = None, max_cols: int = 4) -> Plan:
    H0 = a.canvas[1]
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

    vis_left = _hero_left(a, flow)
    left, right = m, W - m
    if a.visual is not None:
        frac = vis_frac or (0.6 if r < 4 else 0.3)
        vh = H - 2 * m if not strip else H - 2 * max(2.0, 0.08 * H)
        s = min(vh / a.visual.h, frac * (W - 2 * m) / a.visual.w, 1.6 * k)
        vw = a.visual.w * s
        vy = (H - a.visual.h * s) / 2
        vx = left if vis_left else right - vw
        # stay clear of pinned corner items on the visual's side: fit between them
        lo, hi = m, H - m
        for e in corner_els:
            ex0, ey0, ex1, ey1 = box_of(e)
            if ex1 <= vx or ex0 >= vx + vw or ey1 <= vy or ey0 >= vy + a.visual.h * s:
                continue
            if (ey0 + ey1) / 2 < H / 2:
                lo = max(lo, ey1 + 0.5 * m)
            else:
                hi = min(hi, ey0 - 0.5 * m)
        if (lo, hi) != (m, H - m) and hi - lo > 0.4 * H:
            s = min(s, (hi - lo) / a.visual.h)
            vw = a.visual.w * s
            vy = lo + (hi - lo - a.visual.h * s) / 2
            vx = left if vis_left else right - vw
        els += place_rigid(a.visual, vx, vy, s)
        gap = VISUAL_GAP * min(W, H) + m * 0.5
        # the copy starts at the visual's visible edge, not its box edge
        vb, ib = a.visual.box, union(visible(e) for e in a.visual.els)
        bw = max(vb[2] - vb[0], 1e-6)
        if vis_left:
            left = vx + vw * (1 - max(0.0, vb[2] - ib[2]) / bw) + gap
        else:
            right = vx + vw * max(0.0, ib[0] - vb[0]) / bw - gap

    lead, main, action, outside = _split_columns(a, flow, strip, r)

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
    for units, share in ((lead, 0.18), (outside, 0.25), (action, 0.3)):
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
    floor = min(floors_used, default=STRICTEST)
    if floor < STRICTEST:
        # copy only fits as unreadable type: spread it over more columns
        units = flow                # every column's units, in reading order
        panel = frozenset(a.container_children)
        # only floors stricter than what the copy already has can win
        alt = strip_columns(units, left, right, ch_top, ch, gap, column, max_cols, panel,
                            levels=_stricter(floor))
        if alt is not None and alt[0] > floor:
            els = els[:copy_start] + alt[2]
            floor, f_main = alt[0], alt[1]
        if floor < STRICTEST:
            alt = two_rows(units, left, right, ch_top, ch, gap, column, max_cols, panel,
                           levels=_stricter(floor))
            if alt is not None and alt[0] > floor:
                els = els[:copy_start] + alt[2]
                floor, f_main = alt[0], alt[1]
    return Plan(els + corner_els, _quality(a, f_main, els, W, H), floor)


def _hero_left(a: Analysis, flow: list[Unit]) -> bool:
    """Whether the hero visual goes left of the copy: the side it sat on in
    the source - even when it was mostly above/below the copy, a clear
    sideways offset decides (above-right stays right); only a visual
    squarely above/below the copy leads (left) when it came first."""
    if a.visual is None or not flow:
        return True
    W0, H0 = a.canvas
    fb = union(u.box for u in flow)
    dx = (a.visual.cx - (fb[0] + fb[2]) / 2) / W0
    dy = (a.visual.cy - (fb[1] + fb[3]) / 2) / H0
    return dx <= 0 if abs(dx) >= min(abs(dy), SIDE_OFFSET) else dy <= 0


def _split_columns(a: Analysis, flow: list[Unit], strip: bool, r: float):
    """Split the copy (reading order) into the wide-format columns:
    (lead images - strips only, main copy, buttons - wide only, copy outside
    the text panel). Every unit lands in exactly one of them."""
    W0, H0 = a.canvas
    lead, main, action = [], [], []
    # a button inside a text panel stays in the panel's column (strips
    # excepted: there everything needs its own column)
    in_panel = a.container_children if a.container and not strip else set()
    first_button = next((i for i, u in enumerate(flow) if r >= 2.5 and u.is_button
                         and not any(e["id"] in in_panel for e in u.els)), None)
    def panel_side(u):
        return any(e["id"] in a.container_children for e in u.els)

    for i, u in enumerate(flow):
        if (first_button is not None and i >= first_button and first_button >= 2
                and panel_side(u) == panel_side(flow[first_button])):
            action.append(u)          # CTA and whatever followed it (same side of the panel)
        elif first_button == i:
            action.append(u)
        elif strip and not u.has_text and area(u.box) < 0.1 * W0 * H0:
            lead.append(u)
        else:
            main.append(u)
    if not main and action:
        main, action = action, []
    # copy outside the text panel must not share the panel's column (the
    # panel is wrapped around that column and would run under it): it gets
    # its own column at the far right
    outside = []
    if a.container is not None and any(panel_side(u) for u in main):
        outside = [u for u in main if not panel_side(u)]
        main = [u for u in main if panel_side(u)]
    return lead, main, action, outside


def two_rows(units, x0, x1, y, ch, gap, column, max_cols=4, panel=frozenset(),
             levels=LADDER[:-1]):
    """Headline across the full copy width, everything else in columns
    below it - for strips where the headline needs more width than any one
    column can give (long words, long lines). Same return as strip_columns."""
    head = next((i for i, u in enumerate(units) if u.primary), None)
    if head is None or len(units) < 2:
        return None
    rest = units[:head] + units[head + 1:]
    top = column([units[head]], None)
    w = x1 - x0
    row_gap = max(2.0, 0.04 * ch)

    def attempt(floor):
        if top.min_width(ch, floor, w) is None:
            return None
        lo, hi = 1.0, ch                      # least height the headline needs
        for _ in range(14):
            mid = (lo + hi) / 2
            if top.min_width(mid, floor, w) is not None:
                hi = mid
            else:
                lo = mid
        h1 = hi
        below = ch - h1 - row_gap
        if below <= 0:
            return None
        cols = strip_columns(rest, x0, x1, y + h1 + row_gap, below, gap, column, max_cols,
                             panel, levels=tuple(lv for lv in LADDER[:-1] if lv >= floor))
        if cols is None:
            return None
        f = top.solve(w, h1, 1.0, floors=(floor,))
        top.floor = floor
        out = top.place(x0, y, w, h1, f, "left") + cols[2]
        return floor, min(f, cols[1]), out

    return strictest(levels, attempt)


def _stricter(floor: Floor) -> tuple[Floor, ...]:
    return tuple(lv for lv in LADDER[:-1] if lv > floor)


def strictest(levels, attempt):
    """attempt(floor) at the strictest of `levels` (strictest first) where it
    doesn't return None. The original floors (9px, then 7px) are tried in
    turn; below them each level is looser for every role, so copy that fits
    one fits all looser ones and the graded levels are binary-searched."""
    for floor in (lv for lv in levels if lv >= LADDER[1]):
        r = attempt(floor)
        if r is not None:
            return r
    graded = [lv for lv in levels if lv < LADDER[1]]
    lo, hi, found = 0, len(graded) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        r = attempt(graded[mid])
        if r is not None:
            found, hi = r, mid - 1
        else:
            lo = mid + 1
    return found


EXHAUSTIVE_SPLITS = 2000   # above this many ways to split the copy, use the DP


def strip_columns(units, x0, x1, y, ch, gap, column, max_cols=4, panel=frozenset(),
                  levels=LADDER[:-1]):
    """Lay reading-ordered units out as up to max_cols side-by-side columns
    on a thin strip, choosing the split that keeps text at the strictest
    readability floor (of `levels`), then the largest type. Returns
    (floor, f, elements) or None if no split keeps text above the last floor.

    Splits are contiguous runs of the reading order. Few units: every split
    is tried. Many units: for each column count, a DP finds the split with
    the least total width (O(n^2 * columns) instead of combinatorial).
    A column never mixes copy inside the text panel (`panel`: element ids)
    with copy outside it: the panel is re-wrapped around its column(s) and
    would otherwise run under the outside copy."""
    n_units = len(units)
    cols_cache: dict = {}
    mw_cache: dict = {}
    in_panel = [any(e["id"] in panel for e in u.els) for u in units]

    def col(i, j):
        if (i, j) not in cols_cache:
            cols_cache[(i, j)] = column(units[i:j], None)
        return cols_cache[(i, j)]

    full = x1 - x0

    def mw(i, j, floor, avail):
        """Least width units[i:j] need as one column (None: not within avail).
        Computed once against the full strip width, whatever the column count."""
        key = (i, j, floor)
        if key not in mw_cache:
            mixed = len(set(in_panel[i:j])) > 1
            mw_cache[key] = None if mixed else col(i, j).min_width(ch, floor, full)
        w = mw_cache[key]
        return w if w is not None and w <= avail else None

    def splits(n, floor, avail):
        """Candidate splits into n columns, as bounds tuples. The graded
        floors (a last resort) always take the DP's least-width split."""
        if (floor >= LADDER[1] and sum(comb(n_units - 1, k) for k in range(min(max_cols, n_units)))
                <= EXHAUSTIVE_SPLITS):
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
                        break           # longer runs only get taller (or stay mixed)
                    if best[k - 1][i] + w < best[k][j]:
                        best[k][j], back[k][j] = best[k - 1][i] + w, i
        if best[n][n_units] == INF:
            return
        bounds, j = [n_units], n_units
        for k in range(n, 0, -1):
            j = back[k][j]
            bounds.append(j)
        yield tuple(reversed(bounds))

    def attempt(floor):
        """The best split at this floor, over every column count."""
        best = None
        for n in range(1, min(max_cols, n_units) + 1):
            avail = x1 - x0 - gap * (n - 1)
            if avail <= 0:
                break
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
                cols = [col(i, j) for i, j in pairs]
                extra = (avail - sum(mins)) / n
                widths = [w + extra for w in mins]
                fs = [c.solve(w, ch, 1.0, floors=(floor,)) for c, w in zip(cols, widths)]
                key = (floor, min(fs), -n)
                if best is None or key > best[0]:
                    best = (key, cols, widths, fs)
        return best

    best = strictest(levels, attempt)
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
    x = x0
    out = []
    for c, w, u, f in zip(cols, widths, used, fs):
        out += c.place(x, y, w, ch, f, "left")   # fitted at w: same line breaks
        x += u + g
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
    pad = 0.04 * min(W, H) + 2
    x0, y0 = max(0.0, kb[0] - pad), max(0.0, kb[1] - pad)
    x1, y1 = min(float(W), kb[2] + pad), min(float(H), kb[3] + pad)
    # the padding must not run under copy that isn't part of the panel (a
    # caption, a corner badge): trim the side facing it
    kid_ids = {e["id"] for e in kids}
    text_groups = {e.get("group_id") for e in placed.values()
                   if e["type"] == "text" and e.get("group_id")}
    for p in placed.values():
        if p["id"] in kid_ids or p["id"] == src["id"] or not (
                p["type"] == "text" or p.get("group_id") in text_groups - {None}):
            continue
        bx0, by0, bx1, by1 = box_of(p)
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        if by1 <= kb[1]:
            y0 = max(y0, by1)
        elif by0 >= kb[3]:
            y1 = min(y1, by0)
        elif bx1 <= kb[0]:
            x0 = max(x0, bx1)
        elif bx0 >= kb[2]:
            x1 = min(x1, bx0)
    el = dict(src, x=x0, y=y0, width=x1 - x0, height=y1 - y0)
    if "corner_radius" in src:
        el["corner_radius"] = min(src["corner_radius"] * k, 0.5 * min(x1 - x0, y1 - y0))
    return el


def _straddles(lo, hi, edge, pad=1.0) -> bool:
    return lo + pad < edge < hi - pad


def snap_edges(a: Analysis, els: list[dict], W, H) -> None:
    """Copy must sit fully on or fully off an edge band (horizon, side
    panel), never across the band's free edge. Move that edge (the shape,
    never the copy) so each straddling item lands on the side it was on in
    the source; give up on a band if that would leave it under 35% of its
    size or flip another item's side."""
    if not a.edge:
        return
    placed = {e["id"]: e for e in els}
    groups_with_text = {e.get("group_id") for e in els if e["type"] == "text" and e.get("group_id")}
    src = {e["id"]: e for u in a.flow + [u for u, _ in a.corner] + a.visual_parts for e in u.els}
    src.update({e["id"]: e for e in a.edge})
    copy_ids = [i for i, e in src.items() if i in placed and (
        e["type"] == "text" or (e.get("group_id") and e.get("group_id") in groups_with_text))]
    pad = 0.02 * min(W, H)

    def on(c_id, band_id, el_by_id):
        return area(_inter(box_of(el_by_id[c_id]), box_of(el_by_id[band_id]))) > 0

    for band in a.edge:
        pb = placed.get(band["id"])
        if pb is None:
            continue
        x0, y0, x1, y1 = box_of(pb)
        tol = 0.5
        full_w = x0 <= tol and x1 >= W - tol
        full_h = y0 <= tol and y1 >= H - tol
        if full_w == full_h:
            continue                      # a corner block or a full cover: no single free edge
        lo_i, hi_i = (1, 3) if full_w else (0, 2)
        glued_lo = (y0 if full_w else x0) <= tol
        edge = (y1 if full_w else x1) if glued_lo else (y0 if full_w else x0)
        size = (y1 - y0) if full_w else (x1 - x0)
        want_on = {c: on(c, band["id"], src) for c in copy_ids}
        straddlers = [c for c in copy_ids
                      if _straddles(box_of(placed[c])[lo_i], box_of(placed[c])[hi_i], edge)]
        if not straddlers:
            continue
        new = edge
        for c in straddlers:
            b = box_of(placed[c])
            if glued_lo:
                new = max(new, b[hi_i] + pad) if want_on[c] else min(new, b[lo_i] - pad)
            else:
                new = min(new, b[lo_i] - pad) if want_on[c] else max(new, b[hi_i] + pad)
        new_size = (new - (y0 if full_w else x0)) if glued_lo else ((y1 if full_w else x1) - new)
        limit = H if full_w else W
        if not (0.35 * size <= new_size <= limit):
            continue
        trial = dict(pb)
        if full_w:
            trial.update(y=0.0 if glued_lo else new, height=new_size)
        else:
            trial.update(x=0.0 if glued_lo else new, width=new_size)
        check = dict(placed, **{band["id"]: trial})
        if all(on(c, band["id"], check) == want_on[c] or c not in straddlers
               and on(c, band["id"], check) == on(c, band["id"], placed) for c in copy_ids):
            pb.update(trial)


def _inter(a, b):
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def detach(a: Analysis, parts: list[Unit]) -> Analysis:
    """Move these text-bearing units off the hero visual into the copy
    (reading order kept), so they are sized on their own."""
    ids = {e["id"] for u in parts for e in u.els}
    rest = [e for e in a.visual.els if e["id"] not in ids]
    if not parts or not any(e["type"] != "text" for e in rest):
        return a
    visual = Unit(rest)
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

    return alt if unreadable(a.elements, alt) < unreadable(a.elements, out) else out


def _preferred(a: Analysis, W: int, H: int) -> Plan:
    """The arrangement the rules pick for this target shape."""
    r = W / H
    if r < 0.9:
        return vertical(a, W, H)
    if r > 1.25:
        return horizontal(a, W, H)
    if a.visual is None:
        return vertical(a, W, H)
    # Near-square: centred designs stack, left-aligned designs go side by
    # side (what the reference designers did).
    W0, H0 = a.canvas
    if W0 / H0 <= 1.3:
        return vertical(a, W, H) if a.centered else horizontal(a, W, H)
    # wide sources often left-align even centred designs: compare
    v, h = vertical(a, W, H), horizontal(a, W, H)
    return v if v.quality >= h.quality else h


def _candidates(a: Analysis, W: int, H: int):
    yield vertical(a, W, H)
    if a.corner:
        yield vertical(a, W, H, fold_corners=True)
    yield horizontal(a, W, H)
    if W / H > 1.25:
        # a strip too small for the copy: give the hero visual less of the
        # width and allow more columns
        for frac in (0.18, 0.1):
            yield horizontal(a, W, H, vis_frac=frac, max_cols=6)


def _layout(a: Analysis, W: int, H: int) -> list[dict]:
    """Generate and rank: the preferred arrangement, unless its copy breaks
    the readability floor, leaves text unreadable or collides with
    something - then every candidate is finished and ranked by (floor,
    fewest unreadable texts, fewest collisions, preferred, quality)."""
    def rank(plan, out, preferred):
        return (plan.floor, -unreadable(a.elements, out),
                -len(collisions(a.elements, out, W, H)), preferred, plan.quality)

    plan = _preferred(a, W, H)
    out = _finish(a, plan, W, H)
    best = (rank(plan, out, 1), out)
    _, n_small, n_bumps, _, _ = best[0]
    if plan.floor == STRICTEST and n_small == 0 and n_bumps == 0:
        return out
    for alt in _candidates(a, W, H):
        alt_out = _finish(a, alt, W, H)
        key = rank(alt, alt_out, 0)
        if key > best[0]:
            best = (key, alt_out)
    return best[1]


def _finish(a: Analysis, plan: Plan, W: int, H: int) -> list[dict]:
    """A plan plus everything placed around it: background, edge bands and
    decorations, the text panel re-wrapped, edge bands snapped clear of copy."""
    k = _k(a, W, H)
    out = place_free(a, W, H, k) + [dict(e) for e in plan.elements]
    placed = {e["id"]: e for e in out}
    cont = wrap_container(a, placed, W, H, k)
    if cont is not None:
        out.append(cont)
    snap_edges(a, out, W, H)
    return out
