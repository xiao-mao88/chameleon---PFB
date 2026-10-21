"""
Engine tests on designs and sizes the engine has never seen.

    python -m unittest discover tests            # everything
    python -m unittest tests.test_engine -k strip

Sources: the 3 official samples, every hand-drawn reference design (at its
square format), and generated "novel" designs (opaque ids, reversed
z-order, a single element, text only, no text, 30+ elements, rotation,
run-on copy, a missing asset, labels longer than their pills, letter-spacing,
40 text boxes, a photo grid, a menu, two heroes, text on a photo, Unicode,
bleed, zero-size and empty elements). Targets: a catalogue of real platform
formats, most of them not in the reference set. There is no answer to
compare against, so these check what must hold for any design at any size:

  1. the RULES.md hard checks (ids, fields, canvas, on-canvas, images in
     proportion, content / font_family / asset_id untouched)
  2. ellipses keep their proportions
  3. no rendered text overflow
  4. readability: headline >= 9px, all text >= 7px - unless the copy
     physically can't fit the canvas at those sizes (see copy_fits)
  5. no collisions: text newly over other text, or over an image or solid
     shape it didn't overlap in the source (tools.evaluate.collision_checks)
  6. no crash and no fallback to the uniform fit
  7. determinism
  8. rotated elements stay on canvas as drawn (rotated about their centre)
  9. every case finishes in under MAX_SECONDS
 10. contrast (sampled: it renders): no text the layout pushed below WCAG AA
"""

import copy
import json
import math
import time
import unittest
from functools import lru_cache
from pathlib import Path
from unittest import mock

from engine.critic import MIN_HEADLINE as MIN_PRIMARY, MIN_READABLE, key_copy_unreadable
from engine.finalize import extent
from engine.resize import resize
from engine.textmeasure import measure_text
from tools.dataset import ROOT, drawings
from tools.evaluate import (collision_checks, contrast_checks, hard_checks, overflow_checks,
                            readability_checks)

INPUT_DIR = ROOT / "input"

# Real-world formats (RULES.md: hidden targets come from a catalogue like this)
CATALOGUE = {
    "instagram_post": (1080, 1080),
    "instagram_portrait": (1080, 1350),
    "instagram_story": (1080, 1920),
    "facebook_link": (1200, 628),
    "x_post": (1600, 900),
    "x_header": (1500, 500),
    "youtube_thumbnail": (1280, 720),
    "pinterest_pin": (1000, 1500),
    "medium_rectangle": (300, 250),
    "half_page": (300, 600),
    "wide_skyscraper": (160, 600),
    "leaderboard": (728, 90),
    "mobile_banner": (320, 50),
    "large_mobile_banner": (320, 100),
    "billboard": (970, 250),
    "print_poster": (2400, 3600),
    "instagram_grid": (1080, 1440),
    "facebook_cover": (851, 315),
    "linkedin_banner": (1584, 396),
    "linkedin_company_cover": (4200, 700),
    "youtube_channel_art": (2560, 1440),
    "small_rectangle": (180, 150),
    "full_banner": (468, 60),
    "skyscraper": (120, 600),
    "super_leaderboard": (970, 90),
    "app_icon_tile": (100, 100),
}

MAX_SECONDS = 10.0
CONTRAST_EVERY = 13     # render every 13th case for the contrast test


# Readability cases the engine is known to miss, with the reason. Listed
# here so they show in the test output instead of being skipped silently;
# test_known_limitations fails once one is fixed, so the list stays honest.
KNOWN_LIMITATIONS = {
    ("novel_thirty_elements", "mobile_banner"):
        "10 chips + headline + CTA in 320x50: the strip planner can't fit the "
        "chip grid at 7px next to the copy",
    ("novel_thirty_elements", "app_icon_tile"):
        "10 chips + headline + CTA in a 100x100 tile: the pills' padding alone "
        "outgrows the canvas at 7px",
    ("novel_rotated_all", "mobile_banner"):
        "a 30°-rotated headline in 320x50: at 9px its rotated bounds are 42px "
        "tall at best, the strip has 40px",
}

# Cases where even the headline and standalone buttons can't reach 7px.
_TILE = ("in a 100x100 tile the headline and CTA alone need 48% of the canvas "
         "at 7px, before the visual, badges and the rest of the copy")
_RUN_ON = "one unbreakable word is 112px wide at 7px"
KEY_COPY_LIMITS = {
    ("novel_letter_spacing", "app_icon_tile"): _TILE,
    ("novel_missing_asset", "app_icon_tile"): _TILE,
    ("sample_level3", "app_icon_tile"): _TILE,
    ("novel_run_on_copy", "app_icon_tile"): _RUN_ON + ": the tile is 100px wide",
    ("novel_run_on_copy", "skyscraper"): _RUN_ON + ": the strip is 120px wide with margins",
}


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------

def _sample(level: str) -> dict:
    path = next(INPUT_DIR.glob(f"chameleon_sample_{level}*.json"))
    return json.loads(path.read_text())["source"]


def _text(id, x, y, w, h, content, size, **kw) -> dict:
    return {"id": id, "type": "text", "x": x, "y": y, "width": w, "height": h,
            "content": content, "font_family": kw.pop("font_family", "Solstice Sans"),
            "font_size": size, "color": "rgba(255,255,255,1)", **kw}


def _shape(id, x, y, w, h, **kw) -> dict:
    return {"id": id, "type": "shape", "x": x, "y": y, "width": w, "height": h,
            "shape_type": kw.pop("shape_type", "rectangle"),
            "fill_color": kw.pop("fill_color", "rgba(232,106,58,1)"), **kw}


def _image(id, x, y, w, h, asset, **kw) -> dict:
    return {"id": id, "type": "image", "x": x, "y": y, "width": w, "height": h,
            "asset_id": asset, **kw}


def _with_z(elements: list[dict]) -> list[dict]:
    for z, e in enumerate(elements):
        e.setdefault("z_index", z)
    return elements


def novel_designs() -> dict[str, dict]:
    out = {}

    src = _sample("level2")
    ids = {e["id"]: f"el_{n:02d}" for n, e in enumerate(src["elements"])}
    groups = {g: f"g_{n}" for n, g in enumerate(
        sorted({e["group_id"] for e in src["elements"] if e.get("group_id")}))}
    for e in src["elements"]:
        e["id"] = ids[e["id"]]
        if e.get("group_id"):
            e["group_id"] = groups[e["group_id"]]
    out["opaque_ids"] = src

    src = _sample("level2")
    src["elements"] = src["elements"][::-1]
    out["reversed_order"] = src

    out["single_image"] = {"canvas": {"width": 1080, "height": 1080}, "elements": _with_z([
        _image("hero", 0, 0, 1080, 1080, "solstice_bg_gradient.png", fit="cover")])}

    out["text_only"] = {"canvas": {"width": 1080, "height": 1350}, "elements": _with_z([
        _shape("bg", 0, 0, 1080, 1350, fill_color="rgba(40,24,48,1)"),
        _text("kicker", 120, 260, 840, 40, "SUMMER 2026", 28, text_align="center"),
        _text("title", 120, 330, 840, 260, "Chase the Light", 110,
              font_family="Solstice Serif", text_align="center"),
        _text("dek", 160, 640, 760, 120, "Botanical sparkling infusions, crafted for "
              "golden hour.", 36, text_align="center"),
        _text("fine", 160, 1180, 760, 60, "Available at select retailers.", 20,
              text_align="center")])}

    out["no_text"] = {"canvas": {"width": 1080, "height": 1080}, "elements": _with_z([
        _image("bg", 0, 0, 1080, 1080, "solstice_bg_gradient.png", fit="cover"),
        _image("bottle", 390, 200, 300, 600, "solstice_bottle.png", fit="contain"),
        _image("logo", 60, 60, 240, 80, "solstice_logo_lockup.png", fit="contain"),
        _shape("sun", 760, 120, 200, 200, shape_type="ellipse",
               fill_color="rgba(242,183,64,1)")])}

    els = [_image("bg", 0, 0, 1080, 1080, "solstice_bg_gradient.png", fit="cover"),
           _text("headline", 80, 80, 920, 120, "Thirty Things at Once", 72,
                 font_family="Solstice Serif")]
    for i in range(10):
        x, y = 80 + (i % 5) * 190, 300 + (i // 5) * 90
        els += [_shape(f"chip{i}_bg", x, y, 170, 60, corner_radius=30, group_id=f"chip{i}"),
                _text(f"chip{i}_label", x, y + 16, 170, 28, f"OPTION {i}", 20,
                      font_weight="bold", text_align="center", group_id=f"chip{i}")]
    for i in range(8):
        els.append(_shape(f"dot{i}", 100 + i * 115, 700 + (i % 3) * 60, 40, 40,
                          shape_type="ellipse", opacity=0.3))
    els += [_shape("cta_bg", 340, 920, 400, 90, corner_radius=45, group_id="cta"),
            _text("cta_label", 340, 945, 400, 40, "Shop now", 30, font_weight="bold",
                  text_align="center", group_id="cta")]
    out["thirty_elements"] = {"canvas": {"width": 1080, "height": 1080},
                              "elements": _with_z(els)}

    src = _sample("level1")
    for e in src["elements"]:
        if e["id"] == "logo":
            e["rotation"] = 15
    out["rotated"] = src

    src = _sample("level2")
    for e in src["elements"]:
        if e["type"] == "text" and e["font_size"] == max(
                t["font_size"] for t in src["elements"] if t["type"] == "text"):
            e["content"] = ("Chase the Light with Supercalifragilisticexpialidocious "
                            "Botanical Sparkling Infusions All Summer Long")
            e.pop("style_runs", None)
    out["run_on_copy"] = src

    out["missing_asset"] = _sample("level3")   # holiday_watermark has no file

    src = _sample("level2")                    # translated / run-on labels
    for e in src["elements"]:
        if e["type"] == "text" and e.get("group_id"):
            e["content"] = (e["content"] + " ") * 3
    out["long_labels"] = src

    src = _sample("level3")                    # tracked-out type everywhere
    for e in src["elements"]:
        if e["type"] == "text":
            e["letter_spacing"] = 8
    out["letter_spacing"] = src

    out["forty_texts"] = {"canvas": {"width": 1080, "height": 1080}, "elements": _with_z(
        [_shape("bg", 0, 0, 1080, 1080, fill_color="rgba(40,24,48,1)")]
        + [_text(f"item{i}", 80, 20 + i * 25, 920, 22, f"Item {i} of the long list", 16)
           for i in range(40)])}

    out["photo_grid"] = {"canvas": {"width": 1080, "height": 1080}, "elements": _with_z(
        [_image("bg", 0, 0, 1080, 1080, "solstice_bg_gradient.png", fit="cover")]
        + [_image(f"tile{i}", 60 + (i % 3) * 340, 60 + (i // 3) * 340, 300, 300,
                  "solstice_bottle.png") for i in range(9)])}

    out["menu"] = {"canvas": {"width": 1080, "height": 1350}, "elements": _with_z(
        [_shape("bg", 0, 0, 1080, 1350, fill_color="rgba(40,24,48,1)"),
         _text("title", 80, 60, 920, 120, "Summer Menu", 96, font_family="Solstice Serif")]
        + [_text(f"dish{i}", 80, 240 + i * 105, 920, 60, f"Dish number {i} with seasonal "
                 "botanicals", 36) for i in range(10)])}

    out["two_heroes"] = {"canvas": {"width": 1080, "height": 1080}, "elements": _with_z([
        _image("left", 0, 0, 540, 1080, "solstice_bottle.png", fit="cover"),
        _image("right", 540, 0, 540, 1080, "solstice_bottle.png", fit="cover"),
        _text("title", 100, 480, 880, 120, "Two of a Kind", 80, font_family="Solstice Serif",
              text_align="center")])}

    out["text_on_photo"] = {"canvas": {"width": 1080, "height": 1350}, "elements": _with_z([
        _image("photo", 0, 0, 1080, 1350, "solstice_bg_gradient.png", fit="cover"),
        _text("caption", 80, 1100, 920, 140, "Golden hour, bottled", 64,
              font_family="Solstice Serif")])}

    out["unicode"] = {"canvas": {"width": 1080, "height": 1080}, "elements": _with_z([
        _shape("bg", 0, 0, 1080, 1080, fill_color="rgba(40,24,48,1)"),
        _text("title", 100, 380, 880, 200, "Ünïcödé — 你好 — مرحبا ✨", 64, text_align="center"),
        _text("sub", 100, 620, 880, 80, "Crème brûlée · Façade · Smörgåsbord", 32,
              text_align="center")])}

    src = _sample("level2")                    # elements bleeding off the canvas
    for e in src["elements"][1:4]:
        e["x"] -= 200
    out["bleed"] = src

    src = _sample("level2")                    # degenerate elements
    texts = [e for e in src["elements"] if e["type"] == "text"]
    texts[-1]["content"] = ""
    shapes = [e for e in src["elements"] if e["type"] == "shape"]
    shapes[0]["width"] = shapes[0]["height"] = 0
    out["degenerate"] = src

    src = _sample("level1")                    # everything but the bg rotated
    for e in src["elements"]:
        if e.get("fit") != "cover":
            e["rotation"] = 30
    out["rotated_all"] = src
    return out


@lru_cache(maxsize=1)
def sources() -> dict[str, dict]:
    out = {f"sample_{lvl}": _sample(lvl) for lvl in ("level1", "level2", "level3")}
    for d in drawings():
        if d.format == "square":
            out[f"ref_{d.design}"] = d.load()
    for name, src in novel_designs().items():
        out[f"novel_{name}"] = src
    return out


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def run(source: dict, W: int, H: int) -> tuple[dict, bool]:
    """resize() on a copy of source; also reports whether it fell back."""
    with mock.patch("engine.resize.traceback.print_exc") as fallback:
        out = resize(copy.deepcopy(source), {"width": W, "height": H})
    return out, fallback.called


def rotated_off_canvas(output: dict) -> list[str]:
    """Rotated elements whose drawn bounds (rotated about the centre) leave
    the canvas - the unrotated box can be on canvas while the drawing isn't."""
    W, H = output["canvas"]["width"], output["canvas"]["height"]
    out = []
    for e in output["elements"]:
        if not e.get("rotation"):
            continue
        w, h = extent(e)
        cx, cy = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2
        over = max(w / 2 - cx, cx + w / 2 - W, h / 2 - cy, cy + h / 2 - H)
        if over > 0.5:
            out.append(f"{e['id']}: rotated bounds {over:.1f}px off canvas")
    return out


def copy_fits(source: dict, W: int, H: int) -> bool:
    """Can the copy physically fit at the readability floors? Area check:
    every text set on one line at its floor (headline at MIN_PRIMARY), each
    line given 1.2x its font in height and a 15% wrapping allowance, must
    take at most half the canvas (the rest is the visual, buttons, gaps);
    and its longest word must fit across (and, rotated, within) the canvas."""
    texts = [e for e in source["elements"] if e["type"] == "text" and e.get("content")]
    if not texts:
        return True
    head = max(texts, key=lambda e: e["font_size"])["id"]
    need = 0.0
    for e in texts:
        e = dict(e, letter_spacing=0)   # the engine may loosen tracking to fit
        size = MIN_PRIMARY if e["id"] == head else MIN_READABLE
        line = measure_text(e, size, 1e6)["text_width"]
        need += 1.15 * line * size * float(e.get("line_height", 1.2))
        # and no single word may be wider than the canvas at that size -
        # or, once rotated, taller than it
        word = max(str(e["content"]).split(), key=len, default="")
        ww = measure_text(dict(e, content=word), size, 1e6)["text_width"]
        r = math.radians(float(e.get("rotation") or 0.0))
        lh = size * float(e.get("line_height", 1.2))
        if (ww * abs(math.cos(r)) + lh * abs(math.sin(r)) > 0.9 * W
                or ww * abs(math.sin(r)) + lh * abs(math.cos(r)) > 0.9 * H):
            return False
    return need <= 0.5 * W * H


def ellipse_checks(source: dict, output: dict) -> list[str]:
    src = {e["id"]: e for e in source["elements"]}
    out = []
    for e in output["elements"]:
        s = src[e["id"]]
        if (e["type"] == "shape" and s.get("shape_type") == "ellipse"
                and min(s["width"], s["height"], e["width"], e["height"]) > 0):
            ratio = (e["width"] / e["height"]) / (s["width"] / s["height"])
            if abs(ratio - 1) > 0.05:
                out.append(f"{e['id']}: ellipse aspect off by {abs(ratio - 1):.0%}")
    return out


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

class EngineTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.results, cls.seconds = {}, {}
        for sname, src in sources().items():
            for tname, (W, H) in CATALOGUE.items():
                t0 = time.perf_counter()
                cls.results[(sname, tname)] = (src, W, H) + run(src, W, H)
                cls.seconds[(sname, tname)] = time.perf_counter() - t0

    def each(self, check, skip=()):
        """Run check(src, W, H, out, fell_back) on every (source, target)."""
        for (sname, tname), (src, W, H, out, fell_back) in self.results.items():
            if (sname, tname) in skip:
                continue
            with self.subTest(source=sname, target=f"{tname} {W}x{H}"):
                check(src, W, H, out, fell_back)

    def test_hard_rules(self):
        def check(src, W, H, out, _):
            problems = hard_checks(src, out, {"width": W, "height": H})
            self.assertEqual(problems + ellipse_checks(src, out), [])
        self.each(check)

    def test_no_text_overflow(self):
        self.each(lambda src, W, H, out, _: self.assertEqual(overflow_checks(out), []))

    def test_readable(self):
        def check(src, W, H, out, _):
            if copy_fits(src, W, H):    # else exempt: see test_exemptions_are_few
                self.assertEqual(readability_checks(src, out), [])
        self.each(check, skip=KNOWN_LIMITATIONS)

    def test_known_limitations(self):
        print(f"\nknown readability limitations ({len(KNOWN_LIMITATIONS)}):")
        for (sname, tname), why in KNOWN_LIMITATIONS.items():
            print(f"  {sname} -> {tname}: {why}")
            src, W, H, out, _ = self.results[(sname, tname)]
            with self.subTest(source=sname, target=tname):
                self.assertNotEqual(readability_checks(src, out), [],
                                    "fixed - remove it from KNOWN_LIMITATIONS")

    def test_key_copy_readable(self):
        """Even where not all copy can be readable, the headline and
        standalone buttons are: the rest of the copy gives way first."""
        def check(src, W, H, out, _):
            self.assertEqual(key_copy_unreadable(src, out["elements"]), [])
        self.each(check, skip=KEY_COPY_LIMITS)

    def test_logos_visible(self):
        """A lone image in the copy or a corner (a logo) stays recognisable -
        at least 18px of visible height - wherever the canvas has room
        (short side 250px or more). Smaller formats give text priority."""
        from engine.analyze import analyze
        from engine.critic import visible

        def check(src, W, H, out, _):
            if min(W, H) < 250:
                return
            a = analyze(src)
            by = {e["id"]: e for e in out["elements"]}
            for u in a.flow + [u for u, _ in a.corner]:
                if u.role == "mark":
                    vb = visible(by[u.els[0]["id"]])
                    self.assertGreaterEqual(vb[3] - vb[1], 18, u.els[0]["id"])
        self.each(check)

    def test_key_copy_limits(self):
        print(f"\nheadline/CTA limitations ({len(KEY_COPY_LIMITS)}):")
        for (sname, tname), why in KEY_COPY_LIMITS.items():
            print(f"  {sname} -> {tname}: {why}")
            src, W, H, out, _ = self.results[(sname, tname)]
            with self.subTest(source=sname, target=tname):
                self.assertNotEqual(key_copy_unreadable(src, out["elements"]), [],
                                    "fixed - remove it from KEY_COPY_LIMITS")

    def test_exemptions_are_few(self):
        """The readability exemption must stay the exception: only tiny
        formats, and only copy-heavy designs."""
        exempt = sorted({(s, t) for (s, t), (src, W, H, *_rest) in self.results.items()
                         if not copy_fits(src, W, H)})
        print(f"\nreadability exemptions ({len(exempt)}):",
              *(f"  {s} -> {t}" for s, t in exempt), sep="\n")
        self.assertTrue(all(CATALOGUE[t][0] * CATALOGUE[t][1] <= 40_000 or CATALOGUE[t][0] <= 320
                            for _, t in exempt),
                        "only tiny (< 40k px²) or narrow (<= 320px wide) formats may be exempt")
        self.assertLessEqual(len(exempt), 0.05 * len(self.results))

    def test_no_collisions(self):
        def check(src, W, H, out, _):
            if copy_fits(src, W, H):    # copy that can't fit is best effort
                self.assertEqual(collision_checks(src, out), [])
        self.each(check)

    def test_contrast_sampled(self):
        for (sname, tname), (src, W, H, out, _) in list(self.results.items())[::CONTRAST_EVERY]:
            with self.subTest(source=sname, target=tname):
                self.assertEqual(contrast_checks(out, src), [])

    def test_no_fallback(self):
        self.each(lambda src, W, H, out, fell_back: self.assertFalse(
            fell_back, "resize() raised and used the uniform fit"))

    def test_layout_places_everything(self):
        """layout() itself must place every source element: finalize() falls
        back to the source geometry for a missing one, which passes the hard
        checks but lands the element wherever it happened to be."""
        from engine.analyze import analyze
        from engine.layout import layout
        for (sname, tname), (src, W, H, *_rest) in list(self.results.items())[::3]:
            with self.subTest(source=sname, target=tname):
                placed = {e["id"] for e in layout(analyze(copy.deepcopy(src)), W, H)}
                self.assertEqual(sorted({e["id"] for e in src["elements"]} - placed), [])

    def test_rotated_on_canvas(self):
        self.each(lambda src, W, H, out, _: self.assertEqual(rotated_off_canvas(out), []))

    def test_fast(self):
        slow = {k: round(v, 1) for k, v in self.seconds.items() if v > MAX_SECONDS}
        print(f"\nslowest case: {max(self.seconds.values()):.2f}s, "
              f"total {sum(self.seconds.values()):.0f}s for {len(self.seconds)} cases")
        self.assertEqual(slow, {})

    def test_deterministic(self):
        for (sname, tname), (src, W, H, out, _) in list(self.results.items())[::7]:
            with self.subTest(source=sname, target=tname):
                again, _ = run(src, W, H)
                self.assertEqual(json.dumps(out, sort_keys=True),
                                 json.dumps(again, sort_keys=True))

    def test_source_untouched(self):
        src = sources()["sample_level3"]
        before = json.dumps(src, sort_keys=True)
        resize(src, {"width": 320, "height": 50})
        self.assertEqual(json.dumps(src, sort_keys=True), before)



class PartsTest(unittest.TestCase):
    """The engine's building blocks, on hand-made cases."""

    def test_ink_box(self):
        from engine.assets import ink_box
        bottle = {"type": "image", "x": 0, "y": 0, "width": 800, "height": 800,
                  "asset_id": "solstice_bottle.png"}
        self.assertEqual(tuple(round(v) for v in ink_box(bottle)), (230, 130, 571, 681))
        # contain in a wider box: the image is centred, its pixels move with it
        wide = dict(bottle, width=1000, fit="contain")
        self.assertEqual(tuple(round(v) for v in ink_box(wide)), (330, 130, 671, 681))
        self.assertEqual(ink_box(dict(bottle, fit="cover")), (0, 0, 800, 800))
        self.assertEqual(ink_box(dict(bottle, asset_id="no_such_file.png")), (0, 0, 800, 800))
        text = {"type": "text", "x": 5, "y": 6, "width": 7, "height": 8}
        self.assertEqual(ink_box(text), (5, 6, 12, 14))

    def test_collisions_use_visible_pixels(self):
        from engine.critic import collisions
        logo = {"id": "logo", "type": "image", "x": 0, "y": 0, "width": 900, "height": 300,
                "asset_id": "solstice_logo_lockup.png", "z_index": 1}
        text = {"id": "t", "type": "text", "content": "Hi", "x": 0, "y": 0, "width": 200,
                "height": 60, "z_index": 2}
        source = [dict(logo), dict(text, y=400)]
        # the logo's left 29% is transparent: text there touches nothing
        self.assertEqual(collisions(source, [logo, dict(text, x=10, y=100)], 900, 500), [])
        hit = collisions(source, [logo, dict(text, x=350, y=100)], 900, 500)
        self.assertEqual(len(hit), 1)
        self.assertIn("t collides with logo", hit[0])

    def test_rank_avoids_a_colliding_preferred_plan(self):
        """If the rules' preferred arrangement collides, a collision-free
        candidate wins."""
        from engine import layout as L
        from engine.analyze import analyze
        from engine.critic import collisions
        src = _sample("level2")
        W, H = 1080, 1350
        real = L._preferred

        def colliding(a, W, H):
            plan = real(a, W, H)
            els = [dict(e) for e in plan.elements]
            vis = next(e for e in els if e["type"] == "image")
            head = max((e for e in els if e["type"] == "text"), key=lambda e: e["font_size"])
            head.update(x=vis["x"] + vis["width"] / 2 - head["width"] / 2,
                        y=vis["y"] + vis["height"] / 2 - head["height"] / 2)
            return L.Plan(els, plan.quality, plan.floor)

        a = analyze(copy.deepcopy(src))
        with mock.patch.object(L, "_preferred", colliding):
            self.assertNotEqual(collisions(a.elements, L._finish(a, colliding(a, W, H), W, H), W, H), [])
            out = L._layout(a, W, H)
        self.assertEqual(collisions(a.elements, out, W, H), [])

    def test_floor_ladder(self):
        """Strictest first, and each step no stricter than the last for any
        role (the strip planners binary-search the graded steps on that)."""
        from engine.layout import (ALL_9PX, HEADLINE_LEAD, LADDER, LAST_UNGRADED, MARK_MIN,
                                   NO_FLOOR, STRICTEST)
        self.assertEqual(list(LADDER), sorted(LADDER, reverse=True))
        for stricter, looser in zip(LADDER, LADDER[1:]):
            for role in ("headline", "action", "text"):
                self.assertGreaterEqual(stricter.of(role), looser.of(role))
        # the logo minimum gives way before any text floor, and is off on the
        # graded steps (all the binary search needs)
        self.assertEqual(STRICTEST.mark, MARK_MIN)
        self.assertEqual(ALL_9PX.mark, 0)
        self.assertTrue(all(lv.mark == 0 for lv in LADDER if lv <= LAST_UNGRADED))
        roles = ("headline", "action", "text")
        self.assertEqual([STRICTEST.of(r) for r in roles],
                         [HEADLINE_LEAD * MIN_PRIMARY, MIN_PRIMARY, MIN_PRIMARY])
        self.assertEqual([ALL_9PX.of(r) for r in roles], [MIN_PRIMARY] * 3)
        self.assertTrue(all(LADDER[:-1]))
        self.assertFalse(NO_FLOOR)


if __name__ == "__main__":
    unittest.main()
