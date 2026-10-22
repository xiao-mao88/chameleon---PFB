"""
Score resize() against the hand-drawn reference set.

For every (source format -> target format) pair of every design in
adobe-projchameleon/, runs resize() and compares the result with the
designer's own layout at the target size.

    python -m tools.evaluate                      # all 504 pairs
    python -m tools.evaluate --holdout ridgeline  # only one family (unseen-brand check)
    python -m tools.evaluate --save baseline      # write eval/results/baseline.json
    python -m tools.evaluate --compare baseline   # show deltas vs a saved run
    python -m tools.evaluate --engine engine.resize:resize   # any callable
    python -m tools.evaluate --size 320x50 --size 1000x1500  # + unseen sizes
                                                  # (hard checks only: no answer)
    python -m tools.evaluate --contrast 0.1       # + contrast on 10% of tasks (renders)

Two kinds of result per task:

  * Hard checks (pass/fail) — the RULES.md constraints the official
    structural check enforces: same ids, required fields, target canvas,
    nothing off-canvas, contain/non-cover images in proportion, content and
    font_family untouched, font_size >= 1. Plus text overflow (rendered
    text spilling outside its box), which the official PNG judge will see,
    and readability: every text >= 7px and the source's largest-font text
    (the headline) >= 9px (engine.critic). Also counted (not
    hard): collisions (text newly over other text, or over an image / solid
    shape it didn't overlap in the source) and, with --contrast, WCAG
    contrast of each text against what is rendered behind it.

  * Similarity to the reference, each 0..1 (higher is better):
      iou   mean box IoU per element
      pos   1 - centre distance / (0.25 * canvas diagonal), floored at 0
      rel   pairwise layout agreement: for each pair of elements, is A
            left-of / right-of / overlapping B horizontally (and same
            vertically) — does the output match the reference?
      font  1 - |log2(font_size / reference)|, floored at 0
    Full-canvas elements (backgrounds) are left out of iou/pos/rel: they're
    trivially right and would inflate every score.

  score = 100 * (0.30 iou + 0.25 pos + 0.25 rel + 0.20 font)

Matching the reference is a proxy, not the official rubric — the real
judge scores the rendered PNG. Look at the pictures too.
"""

import argparse
import importlib
import json
import math
import statistics
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from engine.critic import collisions, drawn as _drawn, key_copy_unreadable, unreadable_texts
from engine.textmeasure import text_overflow
from tools.dataset import Pair, drawings, make_task, pairs

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"

WEIGHTS = {"iou": 0.30, "pos": 0.25, "rel": 0.25, "font": 0.20}
REQUIRED_COMMON = ["id", "type", "x", "y", "width", "height"]
REQUIRED_BY_TYPE = {
    "text": ["content", "font_family", "font_size", "color"],
    "image": ["asset_id"],
    "shape": ["shape_type", "fill_color"],
}
EDGE_TOL = 0.5           # px slack for on-canvas checks
ASPECT_TOL = 0.05        # RULES.md: images in proportion within ~5%
OVERFLOW_TOL = 2.0       # px of text spill before we flag it
CONTRAST_NORMAL = 4.5    # WCAG 2.x AA, normal text
CONTRAST_LARGE = 3.0     # WCAG 2.x AA, large text: >= 18pt (24px), or >= 14pt (18.7px) bold
ASSETS_DIR = Path(__file__).resolve().parent.parent / "brand_kit" / "assets"


# --------------------------------------------------------------------------
# Hard checks
# --------------------------------------------------------------------------

def hard_checks(source: dict, output: dict, target: dict) -> list[str]:
    if not isinstance(output, dict) or "canvas" not in output or "elements" not in output:
        return ["output missing 'canvas' or 'elements'"]

    problems = []
    canvas = output["canvas"]
    if (canvas.get("width"), canvas.get("height")) != (target["width"], target["height"]):
        problems.append(f"canvas is {canvas.get('width')}x{canvas.get('height')}, "
                        f"expected {target['width']}x{target['height']}")
    if not all(isinstance(canvas.get(k), int) for k in ("width", "height")):
        problems.append("canvas width/height must be integers")

    src_by_id = {e["id"]: e for e in source["elements"]}
    out_ids = [e.get("id") for e in output["elements"]]
    if sorted(out_ids, key=str) != sorted(src_by_id, key=str):
        missing = sorted(set(src_by_id) - set(out_ids))
        extra = sorted(set(out_ids) - set(src_by_id), key=str)
        dupes = sorted({i for i in out_ids if out_ids.count(i) > 1}, key=str)
        problems.append(f"id mismatch: missing={missing} extra={extra} duplicated={dupes}")

    W, H = target["width"], target["height"]
    for el in output["elements"]:
        eid = el.get("id")
        missing = [f for f in REQUIRED_COMMON + REQUIRED_BY_TYPE.get(el.get("type"), [])
                   if f not in el]
        if missing:
            problems.append(f"{eid}: missing fields {missing}")
            continue
        nums = [el["x"], el["y"], el["width"], el["height"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in nums):
            problems.append(f"{eid}: non-finite geometry")
            continue
        if el["width"] < 0 or el["height"] < 0:
            problems.append(f"{eid}: negative size")
        if (el["x"] < -EDGE_TOL or el["y"] < -EDGE_TOL
                or el["x"] + el["width"] > W + EDGE_TOL
                or el["y"] + el["height"] > H + EDGE_TOL):
            problems.append(f"{eid}: off-canvas")

        src = src_by_id.get(eid)
        if src is None:
            continue
        if el.get("type") != src.get("type"):
            problems.append(f"{eid}: type changed")
        if el["type"] == "text":
            if el.get("content") != src.get("content"):
                problems.append(f"{eid}: content changed")
            if el.get("font_family") != src.get("font_family"):
                problems.append(f"{eid}: font_family changed")
            if not el.get("font_size", 0) >= 1:
                problems.append(f"{eid}: font_size < 1")
        if el["type"] == "image":
            if el.get("asset_id") != src.get("asset_id"):
                problems.append(f"{eid}: asset_id changed")
            if src.get("fit") != "cover" and src["height"] > 0 and el["height"] > 0:
                ratio = (el["width"] / el["height"]) / (src["width"] / src["height"])
                if abs(ratio - 1) > ASPECT_TOL:
                    problems.append(f"{eid}: aspect ratio off by {abs(ratio - 1):.0%}")
    return problems


def overflow_checks(output: dict) -> list[str]:
    problems = []
    for el in output.get("elements", []):
        if el.get("type") != "text" or not el.get("content"):
            continue
        try:
            o = text_overflow(el, OVERFLOW_TOL)
        except Exception as exc:  # unknown font etc. — report, don't crash
            problems.append(f"{el.get('id')}: could not measure text ({exc})")
            continue
        if o["x"] or o["y"]:
            problems.append(f"{el['id']}: text overflows box by "
                            f"{o['x']:.0f}px wide, {o['y']:.0f}px tall")
    return problems


def readability_checks(source: dict, output: dict) -> list[str]:
    return [f"{i}: unreadable font {fs:.1f}px (< {floor:.0f}px)"
            for i, fs, floor in unreadable_texts(source["elements"], output.get("elements", []))]


def collision_checks(source: dict, output: dict) -> list[str]:
    """Text that newly overlaps other text, or a visible image / solid shape
    it didn't overlap in the source (see engine/critic.py)."""
    return collisions(source["elements"], output["elements"],
                      output["canvas"]["width"], output["canvas"]["height"])


def _rgba(color: str) -> tuple[float, float, float]:
    import re
    nums = [float(v) for v in re.findall(r"[\d.]+", str(color))]
    if str(color).startswith("#"):
        h = str(color).lstrip("#")
        nums = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return tuple((nums + [0, 0, 0])[:3])


def _luminance(rgb):
    """WCAG relative luminance of sRGB values 0-255 (array or tuple)."""
    import numpy as np
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def text_contrasts(design: dict, max_size: int = 768) -> dict[str, tuple[float, float]]:
    """{text id: (worst-decile contrast ratio, WCAG AA threshold)}: render the
    design without its text, then compare each text's colour with the pixels
    behind its box (4.5:1 normal text, 3:1 large text)."""
    import io
    import numpy as np
    from PIL import Image
    from renderer.converter.convert import _get_renderer, convert_chameleon_to_crello

    W, H = design["canvas"]["width"], design["canvas"]["height"]
    texts = [e for e in design["elements"] if e["type"] == "text" and e.get("content")]
    if not texts:
        return {}
    bare = dict(design, elements=[e for e in design["elements"] if e["type"] != "text"])
    renderer = _get_renderer(str(ASSETS_DIR.parent / "fonts"))
    example, _ = convert_chameleon_to_crello(bare, renderer.font_manager, str(ASSETS_DIR))
    img = Image.open(io.BytesIO(renderer.render(example, max_size=max_size, format="png")))
    px = np.asarray(img.convert("RGB"), dtype=float)
    sx, sy = px.shape[1] / W, px.shape[0] / H
    out = {}
    for e in texts:
        x0, y0, x1, y1 = _drawn(e)
        box = px[max(0, int(y0 * sy)):max(1, int(math.ceil(y1 * sy))),
                 max(0, int(x0 * sx)):max(1, int(math.ceil(x1 * sx)))]
        if box.size == 0:
            continue
        lb = _luminance(box.reshape(-1, 3))
        lt = float(_luminance(_rgba(e.get("color", "rgba(0,0,0,1)"))))
        ratio = (np.maximum(lb, lt) + 0.05) / (np.minimum(lb, lt) + 0.05)
        bold = str(e.get("font_weight", "")).lower() in ("bold", "700", "800", "900")
        large = e["font_size"] >= 24 or (bold and e["font_size"] >= 18.7)
        out[e["id"]] = (float(np.percentile(ratio, 10)),
                        CONTRAST_LARGE if large else CONTRAST_NORMAL)
    return out


def contrast_checks(output: dict, source: dict | None = None) -> list[str]:
    """Text below WCAG AA contrast against what is drawn behind it. With a
    source, only text the layout made worse is counted: brand colours (white
    on an orange button) are the design's choice, not the engine's."""
    now = text_contrasts(output)
    before = text_contrasts(source) if source is not None else {}
    out = []
    for i, (worst, need) in now.items():
        if worst >= need:
            continue
        if i in before and worst >= before[i][0] - 0.5:
            continue
        was = f", source {before[i][0]:.1f}:1" if i in before else ""
        out.append(f"{i}: low contrast {worst:.1f}:1 (< {need}:1{was})")
    return out


# --------------------------------------------------------------------------
# Similarity to the reference
# --------------------------------------------------------------------------

def _box(el):
    return el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"]


def _iou(a, b):
    ax0, ay0, ax1, ay1 = _box(a)
    bx0, by0, bx1, by1 = _box(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _relation(a0, a1, b0, b1, tol):
    if a1 <= b0 + tol:
        return -1
    if b1 <= a0 + tol:
        return 1
    return 0


def similarity(output: dict, reference: dict) -> dict[str, float]:
    W, H = reference["canvas"]["width"], reference["canvas"]["height"]
    diag = math.hypot(W, H)
    ref = {e["id"]: e for e in reference["elements"]}
    out = {e.get("id"): e for e in output.get("elements", [])}

    def full_canvas(e):
        return e["width"] * e["height"] >= 0.95 * W * H

    ids = [i for i, e in ref.items() if i in out and not full_canvas(e)]
    ious, poss, fonts = [], [], []
    for i in ids:
        o, r = out[i], ref[i]
        ious.append(_iou(o, r))
        d = math.hypot((o["x"] + o["width"] / 2) - (r["x"] + r["width"] / 2),
                       (o["y"] + o["height"] / 2) - (r["y"] + r["height"] / 2))
        poss.append(max(0.0, 1 - d / (0.25 * diag)))
    for i, r in ref.items():
        if r["type"] == "text" and i in out and out[i].get("font_size", 0) > 0:
            fonts.append(max(0.0, 1 - abs(math.log2(out[i]["font_size"] / r["font_size"]))))

    agree = total = 0
    tx, ty = 0.01 * W, 0.01 * H
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            for axis, tol in ((0, tx), (1, ty)):
                ra = _relation(_box(ref[a])[axis], _box(ref[a])[axis + 2],
                               _box(ref[b])[axis], _box(ref[b])[axis + 2], tol)
                oa = _relation(_box(out[a])[axis], _box(out[a])[axis + 2],
                               _box(out[b])[axis], _box(out[b])[axis + 2], tol)
                agree += ra == oa
                total += 1

    m = {
        "iou": statistics.fmean(ious) if ious else 1.0,
        "pos": statistics.fmean(poss) if poss else 1.0,
        "rel": agree / total if total else 1.0,
        "font": statistics.fmean(fonts) if fonts else 1.0,
    }
    m["score"] = 100 * sum(WEIGHTS[k] * m[k] for k in WEIGHTS)
    return m


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def load_engine(spec: str):
    module, _, func = spec.partition(":")
    return getattr(importlib.import_module(module), func or "resize")


def evaluate_pair(resize, pair: Pair) -> dict:
    task = make_task(pair)
    reference = pair.target.load()
    row = {
        "task": pair.name, "design": pair.source.design, "family": pair.source.family,
        "variant": pair.source.variant, "source": pair.source.format,
        "target": pair.target.format,
    }
    try:
        output = resize(json.loads(json.dumps(task["source"])), dict(task["target_canvas"]))
    except Exception:
        row.update(crashed=True, hard=["crashed: " + traceback.format_exc(limit=3)],
                   overflow=[], unreadable=[], collisions=[], score=0.0, iou=0.0, pos=0.0, rel=0.0, font=0.0)
        return row
    row["crashed"] = False
    row["hard"] = hard_checks(task["source"], output, task["target_canvas"])
    row["overflow"] = overflow_checks(output)
    row["unreadable"] = readability_checks(task["source"], output)
    row["collisions"] = collision_checks(task["source"], output)
    row["output"] = output
    try:
        row.update(similarity(output, reference))
    except Exception:
        row.update(score=0.0, iou=0.0, pos=0.0, rel=0.0, font=0.0)
        row["hard"].append("similarity failed: " + traceback.format_exc(limit=2))
    return row


def evaluate_size(resize, drawing, W: int, H: int) -> dict:
    """A target size with no hand-drawn answer: hard checks only."""
    source = drawing.load()
    target = {"width": W, "height": H}
    row = {"task": f"{drawing.design}__{drawing.format}_to_{W}x{H}", "target": f"{W}x{H}"}
    try:
        output = resize(json.loads(json.dumps(source)), dict(target))
    except Exception:
        return dict(row, hard=["crashed: " + traceback.format_exc(limit=3)],
                    overflow=[], unreadable=[], key_unreadable=[])
    return dict(row, hard=hard_checks(source, output, target),
                overflow=overflow_checks(output),
                unreadable=readability_checks(source, output),
                key_unreadable=key_copy_unreadable(source, output["elements"]))


def report_sizes(rows: list[dict]) -> dict:
    print("\n== unseen sizes (hard checks only) ==")
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[r["target"]].append(r)
    for name, rs in by.items():
        n = len(rs)
        out[name] = {"n": n, "valid": sum(not r["hard"] for r in rs) / n,
                     "no_overflow": sum(not r["overflow"] for r in rs) / n,
                     "readable": sum(not r["unreadable"] for r in rs) / n,
                     "key_copy": sum(not r["key_unreadable"] for r in rs) / n}
        s = out[name]
        print(f"  {name:<22}{n:>4}  valid {s['valid']:4.0%}  "
              f"no-overflow {s['no_overflow']:4.0%}  readable {s['readable']:4.0%}  "
              f"headline+CTA readable {s['key_copy']:4.0%}")
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "score": statistics.fmean(r["score"] for r in rows),
        **{k: statistics.fmean(r[k] for r in rows) for k in WEIGHTS},
        "valid": sum(not r["hard"] for r in rows) / n,
        "no_overflow": sum(not r["overflow"] for r in rows) / n,
        "readable": sum(not r.get("unreadable") for r in rows) / n,
        "no_collision": sum(not r.get("collisions") for r in rows) / n,
        "crashed": sum(r["crashed"] for r in rows),
    }


def _fmt(s: dict, prev: dict | None = None) -> str:
    def d(key, scale=1.0, digits=1):
        if not prev or key not in prev:
            return ""
        delta = (s[key] - prev[key]) * scale
        return f"({delta:+.{digits}f})" if abs(delta) >= 0.05 else ""
    return (f"{s['n']:>4}  score {s['score']:5.1f}{d('score'):<8}"
            f"iou {s['iou']:.2f}  pos {s['pos']:.2f}  rel {s['rel']:.2f}  font {s['font']:.2f}  "
            f"valid {s['valid']:4.0%}{d('valid', 100):<8}"
            f"no-overflow {s['no_overflow']:4.0%}{d('no_overflow', 100):<8}"
            f"readable {s.get('readable', 0):4.0%}{d('readable', 100):<8}"
            f"no-collision {s.get('no_collision', 0):4.0%}{d('no_collision', 100)}")


def report(rows: list[dict], prev: dict | None = None, worst: int = 10) -> dict:
    groups = {"overall": {"all": rows}}
    for key in ("target", "source", "family", "variant"):
        g = defaultdict(list)
        for r in rows:
            g[r[key]].append(r)
        groups[key] = dict(sorted(g.items()))

    summary = {k: {name: summarize(rs) for name, rs in g.items()} for k, g in groups.items()}
    for key, g in summary.items():
        print(f"\n== by {key} ==" if key != "overall" else "\n== overall ==")
        for name, s in g.items():
            p = (prev or {}).get(key, {}).get(name)
            print(f"  {name:<22}{_fmt(s, p)}")

    problems = defaultdict(int)
    for r in rows:
        for p in r["hard"] + r["overflow"] + r.get("unreadable", []) + r.get("collisions", []):
            kind = (p.split(": ", 1)[-1].split(" by ")[0].split(" (")[0].split(" font ")[0]
                    if " collides with " not in p else "collision")
            problems[kind] += 1
    if problems:
        print("\n== most common problems (occurrences) ==")
        for kind, c in sorted(problems.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {c:>5}  {kind}")

    if worst:
        print(f"\n== {worst} lowest-scoring tasks ==")
        for r in sorted(rows, key=lambda r: r["score"])[:worst]:
            print(f"  {r['score']:5.1f}  {r['task']}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", default="engine.resize:resize",
                    help="module:function to evaluate (default engine.resize:resize)")
    ap.add_argument("--holdout", help="only evaluate this family (e.g. ridgeline, fernway)")
    ap.add_argument("--exclude", help="skip this family")
    ap.add_argument("--target", help="only this target format (e.g. leaderboard)")
    ap.add_argument("--source", help="only this source format (e.g. square)")
    ap.add_argument("--save", metavar="NAME", help="save results to eval/results/NAME.json")
    ap.add_argument("--compare", metavar="NAME", help="show deltas vs eval/results/NAME.json")
    ap.add_argument("--worst", type=int, default=10, help="list N lowest-scoring tasks")
    ap.add_argument("--show", metavar="TASK", help="print every problem for tasks matching TASK")
    ap.add_argument("--size", action="append", default=[], metavar="WxH",
                    help="also run every design (from each format) into this size; repeatable")
    ap.add_argument("--contrast", type=float, default=0.0, metavar="FRACTION",
                    help="render this fraction of tasks (every k-th) and check text contrast")
    args = ap.parse_args(argv)

    resize = load_engine(args.engine)
    selected = [p for p in pairs()
                if (not args.holdout or p.source.family.startswith(args.holdout))
                and (not args.exclude or not p.source.family.startswith(args.exclude))
                and (not args.target or p.target.format == args.target)
                and (not args.source or p.source.format == args.source)]
    if not selected:
        print("no tasks match those filters", file=sys.stderr)
        return 1

    rows = [evaluate_pair(resize, p) for p in selected]
    if args.contrast > 0:
        step = max(1, round(1 / args.contrast))
        sampled = rows[::step]
        by_task = {p.name: p for p in selected}
        for r in sampled:
            r["contrast"] = (contrast_checks(r["output"], by_task[r["task"]].source.load())
                             if not r["crashed"] else [])
        ok = sum(not r["contrast"] for r in sampled)
        print(f"contrast: {ok}/{len(sampled)} sampled tasks have no text the layout pushed "
              f"below WCAG AA ({ok / len(sampled):.0%})")
        for r in sampled:
            for c in r["contrast"]:
                print(f"    {r['task']}: {c}")
    for r in rows:
        r.pop("output", None)
    prev = None
    if args.compare:
        prev = json.loads((RESULTS_DIR / f"{args.compare}.json").read_text())["summary"]
    print(f"engine: {args.engine}")
    summary = report(rows, prev, args.worst)

    size_rows = []
    for spec in args.size:
        W, H = (int(v) for v in spec.lower().split("x"))
        size_rows += [evaluate_size(resize, d, W, H) for d in drawings()
                      if (not args.holdout or d.family.startswith(args.holdout))
                      and (not args.source or d.format == args.source)]
    if size_rows:
        summary["sizes"] = report_sizes(size_rows)
    rows_all = rows + size_rows

    if args.show:
        for r in rows_all:
            if args.show in r["task"]:
                print(f"\n{r['task']}  score {r.get('score', float('nan')):.1f}")
                for p in (r["hard"] + r["overflow"] + r.get("unreadable", [])
                          + r.get("collisions", []) + r.get("contrast", [])):
                    print(f"  - {p}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"{args.save}.json"
        path.write_text(json.dumps({"engine": args.engine, "summary": summary,
                                    "tasks": rows, "size_tasks": size_rows}, indent=1))
        print(f"\nsaved {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
