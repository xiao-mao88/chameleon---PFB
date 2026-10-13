"""
Regenerate the hand-picked task files in input/ (the ones you look at as PNGs).

    python -m tools.make_inputs

Writes two kinds of file next to the 3 official samples (which it never
touches):

  ref_*.json     12 tasks from adobe-projchameleon/, one per design, chosen
                 so every target format appears and sources vary. Each has a
                 hand-drawn answer you can compare the PNG against.
                 (Fernway/Ridgeline photos aren't in brand_kit/assets, so
                 they render as placeholders — layout is still readable.)
  stress_*.json  Adversarial cases built from the Solstice assets: run-on
                 copy, many badges, rotation, a missing asset, opaque ids,
                 and target sizes that aren't in the reference set.

The full 504-pair set is scored in memory by tools/evaluate.py; it doesn't
need files.
"""

import copy
import json
from pathlib import Path

from tools.dataset import ROOT, Pair, drawings, make_task

INPUT_DIR = ROOT / "input"
SAMPLES = {p.stem: json.loads(p.read_text()) for p in INPUT_DIR.glob("chameleon_sample_*.json")}

# (design, source format, target format) — every target format at least
# once, the extreme ratios (leaderboard, skyscraper, banner) more than once.
REF_TASKS = [
    ("solstice_centered_even", "square", "leaderboard"),
    ("solstice_centered_image", "story", "banner"),
    ("solstice_centered_shape", "landscape", "skyscraper"),
    ("solstice_centered_text", "portrait", "landscape"),
    ("fernway_split_even", "square", "story"),
    ("fernway_split_image", "landscape", "portrait"),
    ("fernway_split_shape", "story", "leaderboard"),
    ("fernway_split_text", "banner", "square"),
    ("ridgeline_overlay_even", "square", "banner"),
    ("ridgeline_overlay_image", "portrait", "skyscraper"),
    ("ridgeline_overlay_shape", "skyscraper", "landscape"),
    ("ridgeline_overlay_text", "leaderboard", "story"),
]


def ref_tasks() -> dict[str, dict]:
    index = {(d.design, d.format): d for d in drawings()}
    out = {}
    for design, src, tgt in REF_TASKS:
        pair = Pair(index[(design, src)], index[(design, tgt)])
        out[f"ref_{design}__{src}_to_{tgt}"] = make_task(
            pair, difficulty="reference",
            notes=f"Hand-drawn answer: {pair.target.path.relative_to(ROOT)}")
    return out


def _sample(name: str) -> dict:
    return copy.deepcopy(SAMPLES[name]["source"])


def _task(source: dict, w: int, h: int, notes: str) -> dict:
    return {"difficulty": "stress", "notes": notes, "source": source,
            "target_canvas": {"width": w, "height": h}}


def stress_tasks() -> dict[str, dict]:
    core = "chameleon_sample_level2_core"
    adv = "chameleon_sample_level3_advanced"
    out = {}

    # Same official designs, other real-world formats.
    out["stress_core_to_story_1080x1920"] = _task(
        _sample(core), 1080, 1920, "Level 2 design into a tall Story.")
    out["stress_core_to_youtube_1280x720"] = _task(
        _sample(core), 1280, 720, "Level 2 design into a YouTube thumbnail.")
    out["stress_advanced_to_skyscraper_160x600"] = _task(
        _sample(adv), 160, 600, "Level 3 (run-on copy, 3 badges, rotated ribbon, "
        "missing asset) into a very narrow skyscraper.")
    out["stress_advanced_to_mobile_320x50"] = _task(
        _sample(adv), 320, 50, "Level 3 into a tiny mobile banner — far more "
        "content than fits; text must shrink, not spill.")

    # Opaque ids: nothing in the id hints at an element's role.
    src = _sample(core)
    mapping = {e["id"]: f"el_{n:02d}" for n, e in enumerate(src["elements"])}
    groups = {g: f"grp_{n}" for n, g in enumerate(
        sorted({e["group_id"] for e in src["elements"] if e.get("group_id")}))}
    for e in src["elements"]:
        e["id"] = mapping[e["id"]]
        e.pop("_note", None)
        if e.get("group_id"):
            e["group_id"] = groups[e["group_id"]]
    out["stress_opaque_ids_to_facebook_1200x628"] = _task(
        src, 1200, 628, "Level 2 with ids/group ids renamed to el_NN/grp_N — "
        "roles must come from geometry and style, not names.")

    # Five badges in a row, shuffled element order, into a medium rectangle.
    src = _sample(core)
    els = [e for e in src["elements"] if e["id"] not in ("badge_bg", "badge_label")]
    labels = ["NEW", "VEGAN", "ZERO SUGAR", "LIMITED", "4-PACK"]
    for n, label in enumerate(labels):
        x = 40 + n * 200
        els.append({"id": f"pill{n}_bg", "type": "shape", "x": x, "y": 40,
                    "width": 180, "height": 48, "z_index": 3, "shape_type": "rectangle",
                    "fill_color": "rgba(232,106,58,1)", "corner_radius": 24,
                    "group_id": f"pill{n}"})
        els.append({"id": f"pill{n}_label", "type": "text", "x": x, "y": 52,
                    "width": 180, "height": 26, "z_index": 4, "content": label,
                    "font_family": "Solstice Sans", "font_size": 16, "font_weight": "bold",
                    "color": "rgba(255,255,255,1)", "text_align": "center",
                    "group_id": f"pill{n}"})
    src["elements"] = els[::-1]
    out["stress_five_badges_to_mrec_300x250"] = _task(
        src, 300, 250, "Five grouped badges (elements listed in reverse z order) "
        "into a 300x250 ad.")

    # Run-on headline with a long unbreakable word, into a wide Twitter header.
    src = _sample(core)
    for e in src["elements"]:
        if e["id"] == "headline":
            e["content"] = ("Chase the Light with Supercalifragilisticexpialidocious "
                            "Botanical Sparkling Infusions All Summer Long")
            e.pop("style_runs", None)
    out["stress_longword_to_header_1500x500"] = _task(
        src, 1500, 500, "Very long headline containing one unbreakable word.")
    return out


def main() -> int:
    for old in list(INPUT_DIR.glob("ref_*.json")) + list(INPUT_DIR.glob("stress_*.json")):
        old.unlink()
    tasks = {**ref_tasks(), **stress_tasks()}
    for name, task in tasks.items():
        task.setdefault("task_id", name)
        (INPUT_DIR / f"{name}.json").write_text(json.dumps(task, indent=2))
    print(f"wrote {len(tasks)} task files to {INPUT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
