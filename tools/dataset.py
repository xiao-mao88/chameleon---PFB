"""
Loader for the hand-drawn reference set in adobe-projchameleon/.

Layout: adobe-projchameleon/<design>/[<design>__]<format>_<W>x<H>.json, where
every design is drawn once per format with the same element ids. Any two
formats of one design make a task: the first is the source, the second is
the answer a designer would give for that target size.
"""

import json
import re
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "adobe-projchameleon"
_NAME_RE = re.compile(r"(?:.*__)?([a-z]+)_(\d+)x(\d+)\.json$")


@dataclass(frozen=True)
class Drawing:
    design: str   # e.g. "fernway_split_even"
    family: str   # e.g. "fernway_split" (brand + layout style)
    variant: str  # "even" | "image" | "shape" | "text" (what's emphasised)
    format: str   # e.g. "leaderboard"
    path: Path

    def load(self) -> dict:
        return json.loads(self.path.read_text())

    @property
    def size(self) -> str:
        c = self.load()["canvas"]
        return f"{c['width']}x{c['height']}"


@dataclass(frozen=True)
class Pair:
    source: Drawing
    target: Drawing

    @property
    def name(self) -> str:
        return f"{self.source.design}__{self.source.format}_to_{self.target.format}"


def drawings(dataset_dir: Path = DATASET_DIR) -> list[Drawing]:
    out = []
    for path in sorted(dataset_dir.glob("*/*.json")):
        m = _NAME_RE.match(path.name)
        if not m:
            continue
        design = path.parent.name
        family, _, variant = design.rpartition("_")
        out.append(Drawing(design, family, variant, m.group(1), path))
    return out


def pairs(dataset_dir: Path = DATASET_DIR) -> list[Pair]:
    by_design: dict[str, list[Drawing]] = {}
    for d in drawings(dataset_dir):
        by_design.setdefault(d.design, []).append(d)
    return [Pair(a, b) for ds in by_design.values() for a, b in permutations(ds, 2)]


def make_task(pair: Pair, **extra) -> dict:
    """A task file in the same shape engine/main.py reads from input/."""
    src = pair.source.load()
    return {
        "task_id": pair.name,
        **extra,
        "source": src,
        "target_canvas": dict(pair.target.load()["canvas"]),
    }
