"""
Bundle the hand-drawn references into engine/reference_layouts.json so the
engine (and the Docker image, which only ships engine/) can learn from them.

    python -m tools.build_library
"""

import json
from pathlib import Path

from tools.dataset import drawings

OUT = Path(__file__).resolve().parent.parent / "engine" / "reference_layouts.json"


def main() -> None:
    lib: dict[str, dict] = {}
    for d in drawings():
        lib.setdefault(d.design, {"family": d.family, "formats": {}})
        lib[d.design]["formats"][d.format] = d.load()
    OUT.write_text(json.dumps(lib, separators=(",", ":")))
    n = sum(len(v["formats"]) for v in lib.values())
    print(f"wrote {OUT.name}: {len(lib)} designs, {n} layouts")


if __name__ == "__main__":
    main()
