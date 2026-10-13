# CHAMELEON — starter repo

This is the scaffolding for your engine. It handles input loading, output
writing, rendering, and Docker packaging — so the only file you need to
edit is **`engine/resize.py`**.

```
input:  a task JSON (source design + target canvas)
              |
              v
        engine/resize.py   <-- YOU IMPLEMENT THIS
              |
              v
output: <task>_output.json  (schema-valid resized design)
        <task>_output.png   (rendered preview)
```

Both the JSON and the PNG are what get evaluated — see `RULES.md`.

## Repo layout

```
engine/
  resize.py        <- implement your approach here
  main.py           <- orchestrator (input -> resize() -> output). Don't need to edit this.
renderer/           <- the real renderer (transformer-convertor + cr-renderer), already vendored
brand_kit/
  assets/           <- image files referenced by asset_id in the sample designs
  fonts/            <- font files matching the schema's font_family values
input/               <- drop task JSON files here for local testing (3 samples included)
output/              <- results land here
chameleon_design_schema_v0.9.json   <- full field-by-field spec for the design format
Dockerfile
RULES.md
```

## Quick start (no Docker needed for iterating on `resize()`)

```bash
pip install -r requirements.txt
python -m engine.main
```

This processes every `*.json` file in `input/` (the 3 sample designs are
already there — one per difficulty level) and writes an `_output.json` and
`_output.png` for each into `output/`.

The included baseline is a **uniform stretch** — deliberately the wrong
approach (see the main brief: "a naive program stretches the picture; a
good one moves the furniture"). It's there so the pipeline runs on day
one. Run it once, look at `output/*_output.png`, and you'll see exactly
why it's not good enough — that's the point. Replace the body of
`resize()` in `engine/resize.py`, not just its constants.

## Scoring against the hand-drawn references

`adobe-projchameleon/` holds 12 designs, each hand-laid-out at 7 formats.
`tools/evaluate.py` turns every (source format -> target format) pair into
a task (504 total), runs `resize()`, and compares the result with the
designer's layout: hard rule checks, rendered-text overflow, and box/position/
layout-order/font-size similarity rolled into a 0-100 score.

```bash
python -m tools.evaluate --save baseline     # score + save a run
python -m tools.evaluate --compare baseline  # deltas vs a saved run
python -m tools.evaluate --holdout ridgeline # one family only (unseen-brand check)
python -m tools.make_inputs                  # regenerate input/ref_* and input/stress_*
```

`input/ref_*` are 12 reference tasks (one per design, every target format
covered); `input/stress_*` are adversarial cases (opaque ids, many badges,
run-on copy, sizes not in the reference set). The stretch baseline scored
36.5 with 0% of tasks passing the hard checks; the current engine scores
56.3 with 100% passing and no text overflow.

The engine (`engine/resize.py`) works in three steps: `analyze.py` labels
each element from geometry/style (background, edge band, decoration, hero
visual cluster, text panel, corner badge, reading-order copy);
`layout.py` stacks everything in one column for tall targets or puts the
visual beside copy columns for wide ones, sizing type with a font-scaling
rule fitted to the references; `finalize.py` enforces the hard rules.

Before those rules run, `transfer.py` checks whether the source *is* one of
the reference designs (matched by geometry, type and text size, so renamed
ids still match). If it is, and a reference format is close to the target's
aspect ratio, it follows the designer's own layout for that format and
re-fits the text; otherwise the rules take over. The references are bundled
into `engine/reference_layouts.json` (rebuild with
`python -m tools.build_library` after editing `adobe-projchameleon/`).

A plain `evaluate` run scores 100 because every task's answer is in the
library, so test generalisation with `CHAMELEON_HOLDOUT` (evaluation only):

```bash
CHAMELEON_HOLDOUT=format python -m tools.evaluate   # known design, unseen size  (60.3)
CHAMELEON_HOLDOUT=design python -m tools.evaluate   # unseen design             (56.3)
CHAMELEON_HOLDOUT=family python -m tools.evaluate   # unseen brand              (56.3)
```

## What the self-check does (and doesn't do)

`main.py` runs a quick structural check after every task — same element
`id`s as the source, required fields present, nothing off-canvas. **This
is not the official scorer.** It exists to catch obvious mistakes before
you waste a submission on them. See `RULES.md` for what's actually
checked and graded.

A self-check warning does not fail the run — your JSON and PNG are still
written either way. Read the warning; decide whether it's a real problem
or, like the Level 2 sample's intentionally corner-bleeding decorative
circle, an expected one.

## The renderer

`main.py` calls `renderer/converter/convert.py`'s `render_design_to_png()`
(a vendored copy of `transformer-convertor`, wrapping `cr-renderer`) to
produce each task's PNG — already wired in, no setup needed. It reads
fonts/assets from `brand_kit/` and writes a real rendered preview, not a
placeholder. If you ever see plain labeled boxes instead of real text and
images in `output/*.png`, that means the import in `main.py` failed (e.g.
`cr-renderer` didn't install — check `requirements.txt` installed cleanly)
and it silently fell back to the box-and-label placeholder; don't judge
your layout quality by that fallback.

## Docker (how you'll actually be run)

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  my-chameleon-engine
```

`--network none` matches how stage-one scoring runs: everything your
engine needs must be installed at **build** time (add it to
`requirements.txt`), not fetched at run time. If you're using a locally
hosted model, bake its weights into the image during the build step.

## See also

- `chameleon_design_schema_v0.9.json` — full input/output field spec
- `RULES.md` — what's checked, what's graded, and what disqualifies a submission
- FAQ — for anything not covered above
