# CHAMELEON — design resize engine

*Adobe University Hackathon: North America 2026*

## The challenge

Give the engine a finished design (text, image and shape elements with
position, size, font, stacking order and grouping) and a new canvas size.
It returns the same design re-laid-out for that canvas, with nothing
overlapping that shouldn't, text readable, brand intact, and the hierarchy
kept, all without a human touching it. The brief's analogy is to *move the
furniture, don't stretch the photo of the room*. Every element id comes back
exactly once, images keep their proportions, copy and fonts are never
changed, and no image content is generated. Both the output JSON and its
rendered PNG are judged (see `RULES.md`).

## How the engine works

`engine/resize.py` → `analyze` → `layout` → `finalize`. Everything is
inferred from geometry and style. Ids carry no meaning, and nothing is
tuned to a specific design or size.

| Step | File | What it does |
|---|---|---|
| Analyze | `engine/analyze.py` | Labels every element: background, edge band, faint decoration, hero visual cluster, text panel, corner badge, or reading-order copy (via recursive XY-cut). Groups, badge rows and buttons move as units. |
| Arrange | `engine/layout.py` | Tall targets get one centred column. Wide targets put the hero visual beside copy columns (a CTA gets its own column on wide banners). Thin strips spread the copy over as many side-by-side columns as it needs, using a DP over the reading order, so even 80 text boxes lay out in under a second. Near-square targets follow the source's alignment. If the usual arrangement can't keep the copy readable, every arrangement is tried and the most readable one wins. Short wide strips can also put the headline on its own full-width row above columns. **No collisions**: copy outside a text panel never shares the panel's column, and the panel is trimmed away from it. The hero visual fits between pinned corner items, the copy column steps beside a side panel no copy sat on, and an edge band (a horizon) moves its edge so copy sits fully on or fully off it, never across it. |
| Size type | `engine/layout.py` | Fonts start at a scaling law fitted across the hand-drawn references (big type tracks the canvas, small type less), then each column shrinks until it fits. **Readability floors**: no text goes below 9px (then 7px) while anything else can still give way, and the headline never drops below 9px. Buttons, badges and tags re-wrap around a readable label (always, if the label already spilled out of its pill in the source, e.g. translated copy). Badge rows wrap into compact pills, and captions too small to read on the hero visual move into the copy. Letter-spacing scales with the font and is loosened when wrapped lines would draw wider than their box. Rotated text is sized by its rotated bounds. |
| Enforce | `engine/finalize.py` | The hard rules as a last pass: every id once, protected fields untouched, images and ellipses in proportion, everything on canvas, including rotated elements' drawn bounds. |
| Measure | `engine/textmeasure.py` | Wraps text with the official renderer's fonts and glyph widths, so "does it fit?" matches the PNG. Like the renderer, it breaks lines on plain widths, then measures them as drawn, with letter-spacing (and `capitalize`). |

If layout ever raises an exception, `resize()` falls back to a uniform,
undistorted fit rather than crashing.

## Run it

```bash
pip install -r requirements.txt        # Python 3.10+, installs cr-renderer
python -m engine.main                  # every input/*.json -> output/<task>_output.{json,png}
```

In Docker, the way stage one runs it (offline):

```bash
docker build --platform linux/amd64 -t chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output chameleon-engine
```

## Test it

| Command | What it checks |
|---|---|
| `python -m unittest discover tests` | 13 engine tests on **unseen** cases. Sources: the 3 official samples, all 12 reference designs, and 20 generated designs: opaque ids, reversed z-order, one element, text only, no text, 30+ elements, rotation (one element, and everything at 30°), run-on copy, missing asset, labels longer than their pills, letter-spacing on every line, 40 text boxes, a 9-photo grid, a 10-line menu, two heroes, text on a photo, Unicode, bleed off-canvas, zero-size and empty elements. Each runs into 26 real platform formats: Instagram post / portrait / story / grid, Facebook link and cover, X post and header, LinkedIn banner and company cover, YouTube thumbnail and channel art, Pinterest, IAB 300×250 / 300×600 / 160×600 / 120×600 / 180×150 / 728×90 / 970×90 / 468×60 / 320×50 / 320×100 / 970×250, a 100×100 tile and a print poster. Checks: hard rules, ellipse proportions, no text overflow, readability, no collisions (text against text, images and solid shapes), contrast (sampled, rendered), rotated bounds on canvas, every element placed by the layout itself, no crash or fallback, every case under 10s, determinism. Takes about 2 minutes. |
| `python -m tools.evaluate` | Scores `resize()` against the 504 hand-drawn reference pairs in `adobe-projchameleon/` (every format of a design → every other). Reports box/position/layout-order/font similarity (0–100), plus valid / no-overflow / readable / no-collision rates. Options: `--save NAME`, `--compare NAME`, `--holdout FAMILY`, `--target FMT`, `--show TASK`, `--worst N`. |
| `python -m tools.evaluate --contrast 0.2` | Also renders 20% of tasks with their text hidden and checks each text against what's behind it (worst 10% of pixels, WCAG AA: 4.5:1, or 3:1 for large text). It only counts text the layout made worse than in the source, because brand colours, such as white on an orange button, are the design's choice. |
| `python -m tools.evaluate --size 320x50 --size 1000x1500` | Also runs every reference design into sizes with no hand-drawn answer (hard checks and readability only). |
| `python -m tools.make_inputs` | Regenerates `input/ref_*` (12 reference tasks) and `input/stress_*` (10 adversarial tasks) to inspect as PNGs after `python -m engine.main`. The 3 official samples are left untouched. |

The references in `adobe-projchameleon/` are used **only for evaluation**
and for fitting the general font-scaling law. The engine does not look up
or copy any reference layout.

## Results

`eval/results/tier3.json` vs `eval/results/rules-before.json` and `tier3-before.json`:

| | Score | Valid | No overflow | Readable | No collision |
|---|---|---|---|---|---|
| 504 reference pairs: before | 56.3 | 100% | 100% | 94% | 88% |
| 504 reference pairs: now | **56.9** | 100% | 100% | **100%** | **99%** |
| by family (fernway / ridgeline / solstice) | 56.1 / 51.3 / 63.3 | 100% | 100% | 100% | 99 / 100 / 99% |

Contrast: none of the 101 sampled reference tasks, and none of 182 sampled test cases, has text the layout pushed below WCAG AA.

Unseen catalogue sizes (each reference design from each format, 84 tasks per
size). Valid and no-overflow are 100% everywhere. Readability is 100% at
1080×1080, 1200×628, 1600×900, 1500×500, 1000×1500, 300×250, 300×600,
970×250 and 2400×3600; **100% at 320×100** (up from 19%) and **82% at 320×50**
(up from 15%).

`tests/`: all 13 tests pass on 910 cases (35 sources × 26 formats). The
slowest case takes under 5s.

## Known limitations

- **Copy-heavy designs on tiny or narrow formats.** Some designs' copy
  can't fit at 7px at all, so it shrinks below that (best effort, and it may
  overlap). `tests/test_engine.py` exempts these by a physical-fit check
  (copy area, longest word, rotated bounds): 35 of 910 cases, all on formats
  under 40k px² or ≤ 320px wide. It also names 4 near misses in
  `KNOWN_LIMITATIONS`, each with its reason.
- **Tracking may be loosened.** Wrapped letter-spaced text loses some or all
  of its tracking, because the renderer breaks lines without it and would
  otherwise draw past the box. Short one-line tracked labels keep it.
- **The judge's metric is not ours.** The evaluator measures how close the
  output is to one designer's layout. That's a proxy: the official judge
  scores the rendered PNG against its own rubric.
- **Arrangement choices on some formats.** On banners and leaderboards, a
  hero visual that sat above the copy goes on the left, where the designers
  often put the copy left.
- **Transparent padding inside image files.** Spacing uses each image's box,
  not its visible pixels, so an asset with built-in transparent margins
  (the Solstice bottle, the logo lockup) can look farther from the copy than
  the layout intends, e.g. on 1000×1500.
- **No platform safe zones.** Keeping Story copy out of the app's top and
  bottom UI bands (guides suggest ≈250 / 340px of 1920) was tried. It made
  Story layouts worse against the designers' references (−0.9 to −1.5), so
  it isn't applied.
- **Nothing is hidden.** Stage one requires every element to stay, so on
  tiny formats nothing drops away. Letting minor elements drop is a
  finale-stage idea.

## Repo layout

```
engine/            resize.py (entry) · analyze.py · layout.py · finalize.py · textmeasure.py · main.py (orchestrator)
tests/             test_engine.py — unseen designs × real formats
tools/             evaluate.py · dataset.py · make_inputs.py
eval/results/      saved evaluator runs
adobe-projchameleon/   hand-drawn references (evaluation only)
input/ output/     task files in, JSON + PNG out
renderer/ brand_kit/   official renderer, fonts and assets
chameleon_design_schema_v0.9.json · RULES.md · FAQ.md · Dockerfile
```
