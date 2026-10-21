# CHAMELEON — design resize engine

*Adobe University Hackathon: North America 2026*

Give the engine a finished design (text, images and shapes, each with a
position, size and style) and a new canvas size, such as a square post turned
into a phone story or a thin web banner. It returns the same design
**re-arranged** for the new shape. Nothing is stretched, text stays
readable, and nothing overlaps that shouldn't.

The rules (see `RULES.md`): every element comes back exactly once, images
keep their proportions, text content and fonts never change, and no image
is redrawn. Both the output JSON and its rendered PNG are judged.

## Quick start

```bash
pip install -r requirements.txt   # Python 3.10+
python -m engine.main             # resizes every input/*.json into output/ (JSON + PNG)
```

Stage one runs the engine offline in Docker:

```bash
docker build --platform linux/amd64 -t chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output chameleon-engine
```

## How it works

`engine/resize.py` runs three steps. Everything is worked out from each
element's geometry and style. Element names are ignored, and nothing is
tuned to a particular design or size.

1. **Analyze** (`analyze.py`): decide what each element is, such as the
   background, a decoration, the hero image, a text panel, a corner badge,
   or copy, and put the copy in reading order.
2. **Lay out** (`layout.py`): arrange those pieces for the new shape:
   - **Tall canvases** stack everything in one centred column.
   - **Wide canvases** put the hero image beside the copy.
   - **Very thin banners** spread the copy across several columns.

   Text sizes scale with the canvas, then shrink until everything fits. Text
   never goes below 9px (headline) or 7px (everything else) if anything else
   can give way. When a format is too small for all the copy to be readable,
   the less important text gives way first, so the headline and the call to
   action stay readable. If the first arrangement leaves text unreadable or
   overlapping, every alternative is tried, and `critic.py` picks the best.
3. **Finalize** (`finalize.py`): a last pass enforcing the hard rules
   (every element once, proportions kept, everything on the canvas).

Two helpers keep this accurate:
- **`textmeasure.py`** measures text exactly as the official renderer draws it.
- **`assets.py`** finds each image's visible pixels, so transparent margins
  inside image files don't throw off spacing.

## Testing

| Command | What it does |
|---|---|
| `python -m unittest discover tests` | Runs 35 designs, including 20 generated edge cases (40 text boxes, rotated text, very long labels, missing images…), into 26 real platform sizes (Instagram, Story, YouTube, LinkedIn, standard web ad sizes such as 728×90 and 320×50, and more). Checks the hard rules, text overflow, readability, overlaps, contrast and speed. About 2 minutes. |
| `python -m tools.evaluate` | Compares the output with 504 layouts our team drew by hand (`adobe-projchameleon/`) and gives a 0–100 similarity score. Add `--save NAME` / `--compare NAME` to track changes, `--size 320x50` to also run sizes with no hand-drawn answer, and `--contrast 0.2` to check text contrast on a 20% sample. |
| `python -m tools.make_inputs` | Regenerates the sample tasks in `input/` so you can inspect the PNGs after `python -m engine.main`. |

The hand-drawn layouts are used only for testing. The engine never copies
them.

## Results

| On our 504 hand-drawn layouts | Score | Valid | No text overflow | Readable | No overlaps |
|---|---|---|---|---|---|
| Starting point | 56.3 | 100% | 100% | 94% | 88% |
| **Now** | **57.3** | 100% | 100% | **100%** | **100%** |

On the tiniest web banners, the headline and call to action stay readable
in every design:

| 97 designs resized to | 320×50 | 320×100 | 468×60 |
|---|---|---|---|
| Headline and CTA readable | 100% (was 78%) | 100% (was 95%) | 100% (was 95%) |
| All text readable | 77% | 95% | 95% |

All tests pass. The slowest single resize takes about 6 seconds.

## Known limitations

- **Tiny formats with lots of text.** In a 320×50 banner, some designs' copy
  can't all fit at a readable size. The headline and CTA stay readable, and
  the rest shrinks below 7px. In a 100×100 tile even the headline can't
  always fit. The tests list these cases and why.
- **Letter-spacing can be reduced** to make text fit. Text is never cut off.
- **The similarity score is a proxy.** It measures closeness to one
  set of hand-made choices (our own, not official). The official judge scores the rendered image.
- **Nothing is hidden.** Every element must stay, so on tiny canvases
  nothing is dropped to save space.

## Repo layout

```
engine/                the engine (resize.py is the entry point)
tests/                 test_engine.py
tools/                 evaluate.py · make_inputs.py · dataset.py
adobe-projchameleon/   our hand-drawn layouts, used for testing only
input/  output/        tasks in, JSON + PNG out
renderer/  brand_kit/  official renderer, fonts and images
eval/results/          saved evaluation runs
```
