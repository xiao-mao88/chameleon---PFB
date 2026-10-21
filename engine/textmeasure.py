"""
Text measurement that matches the official renderer.

Uses the same font lookup, skia glyph widths and greedy word-wrap that
renderer/converter/convert.py uses when it draws text, so "does this text
fit its box?" is answered the same way the PNG will be drawn. Like the
renderer, lines are broken on untracked, original-case widths, but drawn
(and so measured) with letter_spacing between glyphs and uppercased when
`capitalize` is set. Shared by the
evaluator (tools/evaluate.py) and, later, by resize()'s text fitting.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from renderer.converter.brand_fonts import load_brand_font_manager
from renderer.converter.convert import (
    _DEFAULT_LINE_HEIGHT,
    _is_italic,
    _normalize_weight,
    _weight_style_to_font_style,
)
from renderer.converter.wrap import char_widths, wrap_text

FONTS_DIR = Path(__file__).resolve().parent.parent / "brand_kit" / "fonts"


@lru_cache(maxsize=1)
def _font_manager():
    return load_brand_font_manager(str(FONTS_DIR))


def _font_args(el: dict[str, Any]) -> tuple[str, str, str]:
    weight = _normalize_weight(el.get("font_weight"))
    is_italic = _is_italic(el.get("font_style") or el.get("italic"))
    style = _weight_style_to_font_style(weight == "bold", is_italic)
    return el.get("font_family", "Solstice Sans"), weight, style


@lru_cache(maxsize=65536)
def _measure(content: str, family: str, weight: str, style: str,
             size: float, box_width: float, spacing: float = 0.0,
             capitalize: bool = False) -> tuple[int, float]:
    widths = char_widths(_font_manager(), family, weight, style, size, content)
    wrapped = wrap_text(content, widths, box_width)
    drawn = widths
    if capitalize and len(content.upper()) == len(content):
        drawn = char_widths(_font_manager(), family, weight, style, size, content.upper())
    longest = line = 0.0
    glyphs = 0
    for c, w in zip(wrapped, drawn):
        if c == "\n":
            longest = max(longest, line + spacing * max(0, glyphs - 1))
            line, glyphs = 0.0, 0
        else:
            line += w
            glyphs += 1
    longest = max(longest, line + spacing * max(0, glyphs - 1))
    return wrapped.count("\n") + 1, longest


def measure_text(el: dict[str, Any], font_size: float | None = None,
                 box_width: float | None = None) -> dict[str, float]:
    """Wrap `el`'s content the way the renderer will and report its extent.

    Returns {"lines", "text_width", "text_height"}: line count, widest line
    in px (can exceed the box when a single word is too long to wrap), and
    total height in px (lines * font_size * line_height).
    """
    size = float(font_size if font_size is not None else el["font_size"])
    width = float(box_width if box_width is not None else el["width"])
    lines, text_width = _measure(str(el.get("content", "")), *_font_args(el),
                                 size, width, float(el.get("letter_spacing") or 0.0),
                                 bool(el.get("capitalize", False)))
    line_height = float(el.get("line_height", _DEFAULT_LINE_HEIGHT))
    return {"lines": lines, "text_width": text_width,
            "text_height": lines * size * line_height}


def text_overflow(el: dict[str, Any], tol: float = 1.0) -> dict[str, float]:
    """How far rendered text spills out of its box, in px (0 = fits)."""
    m = measure_text(el)
    return {"x": max(0.0, m["text_width"] - el["width"] - tol),
            "y": max(0.0, m["text_height"] - el["height"] - tol)}
