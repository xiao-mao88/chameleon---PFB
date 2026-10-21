"""
Where an image's visible pixels are.

Brand assets carry large transparent margins (the Solstice bottle's pixels
cover 43% of its width, the logo lockup's 40%), so spacing and collisions
measured on the element box are off by those margins. ink_box() gives the
part of an image element the renderer actually paints, from the asset's
alpha channel. Pixels are only read, never changed.
"""

from functools import lru_cache
from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent.parent / "brand_kit" / "assets"
ALPHA_MIN = 8               # alpha at or below this counts as transparent


@lru_cache(maxsize=256)
def _ink(asset_id: str) -> tuple[float, float, float, float, float]:
    """(x0, y0, x1, y1) of the visible pixels as fractions of the image, and
    the image's aspect ratio. Unknown or opaque assets: the whole image."""
    try:
        from PIL import Image
        path = ASSETS_DIR / asset_id
        if not path.is_file():
            return 0.0, 0.0, 1.0, 1.0, 0.0
        with Image.open(path) as im:
            aspect = im.width / max(im.height, 1)
            if "A" not in im.getbands():
                return 0.0, 0.0, 1.0, 1.0, aspect
            bbox = im.getchannel("A").point(lambda v: 255 if v > ALPHA_MIN else 0).getbbox()
            if bbox is None:
                return 0.0, 0.0, 1.0, 1.0, aspect
            return (bbox[0] / im.width, bbox[1] / im.height,
                    bbox[2] / im.width, bbox[3] / im.height, aspect)
    except Exception:
        return 0.0, 0.0, 1.0, 1.0, 0.0


def ink_box(el: dict) -> tuple[float, float, float, float]:
    """The box of what is painted for `el`: an image's visible pixels under
    its fit (contain: scaled to fit and centred; fill: stretched); its whole
    box for `cover` images, other elements and unknown assets."""
    x0, y0, w, h = el["x"], el["y"], el["width"], el["height"]
    if el.get("type") != "image" or el.get("fit") == "cover" or not el.get("asset_id"):
        return x0, y0, x0 + w, y0 + h
    fx0, fy0, fx1, fy1, aspect = _ink(el["asset_id"])
    if el.get("fit", "contain") == "contain" and aspect > 0 and w > 0 and h > 0:
        dw, dh = (w, w / aspect) if w / h < aspect else (h * aspect, h)
        x0, y0, w, h = x0 + (w - dw) / 2, y0 + (h - dh) / 2, dw, dh
    return x0 + fx0 * w, y0 + fy0 * h, x0 + fx1 * w, y0 + fy1 * h
