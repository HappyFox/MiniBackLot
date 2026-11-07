"""High-quality "settle" rendering, done off the main thread.

The interactive renderer (:mod:`.renderer`) favours speed: nearest-neighbour
during gestures, then ``smoothscale``/``rotozoom``. Those leave visible
compromises: soft bilinear upscaling, aliasing when shrinking while
rotated (``rotozoom`` doesn't pre-filter), and up to half a pixel of
positioning error from integer rounding.

Once the view has settled, this module renders the whole frame again:

1. Pick a pyramid level *finer* than strictly needed (when affordable), so
   the final reduction has real detail to work with.
2. Crop the visible region and, when shrinking, resize it with Lanczos
   (a properly band-limited filter, so no aliasing or moiré from
   resampling).
3. Apply one exact affine transform with bicubic interpolation: rotation,
   flips, residual scale and *sub-pixel* translation in a single step,
   straight into a screen-sized image with the background colour around it.

Everything here is pure Pillow/NumPy (no pygame display calls), so it runs on
a worker thread; Pillow releases the GIL while resampling.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from dataclasses import dataclass

import numpy as np
import pygame
from PIL import Image

from ..imaging.loader import level_index_for_scale
from .transforms import ViewTransform

log = logging.getLogger(__name__)

# Upper bound on source pixels fed to Lanczos for one frame (cost control).
MAX_CROP_PIXELS = 12_000_000
# Extra source pixels around the visible area so filters have support.
CROP_MARGIN = 4


def level_widths(pyramid) -> list[int]:
    src = getattr(pyramid, "source", pyramid)
    return src.widths


def available_arrays(pyramid) -> dict[int, np.ndarray]:
    """Level index -> pixel array for source or processed pyramids."""
    if hasattr(pyramid, "arrays_by_level"):
        return dict(pyramid.arrays_by_level)
    return dict(enumerate(pyramid.arrays))


def _visible_crop(
    t: ViewTransform, base_size: tuple[int, int], level_shape: tuple[int, int], screen: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Visible region of a level as (x0, y0, x1, y1), or None if off screen."""
    lh, lw = level_shape
    bw, bh = base_size
    kx, ky = lw / bw, lh / bh
    sw, sh = screen
    pts = [t.screen_to_image(c, base_size) for c in ((0, 0), (sw, 0), (0, sh), (sw, sh))]
    xs = [p[0] * kx for p in pts]
    ys = [p[1] * ky for p in pts]
    x0 = max(0, math.floor(min(xs)) - CROP_MARGIN)
    y0 = max(0, math.floor(min(ys)) - CROP_MARGIN)
    x1 = min(lw, math.ceil(max(xs)) + CROP_MARGIN)
    y1 = min(lh, math.ceil(max(ys)) + CROP_MARGIN)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def choose_level(
    arrays: dict[int, np.ndarray],
    widths: list[int],
    t: ViewTransform,
    base_size: tuple[int, int],
    screen: tuple[int, int],
) -> int:
    """One level finer than the display needs, if affordable and available."""
    n = len(widths)
    ideal = level_index_for_scale(widths, t.scale)
    order = [ideal - 1, ideal] + list(range(ideal + 1, n)) + list(range(ideal - 2, -1, -1))
    candidates = [i for i in order if 0 <= i < n and i in arrays]
    for i in candidates:
        crop = _visible_crop(t, base_size, arrays[i].shape[:2], screen)
        if crop is None:
            return i
        x0, y0, x1, y1 = crop
        if (x1 - x0) * (y1 - y0) <= MAX_CROP_PIXELS:
            return i
    return candidates[-1]  # last resort: coarsest available


def render_hq(
    arrays: dict[int, np.ndarray],
    widths: list[int],
    base_size: tuple[int, int],
    t: ViewTransform,
    screen: tuple[int, int],
    background: tuple[int, int, int],
) -> Image.Image:
    """Render a full screen-sized RGB frame at the best practical quality."""
    sw, sh = screen
    level = choose_level(arrays, widths, t, base_size, screen)
    arr = arrays[level]
    lh, lw = arr.shape[:2]
    crop = _visible_crop(t, base_size, (lh, lw), screen)
    if crop is None:
        return Image.new("RGB", screen, background)
    x0, y0, x1, y1 = crop
    cw, ch = x1 - x0, y1 - y0
    src = Image.fromarray(np.ascontiguousarray(arr[y0:y1, x0:x1]), "RGB")

    bw, bh = base_size
    kx, ky = lw / bw, lh / bh
    sx, sy = t.scale / kx, t.scale / ky  # screen px per level px
    if sx < 1.0 or sy < 1.0:
        # Shrink with a proper low-pass filter first; the affine step below
        # then only rotates/positions at ~1:1, where bicubic is excellent.
        rw = max(1, round(cw * min(sx, 1.0)))
        rh = max(1, round(ch * min(sy, 1.0)))
        src = src.resize((rw, rh), Image.Resampling.LANCZOS, reducing_gap=3.0)
    else:
        rw, rh = cw, ch
    fx, fy = rw / cw, rh / ch  # resized px per crop px

    # Affine map: output (screen) pixel coords -> resized-crop coords.
    # Pillow samples output pixel centres and treats input pixel i as
    # covering [i, i+1), the same continuous convention as ViewTransform.
    def to_src(q):
        px, py = t.screen_to_image(q, base_size)
        return ((px * kx - x0) * fx, (py * ky - y0) * fy)

    o = to_src((0.0, 0.0))
    ex = to_src((1.0, 0.0))
    ey = to_src((0.0, 1.0))
    coeffs = (ex[0] - o[0], ey[0] - o[0], o[0], ex[1] - o[1], ey[1] - o[1], o[1])
    return src.transform(
        (sw, sh),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=background,
    )


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HQResult:
    key: object
    surface: pygame.Surface


class HQRenderer:
    """Renders settle frames on a worker thread; only the latest request wins."""

    def __init__(self, wake_event_type: int | None = None) -> None:
        self.wake_event_type = wake_event_type
        self._cond = threading.Condition()
        self._request: tuple | None = None
        self._latest_key: object = None
        self._stop = False
        self._results: queue.Queue[HQResult] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="minibacklot-hq", daemon=True)
        self._thread.start()

    def submit(self, key: object, pyramid, transform: ViewTransform, screen, background) -> None:
        with self._cond:
            self._latest_key = key
            self._request = (
                key,
                available_arrays(pyramid),
                level_widths(pyramid),
                pyramid.base_size,
                transform,
                screen,
                background,
            )
            self._cond.notify()

    def cancel(self) -> None:
        with self._cond:
            self._latest_key = None
            self._request = None

    def poll(self) -> list[HQResult]:
        out = []
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                return out
            if res.key == self._latest_key:
                out.append(res)

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop = True
            self._request = None
            self._cond.notify()
        self._thread.join(timeout)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._request is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                req, self._request = self._request, None
            key, arrays, widths, base_size, transform, screen, bg = req
            try:
                img = render_hq(arrays, widths, base_size, transform, screen, bg)
                surf = pygame.image.frombytes(img.tobytes(), img.size, "RGB")
            except Exception:
                log.exception("High-quality render failed")
                continue
            self._results.put(HQResult(key, surf))
            if self.wake_event_type is not None:
                try:
                    pygame.event.post(pygame.event.Event(self.wake_event_type))
                except pygame.error:
                    pass
