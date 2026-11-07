"""Render an :class:`ImagePyramid` to a surface using a :class:`ViewTransform`.

Strategy, chosen so that 24+ MP images stay responsive:

1. Pick the pyramid level closest to (but not below) the displayed
   resolution, so we never downscale by more than 2x per frame.
2. Map the screen corners back into that level and crop the visible
   bounding box, so we only ever scale/rotate roughly a screen's worth of
   pixels regardless of zoom.
3. Flip, scale and rotate the crop about its center and blit it so the crop
   center lands where the transform says it should.

``smooth=False`` uses nearest-neighbour scaling/rotation for interactive
gestures; ``smooth=True`` uses ``smoothscale``/``rotozoom`` for the final
frame once interaction settles.

Note: ``pygame.transform.rotate`` pads 24-bit surfaces with the top-left pixel
colour, so rotated crops are converted to per-pixel alpha first.
"""

from __future__ import annotations

import math

import pygame

from ..imaging.loader import ImagePyramid, ProcessedPyramid
from .transforms import ViewTransform


class Renderer:
    def __init__(self, background: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.background = background
        self._frame: pygame.Surface | None = None
        self._frame_key: tuple | None = None

    def invalidate(self) -> None:
        self._frame_key = None

    def frame(
        self,
        size: tuple[int, int],
        pyramid: ImagePyramid | ProcessedPyramid | None,
        transform: ViewTransform,
        smooth: bool,
    ) -> pygame.Surface:
        """Return a cached frame, re-rendering only if inputs changed."""
        key = (size, id(pyramid), getattr(pyramid, "version", 0), transform, smooth, self.background)
        if self._frame is None or self._frame.get_size() != size:
            self._frame = pygame.Surface(size)
            self._frame_key = None
        if key != self._frame_key:
            self.render(self._frame, pyramid, transform, smooth)
            self._frame_key = key
        return self._frame

    def render(
        self,
        target: pygame.Surface,
        pyramid: ImagePyramid | ProcessedPyramid | None,
        t: ViewTransform,
        smooth: bool = True,
    ) -> None:
        target.fill(self.background)
        if pyramid is None:
            return
        _, level = pyramid.level_for_scale(t.scale)
        bw, bh = pyramid.base_size
        lw, lh = level.get_size()
        kx, ky = lw / bw, lh / bh
        sw, sh = target.get_size()

        # Visible region in level coordinates.
        corners = [(0, 0), (sw, 0), (0, sh), (sw, sh)]
        pts = [t.screen_to_image(c, (bw, bh)) for c in corners]
        xs = [p[0] * kx for p in pts]
        ys = [p[1] * ky for p in pts]
        x0 = max(0, math.floor(min(xs)) - 2)
        y0 = max(0, math.floor(min(ys)) - 2)
        x1 = min(lw, math.ceil(max(xs)) + 2)
        y1 = min(lh, math.ceil(max(ys)) + 2)
        if x1 <= x0 or y1 <= y0:
            return  # image entirely off screen

        crop = level.subsurface((x0, y0, x1 - x0, y1 - y0))
        cw, ch = crop.get_size()
        s = t.scale / kx  # screen pixels per level pixel
        center_base = ((x0 + x1) / 2.0 / kx, (y0 + y1) / 2.0 / ky)
        scx, scy = t.image_to_screen(center_base, (bw, bh))

        if t.flip_h or t.flip_v:
            crop = pygame.transform.flip(crop, t.flip_h, t.flip_v)

        out_w = max(1, round(cw * s))
        out_h = max(1, round(ch * s))
        rot = t.rotation % 360.0
        if rot == 0.0:
            if (out_w, out_h) == (cw, ch):
                out = crop
            elif smooth:
                out = pygame.transform.smoothscale(crop, (out_w, out_h))
            else:
                out = pygame.transform.scale(crop, (out_w, out_h))
        elif rot in (90.0, 180.0, 270.0):
            scaled = (
                pygame.transform.smoothscale(crop, (out_w, out_h))
                if smooth
                else pygame.transform.scale(crop, (out_w, out_h))
            )
            out = pygame.transform.rotate(scaled, -rot)  # exact, no padding
        else:
            crop = crop.convert_alpha() if pygame.display.get_surface() else _with_alpha(crop)
            if smooth:
                out = pygame.transform.rotozoom(crop, -t.rotation, s)
            else:
                out = pygame.transform.rotate(pygame.transform.scale(crop, (out_w, out_h)), -t.rotation)

        ow, oh = out.get_size()
        target.blit(out, (round(scx - ow / 2.0), round(scy - oh / 2.0)))


def _with_alpha(surface: pygame.Surface) -> pygame.Surface:
    out = pygame.Surface(surface.get_size(), pygame.SRCALPHA, 32)
    out.blit(surface, (0, 0))
    return out
