"""Image-to-screen transformation model.

A :class:`ViewTransform` maps *image pixel coordinates* (origin at the image's
top-left corner) to *screen pixel coordinates* (origin at the window's
top-left corner, y pointing down).

The mapping is::

    screen = center + S * R(rotation) * F * (p - image_center)

where ``F`` applies the horizontal/vertical flips in image space, ``R`` is a
rotation that is **clockwise-positive on screen** (y-down coordinates) and
``S`` is a uniform scale (screen pixels per image pixel).

The transform is immutable; every operation returns a new instance, which
makes undo history and testing trivial. It knows nothing about pygame or the
UI, so it can be tested in isolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

Vec = tuple[float, float]
Size = tuple[int, int]

DEFAULT_MIN_SCALE = 1e-3
DEFAULT_MAX_SCALE = 64.0


def normalize_angle(degrees: float) -> float:
    """Normalize an angle to the half-open range (-180, 180]."""
    a = math.fmod(degrees, 360.0)
    if a <= -180.0:
        a += 360.0
    elif a > 180.0:
        a -= 360.0
    return a


def snap_rotation(angle: float, threshold: float) -> float:
    """Return the nearest multiple of 90 degrees if within ``threshold``, else ``angle``."""
    if threshold <= 0:
        return angle
    m = round(angle / 90.0) * 90.0
    return normalize_angle(m) if abs(angle - m) <= threshold else angle


def hitch_axis(raw: float, half_extent: float, screen_len: float, hitch: float) -> float:
    """Sticky edge alignment along one axis.

    ``raw`` is the unsnapped screen position of the image centre. Returns the
    centre position that puts an image edge exactly on the matching screen
    edge (near edge on 0, far edge on ``screen_len``) when ``raw`` is within
    ``hitch`` of it, otherwise ``raw`` unchanged.
    """
    if hitch <= 0:
        return raw
    best, best_d = raw, hitch
    for target in (half_extent, screen_len - half_extent):
        d = abs(raw - target)
        if d <= best_d:
            best, best_d = target, d
    return best


@dataclass(frozen=True)
class ViewTransform:
    cx: float = 0.0  # screen x of the image center
    cy: float = 0.0  # screen y of the image center
    scale: float = 1.0  # screen pixels per image pixel
    rotation: float = 0.0  # degrees, clockwise on screen, in (-180, 180]
    flip_h: bool = False  # mirror in image space (left/right)
    flip_v: bool = False  # mirror in image space (top/bottom)

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------
    def matrix(self) -> tuple[float, float, float, float]:
        """Return the 2x2 linear part ``S*R*F`` as ``(a, b, c, d)``."""
        r = math.radians(self.rotation)
        cos_r, sin_r = math.cos(r), math.sin(r)
        fx = -1.0 if self.flip_h else 1.0
        fy = -1.0 if self.flip_v else 1.0
        s = self.scale
        return (s * cos_r * fx, -s * sin_r * fy, s * sin_r * fx, s * cos_r * fy)

    def image_to_screen(self, p: Vec, image_size: Size) -> Vec:
        a, b, c, d = self.matrix()
        x = p[0] - image_size[0] / 2.0
        y = p[1] - image_size[1] / 2.0
        return (self.cx + a * x + b * y, self.cy + c * x + d * y)

    def screen_to_image(self, q: Vec, image_size: Size) -> Vec:
        a, b, c, d = self.matrix()
        det = a * d - b * c
        x = q[0] - self.cx
        y = q[1] - self.cy
        ix = (d * x - b * y) / det
        iy = (-c * x + a * y) / det
        return (ix + image_size[0] / 2.0, iy + image_size[1] / 2.0)

    # ------------------------------------------------------------------
    # Operations (all return new transforms)
    # ------------------------------------------------------------------
    def translated(self, dx: float, dy: float) -> ViewTransform:
        return replace(self, cx=self.cx + dx, cy=self.cy + dy)

    def zoomed_about(
        self,
        anchor: Vec,
        factor: float,
        min_scale: float = DEFAULT_MIN_SCALE,
        max_scale: float = DEFAULT_MAX_SCALE,
    ) -> ViewTransform:
        """Scale by ``factor`` keeping the image point under ``anchor`` fixed.

        The resulting scale is clamped to ``[min_scale, max_scale]``; the
        anchor is still honoured with the clamped factor so the image never
        jumps when a limit is hit.
        """
        if factor <= 0 or not math.isfinite(factor):
            return self
        new_scale = min(max(self.scale * factor, min_scale), max_scale)
        k = new_scale / self.scale
        ax, ay = anchor
        return replace(
            self,
            scale=new_scale,
            cx=ax + k * (self.cx - ax),
            cy=ay + k * (self.cy - ay),
        )

    def rotated_about(self, anchor: Vec, degrees: float) -> ViewTransform:
        """Rotate clockwise by ``degrees`` around a screen point."""
        r = math.radians(degrees)
        cos_r, sin_r = math.cos(r), math.sin(r)
        ax, ay = anchor
        x, y = self.cx - ax, self.cy - ay
        return replace(
            self,
            rotation=normalize_angle(self.rotation + degrees),
            cx=ax + cos_r * x - sin_r * y,
            cy=ay + sin_r * x + cos_r * y,
        )

    def mirrored_horizontally(self) -> ViewTransform:
        """Mirror the *displayed* image left/right about its center.

        Mirroring on screen while rotated: ``Fx * R(t) = R(-t) * Fx``, so the
        rotation is negated and the image-space flip toggled.
        """
        return replace(self, rotation=normalize_angle(-self.rotation), flip_h=not self.flip_h)

    def mirrored_vertically(self) -> ViewTransform:
        """Mirror the *displayed* image top/bottom about its center."""
        return replace(self, rotation=normalize_angle(-self.rotation), flip_v=not self.flip_v)

    def resized(self, old_screen: Size, new_screen: Size) -> ViewTransform:
        """Adapt to a new screen size so the image keeps its relative framing."""
        if min(old_screen) <= 0 or min(new_screen) <= 0:
            return self
        k = min(new_screen[0] / old_screen[0], new_screen[1] / old_screen[1])
        ocx, ocy = old_screen[0] / 2.0, old_screen[1] / 2.0
        ncx, ncy = new_screen[0] / 2.0, new_screen[1] / 2.0
        return replace(
            self,
            scale=self.scale * k,
            cx=ncx + k * (self.cx - ocx),
            cy=ncy + k * (self.cy - ocy),
        )

    def half_extents(self, image_size: Size) -> tuple[float, float]:
        """Half width/height of the image's on-screen axis-aligned bounding box."""
        r = math.radians(self.rotation)
        c, s = abs(math.cos(r)), abs(math.sin(r))
        w, h = image_size
        return (self.scale * (w * c + h * s) / 2.0, self.scale * (w * s + h * c) / 2.0)

    def is_close(self, other: ViewTransform, tol: float = 1e-6) -> bool:
        return (
            abs(self.cx - other.cx) <= tol
            and abs(self.cy - other.cy) <= tol
            and abs(self.scale - other.scale) <= tol * max(1.0, self.scale)
            and abs(normalize_angle(self.rotation - other.rotation)) <= tol
            and self.flip_h == other.flip_h
            and self.flip_v == other.flip_v
        )


# ----------------------------------------------------------------------
# Display-mode scale calculations
# ----------------------------------------------------------------------
def _abs_cos_sin(rotation: float) -> tuple[float, float]:
    r = math.radians(rotation)
    return abs(math.cos(r)), abs(math.sin(r))


def fit_scale(image_size: Size, screen_size: Size, rotation: float = 0.0) -> float:
    """Largest scale at which the (rotated) image fits entirely on screen."""
    w, h = image_size
    sw, sh = screen_size
    c, s = _abs_cos_sin(rotation)
    bw = w * c + h * s
    bh = w * s + h * c
    return min(sw / bw, sh / bh)


def fill_scale(image_size: Size, screen_size: Size, rotation: float = 0.0) -> float:
    """Smallest scale at which the (rotated) image covers the whole screen."""
    w, h = image_size
    sw, sh = screen_size
    c, s = _abs_cos_sin(rotation)
    return max((sw * c + sh * s) / w, (sw * s + sh * c) / h)


def centered(
    image_size: Size, screen_size: Size, scale: float, base: ViewTransform | None = None
) -> ViewTransform:
    """A transform with the image centered on screen at ``scale``.

    Rotation and flips are taken from ``base`` when given.
    """
    base = base or ViewTransform()
    return replace(base, cx=screen_size[0] / 2.0, cy=screen_size[1] / 2.0, scale=scale)
