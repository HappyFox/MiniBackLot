"""View controller: owns the image transform, display mode, lock and undo.

All image-position changes go through this class so that locking and undo
are enforced in one place, independent of whether the request came from a
touch gesture, the mouse, the keyboard or a toolbar button.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from .transforms import (
    Size,
    Vec,
    ViewTransform,
    centered,
    fill_scale,
    fit_scale,
    hitch_axis,
    normalize_angle,
    snap_rotation,
)


class DisplayMode(str, Enum):
    FIT = "fit"
    FILL = "fill"
    ACTUAL = "actual"
    FREE = "free"  # user has zoomed/panned/rotated manually


@dataclass(frozen=True)
class ViewSnapshot:
    transform: ViewTransform
    mode: DisplayMode


# Never let the image become smaller than this on screen (longest side).
MIN_DISPLAY_PIXELS = 32.0
# Maximum magnification (screen pixels per image pixel).
MAX_ZOOM = 32.0


class ViewController:
    def __init__(
        self,
        screen_size: Size,
        default_mode: DisplayMode = DisplayMode.FIT,
        history_limit: int = 100,
    ) -> None:
        self.screen_size: Size = screen_size
        self.image_size: Size | None = None
        self.default_mode = default_mode if default_mode != DisplayMode.FREE else DisplayMode.FIT
        self.mode: DisplayMode = self.default_mode
        self.transform = ViewTransform(cx=screen_size[0] / 2, cy=screen_size[1] / 2)
        self.locked = False
        self._history: deque[ViewSnapshot] = deque(maxlen=history_limit)
        # Incremented on every change so renderers can cheaply detect changes.
        self.revision = 0
        # Called whenever a snapshot is recorded (for an app-wide undo stack).
        self.on_history_push: Callable[[], None] | None = None
        # Sticky rotation snapping: the unsnapped angle accumulated during a
        # rotation, and the angle we last displayed (to detect outside changes).
        self._raw_rotation: float | None = None
        self._snap_shown: float | None = None
        # Sticky edge "hitch" while panning: unsnapped centre and last shown centre.
        self._raw_center: tuple[float, float] | None = None
        self._hitch_shown: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set(self, transform: ViewTransform, mode: DisplayMode) -> None:
        if transform != self.transform or mode != self.mode:
            self.transform = transform
            self.mode = mode
            self.revision += 1

    def _scale_limits(self) -> tuple[float, float]:
        if self.image_size is None:
            return (1e-3, MAX_ZOOM)
        min_scale = MIN_DISPLAY_PIXELS / max(self.image_size)
        # Never forbid fit scale, even for tiny images on huge screens.
        min_scale = min(min_scale, self._mode_scale(DisplayMode.FIT))
        return (min_scale, max(MAX_ZOOM, self._mode_scale(DisplayMode.FILL)))

    def _mode_scale(self, mode: DisplayMode) -> float:
        assert self.image_size is not None
        rot = self.transform.rotation
        if mode == DisplayMode.FIT:
            return fit_scale(self.image_size, self.screen_size, rot)
        if mode == DisplayMode.FILL:
            return fill_scale(self.image_size, self.screen_size, rot)
        return 1.0

    def _layout_for_mode(self, mode: DisplayMode) -> ViewTransform:
        assert self.image_size is not None
        return centered(self.image_size, self.screen_size, self._mode_scale(mode), self.transform)

    @property
    def can_edit(self) -> bool:
        return not self.locked and self.image_size is not None

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def snapshot(self) -> ViewSnapshot:
        return ViewSnapshot(self.transform, self.mode)

    def push_history(self) -> None:
        """Record the current state; call at the *start* of an edit."""
        snap = self.snapshot()
        if not self._history or self._history[-1] != snap:
            self._history.append(snap)
            if self.on_history_push:
                self.on_history_push()

    @property
    def history_size(self) -> int:
        return len(self._history)

    @property
    def can_undo(self) -> bool:
        return bool(self._history) and not self.locked

    def undo(self) -> bool:
        if not self.can_undo:
            return False
        # Skip entries identical to the current state (e.g. a tap-only gesture).
        while self._history:
            snap = self._history.pop()
            if snap != self.snapshot():
                self._set(snap.transform, snap.mode)
                return True
        return False

    def clear_history(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Image / screen changes
    # ------------------------------------------------------------------
    def set_image(self, image_size: Size) -> None:
        """Show a new image.

        When locked, a manually positioned view is kept as-is so that
        switching backdrops does not disturb a carefully framed setup.
        Otherwise the current display mode (or the default mode, when the
        view was free) is applied.
        """
        self.image_size = image_size
        if self.locked and self.mode == DisplayMode.FREE:
            self.revision += 1
            return
        mode = self.mode if self.mode != DisplayMode.FREE else self.default_mode
        self._history.clear()
        self._set(self._layout_for_mode(mode), mode)
        self.revision += 1

    def set_screen_size(self, screen_size: Size) -> None:
        if screen_size == self.screen_size or min(screen_size) <= 0:
            return
        old = self.screen_size
        self.screen_size = screen_size
        if self.image_size is not None and self.mode != DisplayMode.FREE:
            self._set(self._layout_for_mode(self.mode), self.mode)
        else:
            self._set(self.transform.resized(old, screen_size), self.mode)
        # Old history entries are expressed in the old screen geometry.
        self._history = deque(
            (ViewSnapshot(s.transform.resized(old, screen_size), s.mode) for s in self._history),
            maxlen=self._history.maxlen,
        )
        self.revision += 1

    # ------------------------------------------------------------------
    # Edits. Each returns True if the view changed, False if refused.
    # Continuous edits (pan/zoom/rotate) do not push history themselves;
    # callers push once at the start of a gesture.
    # ------------------------------------------------------------------
    def pan(self, dx: float, dy: float, hitch_px: float = 0.0) -> bool:
        """Translate the image.

        With ``hitch_px > 0`` the image *sticks* when one of its edges comes
        within ``hitch_px`` of the matching screen edge: it is held exactly
        flush while the finger keeps moving, and released once the finger has
        moved past. This makes edge alignment easy without preventing the
        image from being moved off screen. Call :meth:`end_pan` when the
        gesture ends.
        """
        if not self.can_edit or (dx == 0 and dy == 0):
            return False
        if hitch_px <= 0 or self.image_size is None:
            self.end_pan()
            self._set(self.transform.translated(dx, dy), DisplayMode.FREE)
            return True
        t = self.transform
        cur = (t.cx, t.cy)
        if self._raw_center is None or self._hitch_shown != cur:
            self._raw_center = cur  # (re)start: moved by something else
        rx, ry = self._raw_center[0] + dx, self._raw_center[1] + dy
        self._raw_center = (rx, ry)
        hx, hy = t.half_extents(self.image_size)
        sw, sh = self.screen_size
        shown = (hitch_axis(rx, hx, sw, hitch_px), hitch_axis(ry, hy, sh, hitch_px))
        self._hitch_shown = shown
        if shown == cur:
            return False  # held at an edge
        self._set(replace(t, cx=shown[0], cy=shown[1]), DisplayMode.FREE)
        return True

    def end_pan(self) -> None:
        self._raw_center = None
        self._hitch_shown = None

    def zoom(self, anchor: Vec, factor: float) -> bool:
        if not self.can_edit:
            return False
        lo, hi = self._scale_limits()
        t = self.transform.zoomed_about(anchor, factor, lo, hi)
        if t == self.transform:
            return False
        self._set(t, DisplayMode.FREE)
        return True

    def rotate(self, anchor: Vec, degrees: float, snap_deg: float = 0.0) -> bool:
        """Rotate clockwise around ``anchor``.

        With ``snap_deg > 0`` the rotation is *sticky* near 0/90/180/270:
        the unsnapped angle keeps accumulating, and the displayed angle locks
        to the right angle while the raw one is within ``snap_deg`` of it,
        releasing once the user rotates past. Call :meth:`end_rotation` when
        the gesture ends.
        """
        if not self.can_edit or degrees == 0:
            return False
        cur = self.transform.rotation
        if snap_deg <= 0:
            self.end_rotation()
            self._set(self.transform.rotated_about(anchor, degrees), DisplayMode.FREE)
            return True
        if self._raw_rotation is None or self._snap_shown != cur:
            self._raw_rotation = cur  # (re)start: rotation changed elsewhere
        self._raw_rotation = normalize_angle(self._raw_rotation + degrees)
        target = normalize_angle(snap_rotation(self._raw_rotation, snap_deg))
        delta = normalize_angle(target - cur)
        self._snap_shown = target
        if delta == 0:
            return False  # held at a snap angle
        t = replace(self.transform.rotated_about(anchor, delta), rotation=target)
        self._set(t, DisplayMode.FREE)
        return True

    def end_rotation(self) -> None:
        self._raw_rotation = None
        self._snap_shown = None

    def set_mode(self, mode: DisplayMode) -> bool:
        if not self.can_edit or mode == DisplayMode.FREE:
            return False
        self.push_history()
        self._set(self._layout_for_mode(mode), mode)
        return True

    def reset_zoom(self) -> bool:
        """Fit the image again, keeping rotation and flips (double-tap)."""
        return self.set_mode(DisplayMode.FIT)

    def reset_all(self) -> bool:
        """Fit with no rotation and no flips."""
        if not self.can_edit:
            return False
        self.push_history()
        self.transform = replace(self.transform, rotation=0.0, flip_h=False, flip_v=False)
        self._set(self._layout_for_mode(DisplayMode.FIT), DisplayMode.FIT)
        self.revision += 1
        return True

    def straighten(self) -> bool:
        """Remove rotation, keeping position, zoom and flips."""
        if not self.can_edit or self.transform.rotation == 0:
            return False
        self.push_history()
        t = replace(self.transform, rotation=0.0)
        mode = self.mode
        if mode in (DisplayMode.FIT, DisplayMode.FILL):
            self.transform = t
            t = self._layout_for_mode(mode)
        self._set(t, mode)
        return True

    def flip_horizontal(self) -> bool:
        if not self.can_edit:
            return False
        self.push_history()
        self._set(self.transform.mirrored_horizontally(), self.mode)
        return True

    def flip_vertical(self) -> bool:
        if not self.can_edit:
            return False
        self.push_history()
        self._set(self.transform.mirrored_vertically(), self.mode)
        return True

    def set_locked(self, locked: bool) -> None:
        if locked != self.locked:
            self.locked = locked
            self.revision += 1

    def toggle_lock(self) -> bool:
        self.set_locked(not self.locked)
        return self.locked
