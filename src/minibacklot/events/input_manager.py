"""Normalize pygame-ce touch and mouse events into pointer events.

Touch is the primary input. SDL can synthesize events in both directions:

* touch -> mouse: SDL emits MOUSEBUTTONDOWN/MOTION/UP for the first finger.
  pygame-ce marks these with ``event.touch == True``. We additionally set the
  ``SDL_TOUCH_MOUSE_EVENTS=0`` hint (see :func:`configure_sdl_hints`) and,
  as a last line of defence, ignore mouse events arriving shortly after any
  finger event.
* mouse -> touch: with ``SDL_MOUSE_TOUCH_EVENTS=1`` SDL emits FINGER events
  for the mouse with ``touch_id == SDL_MOUSE_TOUCHID`` (-1). We disable the
  hint and drop such events.

Finger coordinates arrive normalized (0..1) relative to the window and are
converted to window pixels here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Hashable

import pygame

Vec = tuple[float, float]

# SDL_MOUSE_TOUCHID is ((SDL_TouchID)-1); some builds report it unsigned.
_MOUSE_TOUCH_IDS = {-1, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF}

MOUSE_POINTER_ID = ("mouse", 0)


def configure_sdl_hints() -> None:
    """Set SDL hints. Must be called before ``pygame.init()``.

    Values already present in the environment are respected so users can
    override them.
    """
    os.environ.setdefault("SDL_TOUCH_MOUSE_EVENTS", "0")
    os.environ.setdefault("SDL_MOUSE_TOUCH_EVENTS", "0")
    os.environ.setdefault("SDL_VIDEO_ALLOW_SCREENSAVER", "0")
    # Don't minimize the fullscreen window when focus moves (e.g. a
    # notification or on-screen keyboard appears).
    os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")


class PointerKind(Enum):
    DOWN = auto()
    MOVE = auto()
    UP = auto()
    HOVER = auto()  # real mouse motion with no button held


class PointerSource(Enum):
    TOUCH = auto()
    MOUSE = auto()


@dataclass(frozen=True)
class PointerEvent:
    kind: PointerKind
    pid: Hashable
    pos: Vec
    source: PointerSource
    t: int  # milliseconds


class InputManager:
    def __init__(self, window_size: tuple[int, int], mouse_suppress_ms: int = 500) -> None:
        self.window_size = window_size
        self.mouse_suppress_ms = mouse_suppress_ms
        self._last_touch_t: int | None = None
        self._mouse_down = False

    def set_window_size(self, size: tuple[int, int]) -> None:
        self.window_size = size

    # ------------------------------------------------------------------
    def is_synthetic_mouse(self, event: pygame.event.Event, t: int) -> bool:
        """True if a mouse event was generated from touch input."""
        if getattr(event, "touch", False):
            return True
        if self._last_touch_t is not None and t - self._last_touch_t < self.mouse_suppress_ms:
            return True
        return False

    def translate(self, event: pygame.event.Event, t: int) -> list[PointerEvent]:
        """Convert one pygame event into zero or more pointer events."""
        et = event.type
        if et in (pygame.FINGERDOWN, pygame.FINGERMOTION, pygame.FINGERUP):
            if getattr(event, "touch_id", None) in _MOUSE_TOUCH_IDS:
                return []  # mouse-synthesized touch
            self._last_touch_t = t
            w, h = self.window_size
            pos = (event.x * w, event.y * h)
            pid = ("touch", event.touch_id, event.finger_id)
            kind = {
                pygame.FINGERDOWN: PointerKind.DOWN,
                pygame.FINGERMOTION: PointerKind.MOVE,
                pygame.FINGERUP: PointerKind.UP,
            }[et]
            return [PointerEvent(kind, pid, pos, PointerSource.TOUCH, t)]

        if et == pygame.MOUSEBUTTONDOWN:
            if event.button != 1 or self.is_synthetic_mouse(event, t):
                return []
            self._mouse_down = True
            return [PointerEvent(PointerKind.DOWN, MOUSE_POINTER_ID, _fpos(event.pos), PointerSource.MOUSE, t)]

        if et == pygame.MOUSEBUTTONUP:
            if event.button != 1 or not self._mouse_down:
                return []
            # Always deliver the release of a real press, even if touch
            # events arrived meanwhile, so the pointer never gets stuck.
            self._mouse_down = False
            return [PointerEvent(PointerKind.UP, MOUSE_POINTER_ID, _fpos(event.pos), PointerSource.MOUSE, t)]

        if et == pygame.MOUSEMOTION:
            if self._mouse_down:
                return [PointerEvent(PointerKind.MOVE, MOUSE_POINTER_ID, _fpos(event.pos), PointerSource.MOUSE, t)]
            if self.is_synthetic_mouse(event, t):
                return []
            return [PointerEvent(PointerKind.HOVER, MOUSE_POINTER_ID, _fpos(event.pos), PointerSource.MOUSE, t)]

        return []

    def reset(self) -> None:
        """Forget button state (e.g. on focus loss)."""
        self._mouse_down = False


def _fpos(p: tuple[int, int]) -> Vec:
    return (float(p[0]), float(p[1]))
