"""Base class for touch widgets.

Widgets receive normalized :class:`PointerEvent` objects. A widget that
returns True from :meth:`pointer_down` captures that pointer: all further
moves and the release for the same pointer id are routed to it, even if the
finger leaves the widget. This keeps e.g. a slider drag from panning the
image underneath.
"""

from __future__ import annotations

import pygame

from ..events.input_manager import PointerEvent
from .theme import Theme


class Widget:
    def __init__(self) -> None:
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.visible = True

    @property
    def enabled(self) -> bool:
        return True

    def hit(self, pos: tuple[float, float]) -> bool:
        return self.visible and self.rect.collidepoint(pos)

    def preferred_width(self, theme: Theme) -> int:
        return theme.px(theme.BUTTON_MIN_W)

    def pointer_down(self, ev: PointerEvent) -> bool:
        return False

    def pointer_move(self, ev: PointerEvent) -> None:
        pass

    def pointer_up(self, ev: PointerEvent) -> None:
        pass

    def pointer_cancel(self) -> None:
        pass

    def update(self, now: int) -> bool:
        """Called each frame while the widget holds a pointer. Return True to redraw."""
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        pass
