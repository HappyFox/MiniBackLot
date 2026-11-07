"""Touch button with pressed feedback, toggle state, auto-repeat and long-press hint."""

from __future__ import annotations

import math
from typing import Callable

import pygame

from ..events.input_manager import PointerEvent
from .theme import Theme
from .widget import Widget

StrOrFn = str | Callable[[], str]

REPEAT_DELAY_MS = 450
REPEAT_INTERVAL_MS = 70


class Button(Widget):
    def __init__(
        self,
        label: StrOrFn,
        on_click: Callable[[], None],
        *,
        hint: str = "",
        is_active: Callable[[], bool] | None = None,
        is_enabled: Callable[[], bool] | None = None,
        style: str = "normal",  # normal | lock | danger
        width_units: float = 1.0,
        repeat: bool = False,
        icon: str | None = None,
    ) -> None:
        super().__init__()
        self._label = label
        self.on_click = on_click
        self.hint = hint
        self._is_active = is_active
        self._is_enabled = is_enabled
        self.style = style
        self.width_units = width_units
        self.pressed = False
        self.inside = False
        self.long_pressed = False
        self.press_t = 0
        # Auto-repeat buttons fire on press, then repeatedly while held.
        self.repeat = repeat
        self.icon = icon  # "reset" draws a circular arrow instead of text
        self._next_repeat = 0

    @property
    def label(self) -> str:
        return self._label() if callable(self._label) else self._label

    @property
    def active(self) -> bool:
        return bool(self._is_active and self._is_active())

    @property
    def enabled(self) -> bool:
        return self._is_enabled is None or bool(self._is_enabled())

    def preferred_width(self, theme: Theme) -> int:
        text_w = theme.font(theme.FONT).size(self.label)[0] + theme.px(24)
        return max(round(theme.px(theme.BUTTON_MIN_W) * self.width_units), text_w)

    def pointer_down(self, ev: PointerEvent) -> bool:
        self.pressed = True
        self.inside = True
        self.long_pressed = False
        self.press_t = ev.t
        if self.repeat and self.enabled:
            self._next_repeat = ev.t + REPEAT_DELAY_MS
            self.on_click()
        return True

    def pointer_move(self, ev: PointerEvent) -> None:
        # Allow some finger drift outside the button before cancelling.
        slop = max(8, self.rect.height // 4)
        self.inside = self.rect.inflate(slop * 2, slop * 2).collidepoint(ev.pos)

    def pointer_up(self, ev: PointerEvent) -> None:
        fire = self.pressed and self.inside and not self.long_pressed and self.enabled and not self.repeat
        self.pointer_cancel()
        if fire:
            self.on_click()

    def pointer_cancel(self) -> None:
        self.pressed = False
        self.inside = False

    def update(self, now: int) -> bool:
        if self.repeat and self.pressed and self.inside and self.enabled and now >= self._next_repeat:
            self._next_repeat = now + REPEAT_INTERVAL_MS
            self.long_pressed = True  # holding a repeat button never shows the hint
            self.on_click()
            return True
        return False

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        if not self.visible:
            return
        enabled = self.enabled
        if self.style == "lock":
            bg = theme.lock_on if self.active else theme.lock_off
        elif self.active:
            bg = theme.button_active
        elif self.style == "danger" and enabled:
            bg = theme.danger
        else:
            bg = theme.button_bg
        if self.pressed and self.inside and enabled:
            bg = tuple(min(255, c + 45) for c in bg)
        r = theme.px(theme.RADIUS)
        pygame.draw.rect(surface, bg, self.rect, border_radius=r)
        border = theme.button_fg if (self.style == "lock" and self.active) else theme.button_border
        pygame.draw.rect(surface, border, self.rect, width=max(1, theme.px(2)), border_radius=r)
        fg = theme.button_fg if enabled else theme.button_disabled_fg
        if self.icon == "reset":
            draw_reset_icon(surface, self.rect, fg, theme)
            return
        text = theme.font(theme.FONT).render(self.label, True, fg)
        surface.blit(text, text.get_rect(center=self.rect.center))


def draw_reset_icon(surface: pygame.Surface, rect: pygame.Rect, color, theme: Theme) -> None:
    """Circular arrow (the default font has no reliable arrow glyphs)."""
    r = min(rect.width, rect.height) * 0.26
    cx, cy = rect.center
    width = max(2, theme.px(3))
    box = pygame.Rect(0, 0, 2 * r, 2 * r)
    box.center = (cx, cy)
    start, stop = math.radians(100), math.radians(400)
    pygame.draw.arc(surface, color, box, start, stop, width)
    # Arrow head at the start of the arc (pygame arcs run counter-clockwise).
    ax, ay = cx + r * math.cos(start), cy - r * math.sin(start)
    h = r * 0.55
    pygame.draw.polygon(surface, color, [(ax - h, ay - h * 0.1), (ax + h * 0.3, ay - h * 0.9), (ax + h * 0.35, ay + h * 0.6)])
