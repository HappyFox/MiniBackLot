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


class EdgeHandle(Button):
    """A tab on the screen edge with chevrons that opens/closes a side panel.

    Tap to toggle, or swipe: left opens, right closes (the direction the
    panel moves). Long-press shows the hint like any other button.
    """

    SWIPE_PX = 24  # logical px of horizontal travel that counts as a swipe

    def __init__(self, on_toggle: Callable[[], None], is_open: Callable[[], bool], hint: str = "") -> None:
        super().__init__("", on_toggle, hint=hint, is_active=is_open)
        self._x0 = 0.0
        self.swipe_px = self.SWIPE_PX

    def pointer_down(self, ev: PointerEvent) -> bool:
        self._x0 = ev.pos[0]
        return super().pointer_down(ev)

    def pointer_up(self, ev: PointerEvent) -> None:
        dx = ev.pos[0] - self._x0
        was_inside, long_pressed = self.inside, self.long_pressed
        self.pointer_cancel()
        if long_pressed:
            return
        if abs(dx) >= self.swipe_px:
            if (dx < 0) != self.active:  # swipe left while closed / right while open
                self.on_click()
        elif was_inside:
            self.on_click()

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        if not self.visible:
            return
        r = theme.px(theme.RADIUS) * 2
        bg = theme.button_pressed if self.pressed else (40, 40, 46)
        rect = self.rect
        tab = pygame.Surface(rect.size, pygame.SRCALPHA)
        local = tab.get_rect()
        pygame.draw.rect(tab, (*bg, 225), local, border_top_left_radius=r, border_bottom_left_radius=r)
        pygame.draw.rect(
            tab, (*theme.button_border, 255), local, width=max(1, theme.px(2)),
            border_top_left_radius=r, border_bottom_left_radius=r,
        )
        surface.blit(tab, rect.topleft)
        # Double chevron pointing where the panel will move: << to open, >> to close.
        d = 1 if self.active else -1
        a, b = rect.width * 0.12, rect.width * 0.22
        width = max(2, theme.px(3))
        cy = rect.centery
        for off in (-0.13, 0.13):
            cx = rect.centerx + off * rect.width
            pts = [(cx - d * a, cy - b), (cx + d * a, cy), (cx - d * a, cy + b)]
            pygame.draw.lines(surface, theme.button_fg, False, pts, width)


class Tab(Button):
    """One tab of a tab bar.

    The selected tab shares the content area's background and has no bottom
    edge, so it reads as the front card joined to its page; it gets an accent
    strip along the top. Unselected tabs are recessed and dimmer. Tapping the
    selected tab does nothing.
    """

    def __init__(self, label: str, on_select: Callable[[], None], is_active: Callable[[], bool]) -> None:
        super().__init__(label, on_select, is_active=is_active)

    def pointer_up(self, ev: PointerEvent) -> None:
        if self.active:
            self.pointer_cancel()
            return
        super().pointer_up(ev)

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        if not self.visible:
            return
        active = self.active
        rect = self.rect
        r = theme.px(theme.RADIUS)
        w = max(1, theme.px(2))
        if active:
            bg = theme.tab_active_bg
        else:
            bg = theme.tab_pressed_bg if (self.pressed and self.inside) else theme.tab_inactive_bg
        corners = {"border_top_left_radius": r, "border_top_right_radius": r}
        pygame.draw.rect(surface, bg, rect, **corners)
        pygame.draw.rect(surface, theme.button_border, rect, width=w, **corners)
        if active:
            # Open the bottom edge (and the bar's baseline) into the page below.
            pygame.draw.rect(surface, bg, (rect.left + w, rect.bottom - w, rect.width - 2 * w, w))
            strip = theme.px(4)
            pygame.draw.rect(
                surface, theme.button_active, (rect.left + r, rect.top + w, rect.width - 2 * r, strip),
                border_radius=strip // 2,
            )
        fg = theme.header_fg if active else theme.tab_inactive_fg
        text = theme.font(theme.FONT).render(self.label, True, fg)
        surface.blit(text, text.get_rect(center=(rect.centerx, rect.centery + (theme.px(2) if active else 0))))
