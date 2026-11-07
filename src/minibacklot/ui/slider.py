"""Large touch slider and a labelled slider row with -/+ and reset buttons.

Slider behaviour:

* Touching the thumb (generous radius) grabs it *relatively*: the value
  does not jump, so a finger covering the thumb can make fine adjustments.
* Touching elsewhere on the track jumps to that position, then drags.
* Dragging near the default value snaps to it (a detent), so it's easy to
  return to neutral by feel.
* The track captures the finger, so dragging never pans the image.
"""

from __future__ import annotations

from typing import Callable

import pygame

from ..events.input_manager import PointerEvent
from .button import Button
from .theme import Theme
from .widget import Widget


class Slider(Widget):
    def __init__(
        self,
        minimum: float,
        maximum: float,
        default: float,
        step: float,
        get_value: Callable[[], float],
        set_value: Callable[[float], None],
        on_begin: Callable[[], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.minimum, self.maximum = float(minimum), float(maximum)
        self.default = float(default)
        self.step = step
        self.get_value = get_value
        self.set_value = set_value
        self.on_begin = on_begin
        self.on_end = on_end
        self.dragging = False
        self._grab_dx = 0.0
        self._pid = None
        self.thumb_radius = 14
        self.detent_px = 8

    # Geometry --------------------------------------------------------
    @property
    def _x0(self) -> float:
        return self.rect.left + self.thumb_radius

    @property
    def _x1(self) -> float:
        return self.rect.right - self.thumb_radius

    def value_to_x(self, v: float) -> float:
        span = self.maximum - self.minimum
        f = (v - self.minimum) / span if span else 0.0
        return self._x0 + max(0.0, min(1.0, f)) * (self._x1 - self._x0)

    def x_to_value(self, x: float) -> float:
        w = max(1.0, self._x1 - self._x0)
        f = max(0.0, min(1.0, (x - self._x0) / w))
        v = self.minimum + f * (self.maximum - self.minimum)
        if abs(x - self.value_to_x(self.default)) <= self.detent_px:
            return self.default
        if self.step:
            v = round(v / self.step) * self.step
        return max(self.minimum, min(self.maximum, v))

    # Input -----------------------------------------------------------
    def pointer_down(self, ev: PointerEvent) -> bool:
        thumb_x = self.value_to_x(self.get_value())
        if abs(ev.pos[0] - thumb_x) <= self.thumb_radius * 1.8:
            self._grab_dx = ev.pos[0] - thumb_x
        else:
            self._grab_dx = 0.0
        self.dragging = True
        self._pid = ev.pid
        if self.on_begin:
            self.on_begin()
        if self._grab_dx == 0.0:
            self.set_value(self.x_to_value(ev.pos[0]))
        return True

    def pointer_move(self, ev: PointerEvent) -> None:
        if self.dragging:
            self.set_value(self.x_to_value(ev.pos[0] - self._grab_dx))

    def pointer_up(self, ev: PointerEvent) -> None:
        self.pointer_cancel()

    def pointer_cancel(self) -> None:
        if self.dragging:
            self.dragging = False
            if self.on_end:
                self.on_end()

    # Drawing ---------------------------------------------------------
    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        cy = self.rect.centery
        track_h = theme.px(6)
        x0, x1 = round(self._x0), round(self._x1)
        pygame.draw.rect(surface, theme.slider_track, (x0, cy - track_h // 2, x1 - x0, track_h), border_radius=track_h)
        vx = round(self.value_to_x(self.get_value()))
        dx = round(self.value_to_x(self.default))
        lo, hi = sorted((vx, dx))
        if hi > lo:
            pygame.draw.rect(surface, theme.button_active, (lo, cy - track_h // 2, hi - lo, track_h))
        tick_h = theme.px(14)
        pygame.draw.line(surface, theme.button_fg, (dx, cy - tick_h), (dx, cy + tick_h), max(1, theme.px(2)))
        r = self.thumb_radius + (theme.px(3) if self.dragging else 0)
        pygame.draw.circle(surface, theme.button_active if self.dragging else theme.slider_thumb, (vx, cy), r)
        pygame.draw.circle(surface, theme.button_fg, (vx, cy), r, max(1, theme.px(2)))


class SliderRow:
    """Label + value on top; [-] [track] [+] [reset] below.

    Not a widget itself: it lays out and draws its children, and the panel
    routes pointer events to them.
    """

    def __init__(
        self,
        label: str,
        format_value: Callable[[float], str],
        slider: Slider,
        on_step: Callable[[int], None],
        on_reset: Callable[[], None],
    ) -> None:
        self.label = label
        self.format_value = format_value
        self.slider = slider
        self.minus = Button("-", lambda: on_step(-1), hint=f"Decrease {label.lower()} (hold to repeat)", repeat=True)
        self.plus = Button("+", lambda: on_step(1), hint=f"Increase {label.lower()} (hold to repeat)", repeat=True)
        self.reset = Button(
            "Reset",
            on_reset,
            hint=f"Reset {label.lower()}",
            icon="reset",
            is_enabled=lambda: slider.get_value() != slider.default,
        )
        self.rect = pygame.Rect(0, 0, 0, 0)
        self._label_rect = pygame.Rect(0, 0, 0, 0)

    @property
    def children(self) -> list[Widget]:
        return [self.minus, self.slider, self.plus, self.reset]

    def layout(self, x: int, y: int, width: int, theme: Theme) -> int:
        """Position children; return the row height."""
        label_h = theme.font(theme.FONT).get_linesize() + theme.px(4)
        bh = theme.px(56)
        gap = theme.px(6)
        self._label_rect = pygame.Rect(x, y, width, label_h)
        cy = y + label_h
        self.minus.rect = pygame.Rect(x, cy, bh, bh)
        self.reset.rect = pygame.Rect(x + width - bh, cy, bh, bh)
        self.plus.rect = pygame.Rect(self.reset.rect.left - gap - bh, cy, bh, bh)
        sx = self.minus.rect.right + gap
        self.slider.rect = pygame.Rect(sx, cy, max(theme.px(40), self.plus.rect.left - gap - sx), bh)
        self.slider.thumb_radius = theme.px(14)
        self.slider.detent_px = theme.px(8)
        self.rect = pygame.Rect(x, y, width, label_h + bh)
        return self.rect.height

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        font = theme.font(theme.FONT)
        changed = self.slider.get_value() != self.slider.default
        label = font.render(self.label, True, theme.button_fg)
        value = font.render(
            self.format_value(self.slider.get_value()),
            True,
            theme.value_changed if changed else theme.value_default,
        )
        surface.blit(label, label.get_rect(midleft=self._label_rect.midleft))
        surface.blit(value, value.get_rect(midright=self._label_rect.midright))
        for w in self.children:
            w.draw(surface, theme)
