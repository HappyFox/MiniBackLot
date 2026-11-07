"""Dark, high-contrast theme with resolution-dependent scaling.

All sizes are defined in *logical pixels* and multiplied by ``scale``.
Automatic scaling uses the window's shorter side, since SDL2 does not give
reliable physical DPI on Linux (especially under Wayland). Users can
override it with the ``ui_scale`` setting or ``--ui-scale``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pygame

Color = tuple[int, int, int]


def auto_ui_scale(screen_size: tuple[int, int]) -> float:
    return max(1.0, min(3.0, min(screen_size) / 1080.0))


@dataclass
class Theme:
    scale: float = 1.0
    panel_bg: tuple[int, int, int, int] = (16, 16, 18, 215)
    button_bg: Color = (44, 44, 50)
    button_pressed: Color = (90, 90, 100)
    button_active: Color = (30, 110, 200)
    button_disabled_fg: Color = (100, 100, 105)
    button_fg: Color = (235, 235, 240)
    button_border: Color = (75, 75, 85)
    lock_on: Color = (205, 60, 45)
    lock_off: Color = (44, 44, 50)
    danger: Color = (150, 40, 35)
    toast_bg: tuple[int, int, int, int] = (0, 0, 0, 200)
    toast_fg: Color = (245, 245, 245)
    slider_track: Color = (80, 80, 90)
    slider_thumb: Color = (215, 215, 220)
    value_default: Color = (150, 150, 158)
    value_changed: Color = (255, 205, 90)
    header_fg: Color = (255, 255, 255)
    # Tab bar: the selected tab shares the page background.
    page_bg: Color = (32, 32, 38)
    tab_active_bg: Color = (32, 32, 38)
    tab_inactive_bg: Color = (20, 20, 23)
    tab_pressed_bg: Color = (44, 44, 52)
    tab_inactive_fg: Color = (150, 150, 158)
    _fonts: dict[int, pygame.font.Font] = field(default_factory=dict, repr=False)

    # Logical sizes
    BUTTON_H = 64
    BUTTON_MIN_W = 76
    GAP = 8
    PANEL_PAD = 10
    FONT = 20
    SMALL_FONT = 16
    TOAST_FONT = 24
    RADIUS = 10

    def px(self, logical: float) -> int:
        return max(1, round(logical * self.scale))

    def font(self, logical_size: int) -> pygame.font.Font:
        size = self.px(logical_size)
        f = self._fonts.get(size)
        if f is None:
            f = pygame.font.Font(None, round(size * 1.35))  # default font runs small
            self._fonts[size] = f
        return f
