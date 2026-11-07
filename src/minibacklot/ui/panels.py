"""Toolbar container and transient toast messages."""

from __future__ import annotations

from typing import Hashable

import pygame

from ..events.input_manager import PointerEvent, PointerKind
from .button import Button, EdgeHandle
from .slider import SliderRow
from .theme import Theme
from .widget import Widget


def _long_press_hint(widgets, now: int, long_press_ms: int) -> str | None:
    for w in widgets:
        if (
            isinstance(w, Button)
            and w.pressed
            and w.inside
            and not w.long_pressed
            and now - w.press_t >= long_press_ms
        ):
            w.long_pressed = True
            return w.hint or w.label
    return None


class Toolbar:
    """A bottom drawer of widgets that wraps onto several rows if needed.

    The toolbar consumes every pointer that lands on its background, not
    just on its buttons, so taps in the gaps never reach the image.
    """

    def __init__(self, widgets: list[Widget], long_press_ms: int = 600) -> None:
        self.widgets = widgets
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.visible = True
        self.long_press_ms = long_press_ms
        self._captures: dict[Hashable, Widget | None] = {}

    # ------------------------------------------------------------------
    def layout(self, screen_size: tuple[int, int], theme: Theme) -> None:
        sw, sh = screen_size
        gap, pad, bh = theme.px(theme.GAP), theme.px(theme.PANEL_PAD), theme.px(theme.BUTTON_H)
        max_w = sw - 2 * pad
        rows: list[list[tuple[Widget, int]]] = [[]]
        row_w = 0
        for w in self.widgets:
            ww = min(w.preferred_width(theme), max_w)
            needed = ww if not rows[-1] else row_w + gap + ww
            if rows[-1] and needed > max_w:
                rows.append([])
                row_w = ww
            else:
                row_w = needed
            rows[-1].append((w, ww))
        panel_h = len(rows) * bh + (len(rows) - 1) * gap + 2 * pad
        self.rect = pygame.Rect(0, sh - panel_h, sw, panel_h)
        y = self.rect.top + pad
        for row in rows:
            total = sum(ww for _, ww in row) + gap * (len(row) - 1)
            x = (sw - total) // 2
            for w, ww in row:
                w.rect = pygame.Rect(x, y, ww, bh)
                x += ww + gap
            y += bh + gap

    # ------------------------------------------------------------------
    @property
    def interacting(self) -> bool:
        return bool(self._captures)

    def owns(self, pid: Hashable) -> bool:
        return pid in self._captures

    def handle(self, ev: PointerEvent) -> bool:
        """Route a pointer event. Returns True if the toolbar consumed it."""
        if ev.kind == PointerKind.DOWN:
            if not self.visible or not self.rect.collidepoint(ev.pos):
                return False
            target = next((w for w in self.widgets if w.hit(ev.pos)), None)
            if target is not None and not target.pointer_down(ev):
                target = None
            self._captures[ev.pid] = target
            return True
        if ev.pid not in self._captures:
            return False
        target = self._captures[ev.pid]
        if ev.kind == PointerKind.MOVE:
            if target is not None:
                target.pointer_move(ev)
        elif ev.kind == PointerKind.UP:
            del self._captures[ev.pid]
            if target is not None:
                target.pointer_up(ev)
        return True

    def update(self, now: int) -> str | None:
        """Drive held widgets; long-press on a button returns its hint."""
        for w in self._captures.values():
            if w is not None:
                w.update(now)
        return _long_press_hint(self._captures.values(), now, self.long_press_ms)

    def cancel(self) -> None:
        for w in self._captures.values():
            if w is not None:
                w.pointer_cancel()
        self._captures.clear()

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        if not self.visible:
            return
        bg = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        bg.fill(theme.panel_bg)
        surface.blit(bg, self.rect.topleft)
        for w in self.widgets:
            w.draw(surface, theme)


_SCROLL = object()  # capture marker: pointer is scrolling the panel


class SidePanel:
    """A collapsible editing panel on the right side of the screen.

    Layout: a header (title + header buttons), a row of category tabs, then
    a vertically scrollable list of :class:`SliderRow` objects for the
    selected tab. Dragging anywhere in the list that isn't a control
    scrolls it. Like the toolbar, the panel consumes every pointer that lands
    on it.
    """

    def __init__(
        self,
        title: str,
        tabs: dict[str, list[SliderRow]],
        header_buttons: list[Button],
        long_press_ms: int = 600,
        handle_hint: str = "",
    ) -> None:
        self._visible = False
        self._theme: Theme | None = None
        self._screen = (0, 0)
        # Chevron tab on the screen edge that opens/closes the panel. It is
        # shown whenever the controls are (``handle_visible``), even while the
        # panel itself is closed.
        self.edge = EdgeHandle(self.toggle, lambda: self._visible, hint=handle_hint)
        self.handle_visible = True
        self.on_toggle = None  # optional callback after the panel opens/closes
        self.title = title
        self.rows_by_tab = tabs
        self.tab = next(iter(tabs))
        self.header_buttons = header_buttons
        self.tab_buttons = [
            Button(name, (lambda n=name: self.select_tab(n)), is_active=(lambda n=name: self.tab == n))
            for name in tabs
        ]
        self.long_press_ms = long_press_ms
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.content_rect = pygame.Rect(0, 0, 0, 0)
        self.scroll = 0
        self.content_h = 0
        self._title_pos = (0, 0)
        self._captures: dict[Hashable, object] = {}
        self._scroll_start: dict[Hashable, tuple[float, int, bool]] = {}

    # ------------------------------------------------------------------
    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = bool(value)
        self._place_handle()

    def toggle(self) -> None:
        self.cancel_panel_widgets()
        self.visible = not self._visible
        if self.on_toggle:
            self.on_toggle()

    def _place_handle(self) -> None:
        theme = self._theme
        if theme is None:
            return
        w, h = theme.px(52), theme.px(120)
        edge_x = self.rect.left if self._visible else self._screen[0]
        cy = max(h // 2, self.rect.height // 2)
        self.edge.rect = pygame.Rect(edge_x - w, cy - h // 2, w, h)
        self.edge.swipe_px = theme.px(EdgeHandle.SWIPE_PX)

    @property
    def rows(self) -> list[SliderRow]:
        return self.rows_by_tab[self.tab]

    def _content_widgets(self) -> list[Widget]:
        return [w for row in self.rows for w in row.children]

    def _fixed_widgets(self) -> list[Widget]:
        return self.header_buttons + self.tab_buttons

    def select_tab(self, name: str) -> None:
        self.tab = name
        self.scroll = 0
        self._layout_rows()

    def layout(self, screen_size: tuple[int, int], theme: Theme, bottom: int | None = None) -> None:
        self._theme = theme
        self._screen = screen_size
        sw, sh = screen_size
        bottom = sh if bottom is None else bottom
        width = int(min(sw, max(theme.px(380), min(theme.px(520), sw * 0.36))))
        self.rect = pygame.Rect(sw - width, 0, width, max(0, bottom))
        pad, gap, bh = theme.px(theme.PANEL_PAD), theme.px(theme.GAP), theme.px(theme.BUTTON_H)
        x0, inner_w = self.rect.left + pad, width - 2 * pad
        y = self.rect.top + pad

        # Header: title on the left, buttons right-aligned.
        x = self.rect.right - pad
        for b in reversed(self.header_buttons):
            bw = b.preferred_width(theme)
            x -= bw
            b.rect = pygame.Rect(x, y, bw, bh)
            x -= gap
        self._title_pos = (x0, y + bh // 2)
        y += bh + gap

        # Tabs share the width equally.
        n = len(self.tab_buttons)
        tw = (inner_w - gap * (n - 1)) // max(1, n)
        for i, b in enumerate(self.tab_buttons):
            b.rect = pygame.Rect(x0 + i * (tw + gap), y, tw, bh)
        y += bh + gap

        self.content_rect = pygame.Rect(self.rect.left, y, width, max(0, self.rect.bottom - y))
        self._layout_rows()
        self._place_handle()

    def _layout_rows(self) -> None:
        theme = self._theme
        if theme is None:
            return
        pad, row_gap = theme.px(theme.PANEL_PAD), theme.px(14)
        x0, inner_w = self.rect.left + pad, self.rect.width - 2 * pad
        # First pass to measure, second to apply scroll.
        total = 0
        for row in self.rows:
            total += row.layout(x0, 0, inner_w, theme) + row_gap
        self.content_h = total
        self.scroll = max(0, min(self.scroll, self.max_scroll))
        y = self.content_rect.top + theme.px(4) - self.scroll
        for row in self.rows:
            y += row.layout(x0, y, inner_w, theme) + row_gap

    @property
    def max_scroll(self) -> int:
        return max(0, self.content_h - self.content_rect.height + (self._theme.px(8) if self._theme else 0))

    def scroll_by(self, dy: float) -> None:
        new = int(max(0, min(self.max_scroll, self.scroll + dy)))
        if new != self.scroll:
            self.scroll = new
            self._layout_rows()

    # ------------------------------------------------------------------
    @property
    def interacting(self) -> bool:
        return bool(self._captures)

    def handle(self, ev: PointerEvent) -> bool:
        if ev.kind == PointerKind.DOWN:
            if self.handle_visible and self.edge.hit(ev.pos):
                self.edge.pointer_down(ev)
                self._captures[ev.pid] = self.edge
                return True
            if not self.visible or not self.rect.collidepoint(ev.pos):
                return False
            target: object = None
            for w in self._fixed_widgets():
                if w.hit(ev.pos):
                    target = w
                    break
            if target is None and self.content_rect.collidepoint(ev.pos):
                target = next((w for w in self._content_widgets() if w.hit(ev.pos)), _SCROLL)
            if isinstance(target, Widget) and not target.pointer_down(ev):
                target = None
            if target is _SCROLL:
                self._scroll_start[ev.pid] = (ev.pos[1], self.scroll, False)
            self._captures[ev.pid] = target
            return True
        if ev.pid not in self._captures:
            return False
        target = self._captures[ev.pid]
        if ev.kind == PointerKind.MOVE:
            if target is _SCROLL:
                y0, s0, _ = self._scroll_start[ev.pid]
                self.scroll = int(max(0, min(self.max_scroll, s0 - (ev.pos[1] - y0))))
                self._layout_rows()
            elif isinstance(target, Widget):
                target.pointer_move(ev)
        elif ev.kind == PointerKind.UP:
            del self._captures[ev.pid]
            self._scroll_start.pop(ev.pid, None)
            if isinstance(target, Widget):
                target.pointer_up(ev)
        return True

    def update(self, now: int) -> str | None:
        held = [w for w in self._captures.values() if isinstance(w, Widget)]
        for w in held:
            w.update(now)
        return _long_press_hint(held, now, self.long_press_ms)

    def cancel(self) -> None:
        for w in self._captures.values():
            if isinstance(w, Widget):
                w.pointer_cancel()
        self._captures.clear()
        self._scroll_start.clear()

    def cancel_panel_widgets(self) -> None:
        """Cancel captures on the panel's own controls, but not the edge tab."""
        for pid, w in list(self._captures.items()):
            if w is not self.edge:
                if isinstance(w, Widget):
                    w.pointer_cancel()
                del self._captures[pid]
                self._scroll_start.pop(pid, None)

    def draw(self, surface: pygame.Surface, theme: Theme) -> None:
        if self._visible:
            self._draw_panel(surface, theme)
        if self.handle_visible:
            self.edge.draw(surface, theme)

    def _draw_panel(self, surface: pygame.Surface, theme: Theme) -> None:
        bg = pygame.Surface(self.rect.size, pygame.SRCALPHA)
        bg.fill(theme.panel_bg)
        surface.blit(bg, self.rect.topleft)
        pygame.draw.line(surface, theme.button_border, self.rect.topleft, self.rect.bottomleft, max(1, theme.px(2)))
        title = theme.font(theme.TOAST_FONT).render(self.title, True, theme.header_fg)
        surface.blit(title, title.get_rect(midleft=self._title_pos))
        for w in self._fixed_widgets():
            w.draw(surface, theme)
        old_clip = surface.get_clip()
        surface.set_clip(self.content_rect)
        for row in self.rows:
            if row.rect.bottom >= self.content_rect.top and row.rect.top <= self.content_rect.bottom:
                row.draw(surface, theme)
        surface.set_clip(old_clip)
        if self.max_scroll > 0:  # scroll indicator
            track = self.content_rect.inflate(0, -theme.px(8))
            frac = self.content_rect.height / max(1, self.content_h)
            h = max(theme.px(24), int(track.height * frac))
            y = track.top + int((track.height - h) * (self.scroll / self.max_scroll))
            pygame.draw.rect(
                surface, theme.slider_track, (self.rect.right - theme.px(5), y, theme.px(3), h), border_radius=2
            )


class Toast:
    """A short message shown near the top of the screen."""

    def __init__(self) -> None:
        self.text = ""
        self.until = 0

    def show(self, text: str, now: int, duration_ms: int = 1800) -> None:
        self.text = text
        self.until = now + duration_ms

    def active(self, now: int) -> bool:
        return bool(self.text) and now < self.until

    def clear(self) -> None:
        self.text = ""
        self.until = 0

    def draw(self, surface: pygame.Surface, theme: Theme, now: int) -> None:
        if not self.active(now):
            return
        font = theme.font(theme.TOAST_FONT)
        lines = [font.render(line, True, theme.toast_fg) for line in self.text.split("\n")]
        pad = theme.px(14)
        w = max(s.get_width() for s in lines) + 2 * pad
        h = sum(s.get_height() for s in lines) + 2 * pad
        w = min(w, surface.get_width() - theme.px(16))
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(box, theme.toast_bg, box.get_rect(), border_radius=theme.px(theme.RADIUS))
        y = pad
        for s in lines:
            box.blit(s, ((w - s.get_width()) // 2, y))
            y += s.get_height()
        surface.blit(box, ((surface.get_width() - w) // 2, theme.px(24)))
