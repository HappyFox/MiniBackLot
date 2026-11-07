"""MiniBacklot application: window, event loop and wiring.

Threading model: all pygame display and event calls happen on the main
thread. Image decoding and pyramid building run on a worker thread
(:class:`AsyncImageLoader`), and lighting adjustments are applied on another
(:class:`ImageProcessor`). Both wake the main loop with custom events.

Display policy: the renderer always draws ``display_pyramid``. While new
adjustments or a newly loaded image are being processed, the previous
background stays on screen until the first processed level is ready.

The loop blocks in ``pygame.event.wait(timeout)``; the timeout is short only
while something time-dependent is pending (gesture timers, smooth re-render,
toasts, auto-hide), so an idle backdrop uses essentially no CPU.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import pygame

from .config.settings import Settings, save_settings
from .display.hq_render import HQRenderer
from .display.renderer import Renderer
from .display.transforms import fill_scale, fit_scale
from .display.view import DisplayMode, ViewController
from .events.gesture_recognizer import Gesture, GestureConfig, GestureRecognizer, GestureType
from .events.input_manager import (
    InputManager,
    PointerEvent,
    PointerKind,
    PointerSource,
    configure_sdl_hints,
)
from .imaging.loader import (
    SUPPORTED_EXTENSIONS,
    AsyncImageLoader,
    ImagePyramid,
    LoadedImage,
    ProcessedPyramid,
    list_images,
)
from .imaging.pipeline import ADJUSTMENT_SPECS, CATEGORIES, Adjustments, AdjustmentState
from .imaging.processor import ImageProcessor
from .ui.button import Button
from .ui.panels import SidePanel, Toast, Toolbar
from .ui.slider import Slider, SliderRow
from .ui.theme import Theme, auto_ui_scale

log = logging.getLogger(__name__)

# Delay after the last view change before rendering a smooth frame.
SMOOTH_RENDER_DELAY_MS = 120
FAST_FRAME_MS = 16  # ~60 FPS while interacting
IDLE_WAIT_MS = 1000
QUIT_CONFIRM_MS = 3000
PROCESSED_CACHE_SIZE = 2
UNDO_LIMIT = 200


class Application:
    def __init__(
        self,
        settings: Settings,
        settings_path: Path | None = None,
        start_path: Path | None = None,
        fullscreen: bool | None = None,
        ui_scale: float | None = None,
    ) -> None:
        self.settings = settings
        self.settings_path = settings_path
        self.start_path = start_path
        self.fullscreen = settings.start_fullscreen if fullscreen is None else fullscreen
        self.ui_scale_override = ui_scale if ui_scale else (settings.ui_scale or None)

        self.running = False
        self.images: list[Path] = []
        self.index = 0
        self.current: LoadedImage | None = None
        self.loading_path: Path | None = None
        # What the renderer draws: the source pyramid or a processed one.
        self.display_pyramid: ImagePyramid | ProcessedPyramid | None = None
        # Image waiting for its first processed level before being shown.
        self.pending_image: LoadedImage | None = None
        self.pending_processed: ProcessedPyramid | None = None
        self.pending_gen = -1
        self.processed_cache: OrderedDict[tuple, ProcessedPyramid] = OrderedDict()
        self.seen_adj_revision = 0
        self.undo_order: list[str] = []  # "view" / "adjust", most recent last
        # High-quality settle frame (see display/hq_render.py).
        self.hq_frame: pygame.Surface | None = None
        self.hq_key: object = None
        self.hq_requested: object = None

        self.controls_visible = True
        self.last_ui_t = 0
        self.last_mouse_t = -10**9
        self.cursor_visible = True
        self.quit_armed_until = 0
        self.last_lock_toast_t = -10**9
        self.last_wheel_t = -10**9
        self.last_view_change_t = 0
        self.smooth_rendered = False
        self.seen_revision = -1
        self.dirty = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _init_pygame(self) -> None:
        configure_sdl_hints()
        pygame.init()
        pygame.display.set_caption("MiniBacklot")
        try:
            pygame.display.set_allow_screensaver(False)
        except (AttributeError, pygame.error):
            log.info("Cannot inhibit screensaver on this platform")
        self.load_event = pygame.event.custom_type()
        self.process_event = pygame.event.custom_type()
        self.hq_event = pygame.event.custom_type()
        self._create_window()

    def _create_window(self) -> None:
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.settings.window_size, pygame.RESIZABLE)

    def _make_theme(self, size: tuple[int, int]) -> Theme:
        return Theme(scale=self.ui_scale_override or auto_ui_scale(size))

    def _gesture_config(self) -> GestureConfig:
        ts, s = self.settings.touch, self.theme.scale
        return GestureConfig(
            tap_slop=ts.tap_slop * s,
            tap_max_ms=ts.tap_max_ms,
            double_tap_ms=ts.double_tap_ms,
            double_tap_slop=ts.double_tap_slop * s,
            long_press_ms=ts.long_press_ms,
            pinch_threshold=ts.pinch_threshold,
            rotation_threshold_deg=ts.rotation_threshold_deg,
            rotation_enabled=ts.rotation_enabled,
        )

    def _build_toolbar(self) -> Toolbar:
        v = self.view
        has_image = lambda: self.current is not None  # noqa: E731
        editable = lambda: self.view.can_edit  # noqa: E731
        many = lambda: len(self.images) > 1  # noqa: E731
        quit_armed = lambda: pygame.time.get_ticks() < self.quit_armed_until  # noqa: E731
        buttons = [
            Button("< Prev", lambda: self.navigate(-1), hint="Previous image", is_enabled=many),
            Button("Next >", lambda: self.navigate(1), hint="Next image", is_enabled=many),
            Button(
                "Fit",
                lambda: self._view_action(lambda: v.set_mode(DisplayMode.FIT)),
                hint="Show the whole image",
                is_active=lambda: has_image() and v.mode == DisplayMode.FIT,
                is_enabled=editable,
            ),
            Button(
                "Fill",
                lambda: self._view_action(lambda: v.set_mode(DisplayMode.FILL)),
                hint="Fill the screen (crops edges)",
                is_active=lambda: has_image() and v.mode == DisplayMode.FILL,
                is_enabled=editable,
            ),
            Button(
                "1:1",
                lambda: self._view_action(lambda: v.set_mode(DisplayMode.ACTUAL)),
                hint="Actual size: one image pixel per screen pixel",
                is_active=lambda: has_image() and v.mode == DisplayMode.ACTUAL,
                is_enabled=editable,
            ),
            Button("Flip H", lambda: self._view_action(v.flip_horizontal), hint="Mirror left/right", is_enabled=editable),
            Button("Flip V", lambda: self._view_action(v.flip_vertical), hint="Mirror top/bottom", is_enabled=editable),
            Button(
                "0°",
                lambda: self._view_action(v.straighten),
                hint="Remove rotation",
                is_enabled=lambda: editable() and v.transform.rotation != 0,
            ),
            Button(
                "Adjust",
                self.toggle_panel,
                hint="Lighting and colour adjustments",
                is_active=lambda: self.panel.visible,
                width_units=1.2,
            ),
            Button("Undo", self.undo, hint="Undo the last change", is_enabled=self.can_undo),
            Button("Reset", lambda: self._view_action(v.reset_all), hint="Fit, no rotation, no flips", is_enabled=editable),
            Button(
                lambda: "Locked" if v.locked else "Lock",
                self.toggle_lock,
                hint="Lock image position: gestures can't move the image",
                is_active=lambda: v.locked,
                style="lock",
                width_units=1.4,
            ),
            Button(
                lambda: "Window" if self.fullscreen else "Full",
                self.toggle_fullscreen,
                hint="Toggle fullscreen",
            ),
            Button(
                "Hide",
                self.hide_controls,
                hint="Hide controls for photography (tap anywhere to show)",
                is_active=lambda: True,
                width_units=1.4,
            ),
            Button(
                lambda: "Quit?" if quit_armed() else "Quit",
                self.request_quit,
                hint="Exit MiniBacklot (tap twice)",
                style="danger",
                is_active=quit_armed,
            ),
        ]
        return Toolbar(buttons, long_press_ms=self.settings.touch.long_press_ms)

    def _build_panel(self) -> SidePanel:
        st = self.adjust
        tabs: dict[str, list[SliderRow]] = {c: [] for c in CATEGORIES}
        for spec in ADJUSTMENT_SPECS:
            name = spec.name
            slider = Slider(
                spec.minimum,
                spec.maximum,
                spec.default,
                spec.step,
                get_value=lambda n=name: st.current.get(n),
                set_value=lambda v, n=name: st.set_value(n, v),
                on_begin=st.push_history,
            )
            tabs[spec.category].append(
                SliderRow(
                    spec.label,
                    spec.format,
                    slider,
                    on_step=lambda k, n=name: st.step(n, k),
                    on_reset=lambda n=name: st.reset(n),
                )
            )
        header = [
            Button(
                "Reset all",
                st.reset_all,
                hint="Reset every lighting adjustment",
                is_enabled=lambda: not st.current.is_identity(),
            ),
            Button("Done", self.toggle_panel, hint="Close the adjustment panel", is_active=lambda: True),
        ]
        return SidePanel("Adjust", tabs, header, long_press_ms=self.settings.touch.long_press_ms)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self) -> int:
        self._init_pygame()
        size = self.screen.get_size()
        self.theme = self._make_theme(size)
        self.view = ViewController(size, DisplayMode(self.settings.default_display_mode))
        self.view.set_locked(self.settings.position_locked)
        self.input = InputManager(size, self.settings.touch.mouse_suppress_ms)
        self.gestures = GestureRecognizer(self._gesture_config())
        self.renderer = Renderer(self.settings.background_color)
        self.loader = AsyncImageLoader(
            background=self.settings.background_color, wake_event_type=self.load_event
        )
        self.processor = ImageProcessor(wake_event_type=self.process_event)
        self.hq = HQRenderer(wake_event_type=self.hq_event)
        self.adjust = AdjustmentState(Adjustments.from_dict(self.settings.adjustments))
        self.adjust.on_history_push = lambda: self._record_undo("adjust")
        self.seen_adj_revision = self.adjust.revision
        self.view.on_history_push = lambda: self._record_undo("view")
        self.toolbar = self._build_toolbar()
        self.panel = self._build_panel()
        self.toolbar.layout(size, self.theme)
        self.panel.layout(size, self.theme, self.toolbar.rect.top)
        self.toast = Toast()
        now = pygame.time.get_ticks()
        self.last_ui_t = now

        start = self.start_path
        if start is None and self.settings.last_image and Path(self.settings.last_image).is_file():
            start = Path(self.settings.last_image)
        if start is None and self.settings.last_directory and Path(self.settings.last_directory).is_dir():
            start = Path(self.settings.last_directory)
        if start is not None:
            self.open_path(start)

        self.running = True
        try:
            while self.running:
                ev = pygame.event.wait(max(1, self._next_timeout()))
                events = pygame.event.get()
                if ev.type != pygame.NOEVENT:
                    events.insert(0, ev)
                now = pygame.time.get_ticks()
                for e in events:
                    self.handle_event(e, now)
                self.tick(now)
                if self.dirty:
                    self.draw(now)
        finally:
            self._save_settings()
            self.loader.shutdown()
            self.processor.shutdown()
            self.hq.shutdown()
            pygame.quit()
        return 0

    def _next_timeout(self) -> int:
        now = pygame.time.get_ticks()
        if (
            self.gestures.active
            or self.gestures.needs_timer
            or self.toolbar.interacting
            or self.panel.interacting
        ):
            return FAST_FRAME_MS
        if self.loading_path is not None or self.processor.busy:
            return 50  # keep the loading/processing indicator responsive
        deadlines = []
        if not self.smooth_rendered:
            deadlines.append(self.last_view_change_t + SMOOTH_RENDER_DELAY_MS)
        if self.toast.active(now):
            deadlines.append(self.toast.until)
        if self.controls_visible and self.current is not None and not self.panel.visible:
            deadlines.append(self.last_ui_t + int(self.settings.controls_autohide_s * 1000))
        if self.cursor_visible and not self.controls_visible:
            deadlines.append(self.last_mouse_t + int(self.settings.cursor_autohide_s * 1000))
        if self.quit_armed_until > now:
            deadlines.append(self.quit_armed_until)
        if not deadlines:
            return IDLE_WAIT_MS
        return max(1, min(IDLE_WAIT_MS, min(deadlines) - now + 1))

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def handle_event(self, e: pygame.event.Event, now: int) -> None:
        et = e.type
        if et == pygame.QUIT:
            self.running = False
        elif et in (pygame.WINDOWSIZECHANGED, pygame.VIDEORESIZE):
            self.on_resize()
        elif et in (pygame.WINDOWFOCUSLOST, pygame.WINDOWMINIMIZED):
            self.cancel_interaction()
        elif et in (pygame.WINDOWEXPOSED, pygame.WINDOWRESTORED, pygame.WINDOWSHOWN):
            self.dirty = True
        elif et == pygame.KEYDOWN:
            self.on_key(e, now)
        elif et == pygame.MOUSEWHEEL:
            self.on_wheel(e, now)
        elif et in (self.load_event, self.process_event, self.hq_event):
            pass  # results are collected in tick()
        else:
            for pe in self.input.translate(e, now):
                self.on_pointer(pe, now)

    def on_pointer(self, pe: PointerEvent, now: int) -> None:
        if pe.kind == PointerKind.HOVER:
            self._mouse_activity(now)
            return
        if pe.source == PointerSource.TOUCH:
            self._set_cursor(False)
        else:
            self._mouse_activity(now)

        if self.panel.handle(pe) or (self.toolbar.visible and self.toolbar.handle(pe)):
            self.last_ui_t = now
            self.dirty = True
            return

        g = self.gestures
        if pe.kind == PointerKind.DOWN:
            gestures = g.pointer_down(pe.pid, pe.pos, pe.t)
        elif pe.kind == PointerKind.MOVE:
            gestures = g.pointer_move(pe.pid, pe.pos, pe.t)
        else:
            gestures = g.pointer_up(pe.pid, pe.pos, pe.t)
        self.apply_gestures(gestures, now)

    def apply_gestures(self, gestures: list[Gesture], now: int) -> None:
        v = self.view
        for gs in gestures:
            t = gs.type
            if t == GestureType.GESTURE_START:
                v.end_rotation()
                if v.can_edit:
                    v.push_history()
                elif v.locked:
                    self._lock_toast(now)
            elif t == GestureType.GESTURE_END:
                v.end_rotation()
                self.smooth_rendered = False
                self.dirty = True
            elif t == GestureType.PAN:
                v.pan(*gs.delta)
            elif t == GestureType.PINCH:
                v.zoom(gs.pos, gs.scale)
            elif t == GestureType.ROTATE:
                v.rotate(gs.pos, gs.angle, snap_deg=self.settings.touch.rotation_snap_deg)
            elif t == GestureType.TAP:
                if self.controls_visible:
                    self.hide_controls()
                else:
                    self.show_controls(now)
            elif t == GestureType.DOUBLE_TAP:
                if v.locked:
                    self._lock_toast(now)
                else:
                    v.reset_zoom()
            elif t == GestureType.LONG_PRESS:
                self.show_controls(now)
                self.show_info(now)
            if self.controls_visible:
                self.last_ui_t = now

    def on_wheel(self, e: pygame.event.Event, now: int) -> None:
        self._mouse_activity(now)
        pos = pygame.mouse.get_pos()
        if self.panel.visible and self.panel.rect.collidepoint(pos):
            self.panel.scroll_by(-e.y * self.theme.px(48))
            self.last_ui_t = now
            self.dirty = True
            return
        if self.toolbar.visible and self.toolbar.rect.collidepoint(pos):
            return
        if not self.view.can_edit:
            if self.view.locked:
                self._lock_toast(now)
            return
        if now - self.last_wheel_t > 500:  # new wheel burst
            self.view.end_rotation()
            self.view.push_history()
        self.last_wheel_t = now
        mods = pygame.key.get_mods()
        amount = e.precise_y if hasattr(e, "precise_y") else e.y
        if mods & (pygame.KMOD_CTRL | pygame.KMOD_SHIFT):
            self.view.rotate(pos, -2.0 * amount, snap_deg=self.settings.touch.rotation_snap_deg)
        else:
            self.view.zoom(pos, self.settings.touch.wheel_zoom_step**amount)

    def on_key(self, e: pygame.event.Event, now: int) -> None:
        k, mods = e.key, e.mod
        v = self.view
        center = (self.screen.get_width() / 2, self.screen.get_height() / 2)
        ctrl = mods & pygame.KMOD_CTRL
        shift = mods & pygame.KMOD_SHIFT
        if k == pygame.K_ESCAPE:
            if self.controls_visible:
                self.hide_controls()
            elif self.fullscreen:
                self.toggle_fullscreen()
        elif k == pygame.K_q or (ctrl and k == pygame.K_w):
            self.running = False
        elif k in (pygame.K_F11, pygame.K_f):
            self.toggle_fullscreen()
        elif k in (pygame.K_TAB, pygame.K_c):
            self.hide_controls() if self.controls_visible else self.show_controls(now)
        elif k == pygame.K_i:
            self.show_info(now)
        elif k == pygame.K_l:
            self.toggle_lock()
        elif (ctrl and k == pygame.K_z) or k == pygame.K_u:
            self.undo()
        elif k == pygame.K_a:
            self.toggle_panel()
        elif shift and k in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
            step = 0.05 * min(self.screen.get_size())
            dx = {pygame.K_LEFT: -step, pygame.K_RIGHT: step}.get(k, 0.0)
            dy = {pygame.K_UP: -step, pygame.K_DOWN: step}.get(k, 0.0)
            self._view_action(lambda: v.pan(dx, dy), push=True)
        elif k in (pygame.K_RIGHT, pygame.K_PAGEDOWN, pygame.K_SPACE, pygame.K_n):
            self.navigate(1)
        elif k in (pygame.K_LEFT, pygame.K_PAGEUP, pygame.K_p):
            self.navigate(-1)
        elif k == pygame.K_HOME and self.images:
            self.show_index(0)
        elif k == pygame.K_END and self.images:
            self.show_index(len(self.images) - 1)
        elif k in (pygame.K_0, pygame.K_KP0):
            self._view_action(lambda: v.set_mode(DisplayMode.FIT))
        elif k in (pygame.K_1, pygame.K_KP1):
            self._view_action(lambda: v.set_mode(DisplayMode.ACTUAL))
        elif k in (pygame.K_2, pygame.K_KP2):
            self._view_action(lambda: v.set_mode(DisplayMode.FILL))
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._view_action(lambda: v.zoom(center, 1.25), push=True)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._view_action(lambda: v.zoom(center, 0.8), push=True)
        elif k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            deg = 15.0 if shift else 1.0
            deg = -deg if k == pygame.K_LEFTBRACKET else deg
            self._view_action(lambda: v.rotate(center, deg), push=True)
        elif k == pygame.K_r:
            self._view_action(v.straighten)
        elif k == pygame.K_BACKSPACE:
            self._view_action(v.reset_all)
        elif k == pygame.K_h:
            self._view_action(v.flip_horizontal)
        elif k == pygame.K_v:
            self._view_action(v.flip_vertical)
        self.dirty = True

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _view_action(self, fn, push: bool = False) -> None:
        now = pygame.time.get_ticks()
        if self.view.locked:
            self._lock_toast(now)
            return
        if push and self.view.can_edit:
            self.view.push_history()
        fn()
        self.dirty = True

    def _record_undo(self, kind: str) -> None:
        self.undo_order.append(kind)
        if len(self.undo_order) > UNDO_LIMIT:
            del self.undo_order[0]

    def can_undo(self) -> bool:
        return self.view.can_undo or self.adjust.can_undo

    def undo(self) -> None:
        """Undo the most recent view *or* adjustment change."""
        now = pygame.time.get_ticks()
        while self.undo_order:
            kind = self.undo_order.pop()
            if kind == "view":
                if self.view.locked and self.view.history_size:
                    self.undo_order.append(kind)
                    self._lock_toast(now)
                    return
                if self.view.undo():
                    break
            elif self.adjust.undo():
                break
        else:
            # Stack out of sync (e.g. history cleared on image change).
            if not self.view.undo():
                self.adjust.undo()
        self.dirty = True

    def toggle_panel(self) -> None:
        self.panel.cancel()
        self.panel.visible = not self.panel.visible
        if self.panel.visible:
            self.show_controls(pygame.time.get_ticks())
            self.panel.layout(self.screen.get_size(), self.theme, self.toolbar.rect.top)
        self.dirty = True

    def _lock_toast(self, now: int) -> None:
        if now - self.last_lock_toast_t > 1500:
            self.last_lock_toast_t = now
            self.toast.show("Image position is locked", now)
            self.dirty = True

    def toggle_lock(self) -> None:
        locked = self.view.toggle_lock()
        self.cancel_gestures()
        self.toast.show("Image position locked" if locked else "Image position unlocked", pygame.time.get_ticks())
        self.dirty = True

    def toggle_fullscreen(self) -> None:
        self.cancel_interaction()
        self.fullscreen = not self.fullscreen
        self._create_window()
        self.on_resize()

    def request_quit(self) -> None:
        now = pygame.time.get_ticks()
        if now < self.quit_armed_until:
            self.running = False
        else:
            self.quit_armed_until = now + QUIT_CONFIRM_MS
            self.toast.show("Tap Quit again to exit", now, QUIT_CONFIRM_MS)

    def show_controls(self, now: int) -> None:
        self.controls_visible = True
        self.toolbar.visible = True
        self.last_ui_t = now
        self.dirty = True

    def hide_controls(self) -> None:
        self.toolbar.cancel()
        self.panel.cancel()
        self.panel.visible = False
        self.controls_visible = False
        self.toolbar.visible = False
        self.toast.clear()
        self.quit_armed_until = 0
        self.dirty = True

    def show_info(self, now: int) -> None:
        if self.current is None:
            self.toast.show("No image loaded", now)
            return
        t = self.view.transform
        w, h = self.current.size
        flips = " ".join(n for n, on in (("flip-H", t.flip_h), ("flip-V", t.flip_v)) if on)
        line2 = f"{w}×{h} ({w * h / 1e6:.1f} MP)   zoom {t.scale * 100:.0f}%   rotation {t.rotation:.1f}°"
        if flips:
            line2 += f"   {flips}"
        if self.view.locked:
            line2 += "   LOCKED"
        pos = f"{self.index + 1}/{len(self.images)}  " if self.images else ""
        self.toast.show(f"{pos}{self.current.path.name}\n{line2}", now, 3500)
        self.dirty = True

    def cancel_gestures(self) -> None:
        if self.gestures.cancel():
            self.smooth_rendered = False

    def cancel_interaction(self) -> None:
        self.cancel_gestures()
        self.toolbar.cancel()
        self.panel.cancel()
        self.input.reset()
        self.dirty = True

    def _mouse_activity(self, now: int) -> None:
        self.last_mouse_t = now
        self._set_cursor(True)

    def _set_cursor(self, visible: bool) -> None:
        if visible != self.cursor_visible:
            self.cursor_visible = visible
            pygame.mouse.set_visible(visible)

    def on_resize(self) -> None:
        self.screen = pygame.display.get_surface()
        size = self.screen.get_size()
        if not self.fullscreen:
            self.settings.window_size = size
        self.theme = self._make_theme(size)
        self.gestures.config = self._gesture_config()
        self.input.set_window_size(size)
        self.view.set_screen_size(size)
        self.toolbar.layout(size, self.theme)
        self.panel.layout(size, self.theme, self.toolbar.rect.top)
        self.renderer.invalidate()
        self.dirty = True

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def open_path(self, path: Path) -> None:
        now = pygame.time.get_ticks()
        path = path.expanduser()
        try:
            path = path.resolve()
        except OSError:
            pass
        if path.is_dir():
            directory, target = path, None
        elif path.is_file():
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                self.toast.show(f"Unsupported file type: {path.name}", now, 3000)
                return
            directory, target = path.parent, path
        else:
            self.toast.show(f"Not found: {path}", now, 3000)
            return
        self.images = list_images(directory)
        self.settings.last_directory = str(directory)
        if not self.images:
            self.toast.show(f"No supported images in {directory}", now, 3000)
            return
        if target is None and self.settings.last_image:
            last = Path(self.settings.last_image)
            if last.parent == directory and last in self.images:
                target = last
        self.show_index(self.images.index(target) if target in self.images else 0)

    def navigate(self, delta: int) -> None:
        if self.images:
            self.show_index((self.index + delta) % len(self.images), direction=delta)

    def show_index(self, i: int, direction: int = 1) -> None:
        self.index = i
        path = self.images[i]
        self.nav_direction = 1 if direction >= 0 else -1
        img = self.loader.request(path)
        if img is not None:
            self.loading_path = None
            self._request_display(img)
        else:
            self.loading_path = path
        self.dirty = True

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def _expected_scale(self, img: LoadedImage) -> float:
        """Scale ``img`` will be shown at once it becomes current."""
        v = self.view
        if img is self.current or (v.locked and v.mode == DisplayMode.FREE):
            return v.transform.scale
        mode = v.mode if v.mode != DisplayMode.FREE else v.default_mode
        if mode == DisplayMode.ACTUAL:
            return 1.0
        f = fill_scale if mode == DisplayMode.FILL else fit_scale
        return f(img.size, v.screen_size, v.transform.rotation)

    def _request_display(self, img: LoadedImage) -> None:
        """Show ``img`` with the current adjustments, processing if needed."""
        adj = self.adjust.current
        key = (img.path, adj)
        ready: ImagePyramid | ProcessedPyramid | None = None
        if adj.is_identity():
            ready = img.pyramid
        elif key in self.processed_cache:
            self.processed_cache.move_to_end(key)
            ready = self.processed_cache[key]
        if ready is not None:
            self.processor.cancel()
            self.pending_image = None
            self.pending_processed = None
            self._activate(img, ready)
            return
        # Keep showing the previous background until the first level is ready.
        self.pending_image = img if img is not self.current else None
        self.pending_processed = ProcessedPyramid(img.pyramid, key)
        first = img.pyramid.index_for_scale(self._expected_scale(img))
        self.pending_gen = self.processor.submit(key, img.pyramid, adj, first_level=first)
        self.dirty = True

    def _poll_processor(self) -> None:
        for res in self.processor.poll():
            pp = self.pending_processed
            if pp is None or res.generation != self.pending_gen:
                continue
            pp.add(res.level, res.array, res.surface)
            if self.pending_image is not None:
                img, self.pending_image = self.pending_image, None
                self._activate(img, pp)
            else:
                self.display_pyramid = pp
            if pp.complete:
                self.processed_cache[pp.key] = pp
                self.processed_cache.move_to_end(pp.key)
                while len(self.processed_cache) > PROCESSED_CACHE_SIZE:
                    self.processed_cache.popitem(last=False)
                self.pending_processed = None
            self.dirty = True

    def _activate(self, img: LoadedImage, pyramid: ImagePyramid | ProcessedPyramid) -> None:
        self.display_pyramid = pyramid
        self.dirty = True
        if img is self.current:
            return
        self.current = img
        self.view.set_image(img.size)
        self.renderer.invalidate()
        self.smooth_rendered = False
        self.settings.last_image = str(img.path)
        self.dirty = True
        # Prefetch the next image in the direction of travel.
        if len(self.images) > 1:
            nxt = self.images[(self.index + getattr(self, "nav_direction", 1)) % len(self.images)]
            self.loader.request(nxt, prefetch=True)

    def _poll_loader(self, now: int) -> None:
        for res in self.loader.poll():
            if res.path != self.loading_path:
                continue  # user has moved on; result stays cached
            self.loading_path = None
            if res.image is not None:
                self._request_display(res.image)
            else:
                log.warning("Cannot open %s: %s", res.path, res.error)
                self.toast.show(f"Cannot open {res.path.name}\n{res.error}", now, 4000)
                self.dirty = True

    # ------------------------------------------------------------------
    # Per-iteration update and drawing
    # ------------------------------------------------------------------
    def tick(self, now: int) -> None:
        self._poll_loader(now)
        if self.adjust.revision != self.seen_adj_revision:
            self.seen_adj_revision = self.adjust.revision
            target = self.pending_image or self.current
            if target is not None:
                self._request_display(target)
            self.dirty = True
        self._poll_processor()
        for res in self.hq.poll():
            self.hq_frame, self.hq_key = res.surface, res.key
            self.dirty = True
        if self.gestures.needs_timer:
            self.apply_gestures(self.gestures.update(now), now)
        hint = None
        if self.toolbar.interacting:
            hint = self.toolbar.update(now)
            self.dirty = True
        if self.panel.interacting:
            hint = self.panel.update(now) or hint
            self.dirty = True
        if hint:
            self.toast.show(hint, now, 2500)
            self.dirty = True

        if self.view.revision != self.seen_revision:
            self.seen_revision = self.view.revision
            self.last_view_change_t = now
            self.smooth_rendered = False
            self.dirty = True
        elif not self.smooth_rendered and not self.gestures.manipulating:
            if now - self.last_view_change_t >= SMOOTH_RENDER_DELAY_MS:
                self.dirty = True

        if (
            self.controls_visible
            and self.current is not None
            and not self.panel.visible
            and not self.toolbar.interacting
            and not self.gestures.active
            and now - self.last_ui_t > self.settings.controls_autohide_s * 1000
        ):
            self.hide_controls()
        if (
            self.cursor_visible
            and not self.controls_visible
            and now - self.last_mouse_t > self.settings.cursor_autohide_s * 1000
        ):
            self._set_cursor(False)
        if self.quit_armed_until and now >= self.quit_armed_until:
            self.quit_armed_until = 0
            self.dirty = True
        if self.toast.text and not self.toast.active(now):
            self.toast.clear()
            self.dirty = True

    def draw(self, now: int) -> None:
        size = self.screen.get_size()
        interactive = self.gestures.manipulating or (
            now - self.last_view_change_t < SMOOTH_RENDER_DELAY_MS
        )
        smooth = not interactive
        frame = self._frame(size, smooth)
        self.screen.blit(frame, (0, 0))
        if self.current is None:
            self._draw_placeholder()
        if self.controls_visible:
            self._draw_status()
            self.toolbar.draw(self.screen, self.theme)
            self.panel.draw(self.screen, self.theme)
        self.toast.draw(self.screen, self.theme, now)
        pygame.display.flip()
        self.smooth_rendered = smooth or self.current is None
        self.dirty = False

    def _frame(self, size: tuple[int, int], smooth: bool) -> pygame.Surface:
        """The image frame: HQ if ready, else the interactive renderer's.

        Once the view has settled, a high-quality re-render is requested from
        the worker; the instant smooth frame is shown until it arrives.
        """
        pyr = self.display_pyramid
        t = self.view.transform
        if not (smooth and pyr is not None and self.settings.high_quality_render):
            if self.hq_requested is not None:
                self.hq.cancel()
                self.hq_requested = None
            return self.renderer.frame(size, pyr, t, smooth)
        key = (size, id(pyr), getattr(pyr, "version", 0), t, self.renderer.background)
        if self.hq_frame is not None and self.hq_key == key:
            return self.hq_frame
        if self.hq_requested != key:
            self.hq.submit(key, pyr, t, size, self.renderer.background)
            self.hq_requested = key
        return self.renderer.frame(size, pyr, t, smooth)

    def _draw_placeholder(self) -> None:
        th = self.theme
        if self.loading_path is not None:
            lines = [f"Loading {self.loading_path.name}…"]
        elif self.pending_image is not None:
            lines = [f"Processing {self.pending_image.path.name}…"]
        else:
            exts = ", ".join(sorted({e.lstrip('.').upper() for e in SUPPORTED_EXTENSIONS} - {"JPEG", "TIF"}))
            lines = [
                "MiniBacklot",
                "Digital backdrops for miniature photography",
                "",
                "No images loaded. Start with:",
                "minibacklot /path/to/photos",
                f"Supported: {exts}",
            ]
        font = th.font(th.TOAST_FONT)
        surfs = [font.render(line, True, (200, 200, 205)) for line in lines]
        total = sum(s.get_height() for s in surfs)
        y = (self.screen.get_height() - total) // 2 - th.px(60)
        for s in surfs:
            self.screen.blit(s, s.get_rect(midtop=(self.screen.get_width() // 2, y)))
            y += s.get_height()

    def _draw_status(self) -> None:
        th = self.theme
        parts = []
        if self.images:
            parts.append(f"{self.index + 1}/{len(self.images)}")
        if self.current is not None:
            t = self.view.transform
            parts.append(self.current.path.name)
            parts.append(f"{t.scale * 100:.0f}%")
            if t.rotation:
                parts.append(f"{t.rotation:+.1f}°")
        if self.loading_path is not None:
            parts.append("Loading…")
        elif self.processor.busy:
            parts.append("Processing…")
        if not self.adjust.current.is_identity():
            parts.append("ADJUSTED")
        if self.view.locked:
            parts.append("LOCKED")
        if not parts:
            return
        text = th.font(th.SMALL_FONT).render("   ".join(parts), True, th.button_fg)
        pad = th.px(8)
        box = pygame.Surface((text.get_width() + 2 * pad, text.get_height() + 2 * pad), pygame.SRCALPHA)
        box.fill(th.panel_bg)
        box.blit(text, (pad, pad))
        self.screen.blit(box, (th.px(10), self.toolbar.rect.top - box.get_height() - th.px(8)))

    def _save_settings(self) -> None:
        self.settings = replace(
            self.settings,
            position_locked=self.view.locked,
            adjustments=self.adjust.current.to_dict(),
        )
        save_settings(self.settings, self.settings_path)
