"""Touch widgets: slider, buttons, panel capture and scrolling."""

import pygame
import pytest

from minibacklot.events.input_manager import PointerEvent, PointerKind, PointerSource
from minibacklot.imaging.pipeline import ADJUSTMENT_SPECS, AdjustmentState
from minibacklot.ui.button import Button
from minibacklot.ui.panels import SidePanel, Toolbar
from minibacklot.ui.slider import Slider, SliderRow
from minibacklot.ui.theme import Theme


def ev(kind, pos, t=0, pid="f1"):
    return PointerEvent(kind, pid, pos, PointerSource.TOUCH, t)


class Box:
    def __init__(self, v=0.0):
        self.v = v
        self.begins = 0
        self.ends = 0


def make_slider(box, lo=-100, hi=100, default=0, step=1):
    s = Slider(
        lo,
        hi,
        default,
        step,
        get_value=lambda: box.v,
        set_value=lambda v: setattr(box, "v", v),
        on_begin=lambda: setattr(box, "begins", box.begins + 1),
        on_end=lambda: setattr(box, "ends", box.ends + 1),
    )
    s.rect = pygame.Rect(0, 0, 228, 56)  # track from x=14 to x=214 (200 px)
    s.thumb_radius = 14
    s.detent_px = 8
    return s


def test_slider_value_x_mapping():
    s = make_slider(Box())
    assert s.value_to_x(-100) == 14 and s.value_to_x(100) == 214 and s.value_to_x(0) == 114
    assert s.x_to_value(14) == -100 and s.x_to_value(214) == 100
    assert s.x_to_value(-500) == -100 and s.x_to_value(900) == 100


def test_slider_tap_on_track_jumps():
    box = Box()
    s = make_slider(box)
    s.pointer_down(ev(PointerKind.DOWN, (184, 28)))
    assert box.v == 70 and box.begins == 1
    s.pointer_up(ev(PointerKind.UP, (184, 28)))
    assert box.ends == 1 and not s.dragging


def test_slider_grabbing_thumb_is_relative():
    box = Box(50)
    s = make_slider(box)
    thumb = s.value_to_x(50)  # 164
    s.pointer_down(ev(PointerKind.DOWN, (thumb + 10, 28)))  # touch slightly right of thumb
    assert box.v == 50  # no jump
    s.pointer_move(ev(PointerKind.MOVE, (thumb + 30, 28)))
    assert box.v == 70  # moved by 20 px = 20 units


def test_slider_detent_snaps_to_default():
    box = Box(40)
    s = make_slider(box)
    s.pointer_down(ev(PointerKind.DOWN, (s.value_to_x(40), 28)))
    s.pointer_move(ev(PointerKind.MOVE, (s.value_to_x(0) + 5, 28)))
    assert box.v == 0


def test_slider_respects_step():
    box = Box(0.0)
    s = make_slider(box, lo=-3, hi=3, step=0.05)
    s.pointer_down(ev(PointerKind.DOWN, (60, 28)))
    assert abs(box.v / 0.05 - round(box.v / 0.05)) < 1e-9


def test_repeat_button_fires_on_press_and_while_held():
    hits = []
    b = Button("+", lambda: hits.append(1), repeat=True)
    b.rect = pygame.Rect(0, 0, 56, 56)
    b.pointer_down(ev(PointerKind.DOWN, (10, 10), t=0))
    assert len(hits) == 1
    assert not b.update(100)
    assert b.update(500) and len(hits) == 2
    assert b.update(580) and len(hits) == 3
    b.pointer_up(ev(PointerKind.UP, (10, 10), t=600))
    assert len(hits) == 3  # release doesn't fire again
    assert not b.update(2000)


def test_button_click_requires_release_inside():
    hits = []
    b = Button("x", lambda: hits.append(1))
    b.rect = pygame.Rect(0, 0, 60, 60)
    b.pointer_down(ev(PointerKind.DOWN, (30, 30)))
    b.pointer_move(ev(PointerKind.MOVE, (400, 30)))
    b.pointer_up(ev(PointerKind.UP, (400, 30)))
    assert hits == []
    b.pointer_down(ev(PointerKind.DOWN, (30, 30)))
    b.pointer_up(ev(PointerKind.UP, (32, 31)))
    assert hits == [1]


def test_disabled_button_does_not_fire():
    hits = []
    b = Button("x", lambda: hits.append(1), is_enabled=lambda: False)
    b.rect = pygame.Rect(0, 0, 60, 60)
    b.pointer_down(ev(PointerKind.DOWN, (30, 30)))
    b.pointer_up(ev(PointerKind.UP, (30, 30)))
    assert hits == []


# ----------------------------------------------------------------------
# Side panel
# ----------------------------------------------------------------------
@pytest.fixture
def panel_setup(pg_display):
    st = AdjustmentState()
    tabs = {}
    for spec in ADJUSTMENT_SPECS:
        slider = Slider(
            spec.minimum,
            spec.maximum,
            spec.default,
            spec.step,
            get_value=lambda n=spec.name: st.current.get(n),
            set_value=lambda v, n=spec.name: st.set_value(n, v),
            on_begin=st.push_history,
        )
        tabs.setdefault(spec.category, []).append(
            SliderRow(spec.label, spec.format, slider, lambda k, n=spec.name: st.step(n, k), lambda n=spec.name: st.reset(n))
        )
    closed = []
    panel = SidePanel("Adjust", tabs, [Button("Done", lambda: closed.append(1))])
    panel.visible = True
    theme = Theme(scale=1.0)
    panel.layout((1280, 600), theme, bottom=500)
    return panel, st, theme, closed


def test_panel_slider_drag_changes_state_and_is_captured(panel_setup):
    panel, st, theme, _ = panel_setup
    row = panel.rows[0]  # Exposure
    assert row.label == "Exposure"
    track = row.slider.rect
    p0 = (row.slider.value_to_x(0), track.centery)
    assert panel.handle(ev(PointerKind.DOWN, p0))
    # Drag far outside the panel: still captured by the slider.
    assert panel.handle(ev(PointerKind.MOVE, (2000, -50)))
    assert st.current.exposure == 3.0
    assert panel.handle(ev(PointerKind.UP, (2000, -50)))
    assert not panel.interacting
    assert st.undo() and st.current.exposure == 0


def test_panel_plus_minus_and_reset(panel_setup):
    panel, st, _, _ = panel_setup
    row = panel.rows[1]  # Contrast
    panel.handle(ev(PointerKind.DOWN, row.plus.rect.center))
    panel.handle(ev(PointerKind.UP, row.plus.rect.center))
    assert st.current.contrast == 1
    panel.handle(ev(PointerKind.DOWN, row.reset.rect.center, pid="f2"))
    panel.handle(ev(PointerKind.UP, row.reset.rect.center, pid="f2"))
    assert st.current.contrast == 0


def test_panel_tabs_switch_rows(panel_setup):
    panel, _, _, _ = panel_setup
    color_tab = next(b for b in panel.tab_buttons if b.label == "Color")
    panel.handle(ev(PointerKind.DOWN, color_tab.rect.center))
    panel.handle(ev(PointerKind.UP, color_tab.rect.center))
    assert panel.tab == "Color"
    assert [r.label for r in panel.rows] == ["Temperature", "Tint", "Saturation"]


def test_panel_scrolls_by_dragging_background(panel_setup):
    panel, st, theme, _ = panel_setup
    assert panel.max_scroll > 0  # 6 rows don't fit in 500 px
    row0_y = panel.rows[0].rect.top
    # Start on the label area of the first row (not a control).
    start = (panel.rows[0]._label_rect.left + 5, panel.rows[0]._label_rect.centery)
    panel.handle(ev(PointerKind.DOWN, start))
    panel.handle(ev(PointerKind.MOVE, (start[0], start[1] - 100)))
    assert panel.scroll == min(100, panel.max_scroll)
    assert panel.rows[0].rect.top == row0_y - panel.scroll
    panel.handle(ev(PointerKind.UP, (start[0], start[1] - 100)))
    assert st.current.is_identity()  # scrolling changed nothing


def test_panel_ignores_pointers_outside_and_when_hidden(panel_setup):
    panel, _, _, _ = panel_setup
    assert not panel.handle(ev(PointerKind.DOWN, (10, 10)))
    panel.visible = False
    assert not panel.handle(ev(PointerKind.DOWN, panel.rect.center))


def test_panel_consumes_background_taps(panel_setup):
    panel, _, _, _ = panel_setup
    title_area = (panel.rect.left + 20, panel.rect.top + 30)
    assert panel.handle(ev(PointerKind.DOWN, title_area))
    assert panel.handle(ev(PointerKind.UP, title_area))


def test_panel_draws(panel_setup, pg_display):
    panel, _, theme, _ = panel_setup
    panel.draw(pg_display, theme)  # smoke test: no exceptions


def test_toolbar_drives_repeat_buttons(pg_display):
    hits = []
    tb = Toolbar([Button("+", lambda: hits.append(1), repeat=True)])
    tb.layout((800, 600), Theme())
    b = tb.widgets[0]
    tb.handle(ev(PointerKind.DOWN, b.rect.center, t=0))
    tb.update(1000)
    assert len(hits) == 2
