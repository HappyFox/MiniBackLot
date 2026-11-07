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


# ----------------------------------------------------------------------
# Edge handle (chevron tab that opens/closes the panel)
# ----------------------------------------------------------------------
@pytest.fixture
def closed_panel(panel_setup):
    panel, st, theme, _ = panel_setup
    panel.visible = False
    toggles = []
    panel.on_toggle = lambda: toggles.append(panel.visible)
    return panel, toggles


def tap(panel, pos, pid="h", t=0):
    panel.handle(ev(PointerKind.DOWN, pos, t=t, pid=pid))
    panel.handle(ev(PointerKind.UP, pos, t=t + 50, pid=pid))


def test_handle_sits_on_screen_edge_when_closed_and_on_panel_when_open(closed_panel):
    panel, _ = closed_panel
    assert panel.edge.rect.right == 1280  # screen width in fixture
    assert panel.edge.rect.width >= 48  # touch target
    assert panel.edge.rect.bottom <= 500  # above the toolbar
    panel.visible = True
    assert panel.edge.rect.right == panel.rect.left


def test_tap_handle_toggles_panel(closed_panel):
    panel, toggles = closed_panel
    tap(panel, panel.edge.rect.center)
    assert panel.visible and toggles == [True]
    tap(panel, panel.edge.rect.center)  # handle moved with the panel
    assert not panel.visible and toggles == [True, False]


def test_swipe_left_opens_swipe_right_closes(closed_panel):
    panel, _ = closed_panel
    x, y = panel.edge.rect.center
    panel.handle(ev(PointerKind.DOWN, (x, y), pid="s"))
    panel.handle(ev(PointerKind.MOVE, (x - 200, y), pid="s"))
    panel.handle(ev(PointerKind.UP, (x - 200, y), pid="s"))
    assert panel.visible
    x, y = panel.edge.rect.center
    # Swiping left again while open does nothing.
    panel.handle(ev(PointerKind.DOWN, (x, y), pid="s"))
    panel.handle(ev(PointerKind.UP, (x - 200, y), pid="s"))
    assert panel.visible
    panel.handle(ev(PointerKind.DOWN, (x, y), pid="s"))
    panel.handle(ev(PointerKind.UP, (x + 200, y), pid="s"))
    assert not panel.visible


def test_long_press_on_handle_shows_hint_without_toggling(closed_panel):
    panel, toggles = closed_panel
    panel.edge.hint = "Adjustments"
    pos = panel.edge.rect.center
    panel.handle(ev(PointerKind.DOWN, pos, t=0, pid="l"))
    assert panel.update(1000) == "Adjustments"
    panel.handle(ev(PointerKind.UP, pos, t=1100, pid="l"))
    assert not panel.visible and toggles == []


def test_hidden_handle_is_not_hittable(closed_panel):
    panel, _ = closed_panel
    panel.handle_visible = False
    assert not panel.handle(ev(PointerKind.DOWN, panel.edge.rect.center))
    assert not panel.visible


def test_handle_drawn_even_when_panel_closed(closed_panel, pg_display):
    panel, _ = closed_panel
    surf = pygame.Surface((1280, 600))
    surf.fill((0, 0, 0))
    panel.draw(surf, Theme())
    assert surf.get_at(panel.edge.rect.center)[:3] != (0, 0, 0) or surf.get_at(
        (panel.edge.rect.left + 6, panel.edge.rect.centery)
    )[:3] != (0, 0, 0)
    assert surf.get_at((100, 100))[:3] == (0, 0, 0)  # panel body not drawn


# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
def tab_named(panel, name):
    return next(b for b in panel.tab_buttons if b.label == name)


def test_tabs_are_tab_widgets_sitting_on_the_page(panel_setup):
    from minibacklot.ui.button import Tab

    panel, _, _, _ = panel_setup
    assert all(isinstance(b, Tab) for b in panel.tab_buttons)
    assert all(b.rect.bottom == panel.content_rect.top for b in panel.tab_buttons)
    assert all(b.rect.height >= 48 for b in panel.tab_buttons)


def test_exactly_one_tab_selected(panel_setup):
    panel, _, _, _ = panel_setup
    assert [b.active for b in panel.tab_buttons] == [True, False]
    tap(panel, tab_named(panel, "Color").rect.center)
    assert [b.active for b in panel.tab_buttons] == [False, True]


def test_tapping_selected_tab_does_nothing(panel_setup):
    panel, _, _, _ = panel_setup
    panel.scroll_by(80)
    before = panel.scroll
    tap(panel, tab_named(panel, "Light").rect.center)
    assert panel.tab == "Light" and panel.scroll == before


def test_tab_change_cancelled_by_dragging_off(panel_setup):
    panel, _, _, _ = panel_setup
    c = tab_named(panel, "Color").rect.center
    panel.handle(ev(PointerKind.DOWN, c, pid="d"))
    panel.handle(ev(PointerKind.MOVE, (c[0], c[1] + 300), pid="d"))
    panel.handle(ev(PointerKind.UP, (c[0], c[1] + 300), pid="d"))
    assert panel.tab == "Light"


def test_each_tab_remembers_its_scroll(panel_setup):
    panel, _, _, _ = panel_setup
    panel.scroll_by(90)
    light_scroll = panel.scroll
    assert light_scroll > 0
    tap(panel, tab_named(panel, "Color").rect.center)
    assert panel.scroll == 0  # Color fits without scrolling
    tap(panel, tab_named(panel, "Light").rect.center)
    assert panel.scroll == light_scroll


def test_selected_tab_joins_page_unselected_is_recessed(panel_setup, pg_display):
    panel, _, theme, _ = panel_setup
    surf = pygame.Surface((1280, 600))
    panel.draw(surf, theme)
    light, color = tab_named(panel, "Light"), tab_named(panel, "Color")
    # Just above the tab bottom: selected tab is page-coloured (no edge), the
    # other shows the baseline/border.
    y = light.rect.bottom - 1
    assert surf.get_at((light.rect.centerx, y))[:3] == theme.page_bg
    assert surf.get_at((color.rect.centerx, y))[:3] == theme.button_border
    # Body colours: selected matches the page, unselected is darker.
    mid = light.rect.top + light.rect.height * 3 // 4
    assert surf.get_at((light.rect.left + 8, mid))[:3] == theme.page_bg
    assert surf.get_at((color.rect.left + 8, mid))[:3] == theme.tab_inactive_bg
