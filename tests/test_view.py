import pytest

from minibacklot.display.view import DisplayMode, ViewController

SCREEN = (1920, 1080)
IMG = (4000, 3000)


def make():
    v = ViewController(SCREEN)
    v.set_image(IMG)
    return v


def test_new_image_is_fitted_and_centered():
    v = make()
    assert v.mode == DisplayMode.FIT
    assert v.transform.scale == pytest.approx(0.36)
    assert (v.transform.cx, v.transform.cy) == (960, 540)


def test_pan_switches_to_free_mode():
    v = make()
    assert v.pan(10, 5)
    assert v.mode == DisplayMode.FREE
    assert (v.transform.cx, v.transform.cy) == (970, 545)


def test_lock_blocks_every_position_change():
    v = make()
    before = v.snapshot()
    v.set_locked(True)
    assert not v.pan(10, 0)
    assert not v.zoom((100, 100), 2.0)
    assert not v.rotate((100, 100), 30)
    assert not v.set_mode(DisplayMode.FILL)
    assert not v.flip_horizontal()
    assert not v.flip_vertical()
    assert not v.reset_all()
    assert not v.undo()
    assert v.snapshot() == before
    v.set_locked(False)
    assert v.pan(1, 0)


def test_locked_free_view_survives_image_change():
    v = make()
    v.pan(100, 0)
    v.set_locked(True)
    t = v.transform
    v.set_image((3000, 2000))
    assert v.transform == t


def test_unlocked_free_view_refits_on_image_change():
    v = make()
    v.pan(100, 0)
    v.set_image((3000, 2000))
    assert v.mode == DisplayMode.FIT
    assert v.transform.cx == 960


def test_undo_restores_previous_state():
    v = make()
    start = v.snapshot()
    v.push_history()
    v.pan(50, 50)
    v.zoom((0, 0), 2)
    assert v.undo()
    assert v.snapshot() == start
    assert not v.undo()


def test_undo_skips_duplicate_snapshots():
    v = make()
    v.push_history()  # e.g. a gesture that did nothing
    v.push_history()
    assert not v.undo()


def test_zoom_is_clamped():
    v = make()
    for _ in range(100):
        v.zoom((960, 540), 2.0)
    assert v.transform.scale <= 32.0 + 1e-9
    for _ in range(200):
        v.zoom((960, 540), 0.5)
    assert max(IMG) * v.transform.scale >= 32.0 - 1e-6


def test_actual_and_fill_modes():
    v = make()
    v.set_mode(DisplayMode.ACTUAL)
    assert v.transform.scale == 1.0
    v.set_mode(DisplayMode.FILL)
    assert v.transform.scale == pytest.approx(0.48)


def test_screen_resize_refits_in_fit_mode():
    v = make()
    v.set_screen_size((960, 540))
    assert v.transform.scale == pytest.approx(0.18)
    assert (v.transform.cx, v.transform.cy) == (480, 270)


def test_screen_resize_scales_free_view_and_history():
    v = make()
    v.push_history()
    v.pan(100, 0)
    v.set_screen_size((960, 540))
    assert v.transform.cx == pytest.approx(530)
    v.undo()
    assert v.transform.cx == pytest.approx(480)


def test_reset_all_clears_rotation_and_flips():
    v = make()
    v.rotate((0, 0), 30)
    v.flip_horizontal()
    v.reset_all()
    t = v.transform
    assert (t.rotation, t.flip_h, t.flip_v) == (0.0, False, False)
    assert v.mode == DisplayMode.FIT


def test_revision_changes_on_edit():
    v = make()
    r = v.revision
    v.pan(1, 1)
    assert v.revision > r


# ----------------------------------------------------------------------
# Sticky rotation snapping
# ----------------------------------------------------------------------
def rotate_in_steps(v, total, step=0.5, snap=4.0, anchor=(700.0, 300.0)):
    n = round(abs(total) / step)
    for _ in range(n):
        v.rotate(anchor, step if total > 0 else -step, snap_deg=snap)


def test_snap_holds_near_right_angle_then_releases():
    v = make()
    v.rotate((960, 540), 80)  # unsnapped setup
    v.end_rotation()
    rotate_in_steps(v, 7)  # raw 87 -> within 4 of 90
    assert v.transform.rotation == 90.0  # exact, not 89.999
    rotate_in_steps(v, 6)  # raw 93: still held
    assert v.transform.rotation == 90.0
    rotate_in_steps(v, 1.5)  # raw 94.5: released
    assert v.transform.rotation == pytest.approx(94.5)
    rotate_in_steps(v, -1.5)  # back to raw 93: grabbed again
    assert v.transform.rotation == 90.0


@pytest.mark.parametrize("target", [0.0, 180.0, -90.0])
def test_snaps_to_each_right_angle(target):
    v = make()
    start = target - 10
    v.rotate((960, 540), start)
    v.end_rotation()
    rotate_in_steps(v, 7)  # raw target - 3
    assert v.transform.rotation == pytest.approx(180.0 if target == 180 else target)


def test_snap_keeps_anchor_point_fixed():
    v = make()
    v.rotate((960, 540), 84)
    v.end_rotation()
    anchor = (700.0, 300.0)
    before = v.transform.screen_to_image(anchor, IMG)
    rotate_in_steps(v, 3, anchor=anchor)  # snaps to 90
    assert v.transform.rotation == 90.0
    after = v.transform.image_to_screen(before, IMG)
    assert after == pytest.approx(anchor)


def test_snap_disabled():
    v = make()
    v.rotate((960, 540), 88, snap_deg=0)
    assert v.transform.rotation == pytest.approx(88)


def test_small_keyboard_steps_never_get_stuck():
    v = make()
    for _ in range(3):
        v.rotate((960, 540), 1.0)  # keyboard: no snap
    assert v.transform.rotation == pytest.approx(3.0)


def test_snap_restarts_after_outside_change():
    v = make()
    rotate_in_steps(v, 2)  # raw 2 -> held at 0
    assert v.transform.rotation == 0.0
    v.straighten()  # no-op at 0, but simulate an outside change:
    v.rotate((960, 540), 50, snap_deg=0)  # resets snap tracking
    rotate_in_steps(v, 1)
    assert v.transform.rotation == pytest.approx(51)


def test_snap_rotation_function():
    from minibacklot.display.transforms import snap_rotation

    assert snap_rotation(88.0, 4) == 90.0
    assert snap_rotation(85.0, 4) == 85.0
    assert snap_rotation(-178.0, 4) == 180.0
    assert snap_rotation(179.0, 4) == 180.0
    assert snap_rotation(3.0, 0) == 3.0
