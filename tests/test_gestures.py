"""Gesture recognition driven entirely by synthetic pointer events."""

import math

import pytest

from minibacklot.events.gesture_recognizer import (
    GestureConfig,
    GestureRecognizer,
    GestureType as G,
)

A, B, C = "a", "b", "c"


def types(gestures):
    return [g.type for g in gestures]


@pytest.fixture
def rec():
    return GestureRecognizer(GestureConfig())


def run_until(rec, t):
    return rec.update(t)


# ----------------------------------------------------------------------
# Taps
# ----------------------------------------------------------------------
def test_single_tap_is_delayed_until_double_tap_window_passes(rec):
    assert rec.pointer_down(A, (100, 100), 0) == []
    assert rec.pointer_up(A, (102, 101), 100) == []
    assert rec.update(300) == []  # still inside double-tap window
    out = rec.update(500)
    assert types(out) == [G.TAP]
    assert out[0].pos == (100, 100)


def test_double_tap(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_up(A, (100, 100), 80)
    rec.pointer_down(A, (110, 105), 200)
    out = rec.pointer_up(A, (110, 105), 260)
    assert types(out) == [G.DOUBLE_TAP]
    assert rec.update(2000) == []  # no stray single tap


def test_two_slow_taps_are_two_taps(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_up(A, (100, 100), 80)
    assert types(rec.update(500)) == [G.TAP]
    rec.pointer_down(A, (100, 100), 1000)
    rec.pointer_up(A, (100, 100), 1080)
    assert types(rec.update(1500)) == [G.TAP]


def test_distant_second_tap_flushes_first(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_up(A, (100, 100), 80)
    out = rec.pointer_down(A, (600, 600), 150)
    assert types(out) == [G.TAP]


def test_slow_press_is_not_a_tap(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_up(A, (100, 100), 450)  # > tap_max_ms, < long_press
    assert rec.update(2000) == []


# ----------------------------------------------------------------------
# Long press
# ----------------------------------------------------------------------
def test_long_press(rec):
    rec.pointer_down(A, (50, 60), 0)
    assert rec.needs_timer
    assert rec.update(500) == []
    out = rec.update(650)
    assert types(out) == [G.LONG_PRESS]
    assert out[0].pos == (50, 60)
    assert rec.update(700) == []  # fires once
    assert rec.pointer_up(A, (50, 60), 900) == []
    assert rec.update(2000) == []  # and no tap afterwards


def test_movement_cancels_long_press(rec):
    rec.pointer_down(A, (50, 60), 0)
    rec.pointer_move(A, (90, 60), 100)
    assert G.LONG_PRESS not in types(rec.update(1000))


# ----------------------------------------------------------------------
# Pan
# ----------------------------------------------------------------------
def test_small_movement_within_slop_does_not_pan(rec):
    rec.pointer_down(A, (100, 100), 0)
    assert rec.pointer_move(A, (105, 104), 20) == []


def test_pan_follows_finger_exactly(rec):
    rec.pointer_down(A, (100, 100), 0)
    out = rec.pointer_move(A, (130, 100), 20)
    assert types(out) == [G.GESTURE_START, G.PAN]
    assert out[1].delta == (30, 0)  # includes the slop distance
    out = rec.pointer_move(A, (140, 110), 40)
    assert types(out) == [G.PAN]
    assert out[0].delta == (10, 10)
    out = rec.pointer_up(A, (140, 110), 60)
    assert types(out) == [G.GESTURE_END]
    assert rec.update(1000) == []  # a pan is never a tap


def test_single_finger_never_zooms_or_rotates(rec):
    rec.pointer_down(A, (100, 100), 0)
    out = []
    for i in range(1, 50):
        out += rec.pointer_move(A, (100 + 10 * math.cos(i), 100 + 200 * math.sin(i)), i * 10)
    assert not {G.PINCH, G.ROTATE} & set(types(out))


# ----------------------------------------------------------------------
# Two fingers
# ----------------------------------------------------------------------
def start_two(rec, a=(100, 100), b=(300, 100)):
    rec.pointer_down(A, a, 0)
    return rec.pointer_down(B, b, 10)


def test_second_finger_starts_gesture_and_cancels_tap(rec):
    out = start_two(rec)
    assert types(out) == [G.GESTURE_START]
    rec.pointer_up(B, (300, 100), 50)
    out = rec.pointer_up(A, (100, 100), 60)
    assert types(out) == [G.GESTURE_END]
    assert rec.update(1000) == []


def test_pinch_zooms_around_midpoint(rec):
    start_two(rec)
    # Spread symmetrically: midpoint stays at (200, 100).
    out = rec.pointer_move(B, (340, 100), 20)
    out += rec.pointer_move(A, (60, 100), 30)
    pinches = [g for g in out if g.type == G.PINCH]
    assert pinches
    total = math.prod(g.scale for g in pinches)
    # Distance went 200 -> 280 (1.4x); threshold (3%) is absorbed.
    assert total == pytest.approx(1.4 / 1.03, rel=1e-6)
    assert pinches[-1].pos == pytest.approx((200, 100))
    assert G.ROTATE not in types(out)


def test_pinch_below_threshold_is_ignored(rec):
    start_two(rec)
    out = rec.pointer_move(B, (304, 100), 20)  # 2% change
    assert G.PINCH not in types(out)


def test_rotation_has_threshold_and_is_continuous(rec):
    start_two(rec)
    # Rotate finger B around A by 5 degrees: below threshold.
    def at(deg):
        r = math.radians(deg)
        return (100 + 200 * math.cos(r), 100 + 200 * math.sin(r))

    out = rec.pointer_move(B, at(5), 20)
    assert G.ROTATE not in types(out)
    out = rec.pointer_move(B, at(12), 30)
    rot = [g for g in out if g.type == G.ROTATE]
    assert len(rot) == 1 and rot[0].angle == pytest.approx(12 - 8)  # clockwise, excess only
    out = rec.pointer_move(B, at(20), 40)
    rot = [g for g in out if g.type == G.ROTATE]
    assert rot[0].angle == pytest.approx(8)


def test_rotation_can_be_disabled():
    rec = GestureRecognizer(GestureConfig(rotation_enabled=False))
    start_two(rec)
    out = rec.pointer_move(B, (100, 300), 20)
    assert G.ROTATE not in types(out)


def test_two_finger_drag_pans_by_midpoint(rec):
    start_two(rec)
    out = rec.pointer_move(A, (120, 100), 20)
    pans = [g for g in out if g.type == G.PAN]
    assert pans[0].delta == pytest.approx((10, 0))


def test_lifting_one_finger_continues_pan_without_jump(rec):
    start_two(rec)
    rec.pointer_move(A, (80, 100), 20)
    assert rec.pointer_up(A, (80, 100), 30) == []  # gesture continues
    out = rec.pointer_move(B, (305, 102), 40)
    assert types(out) == [G.PAN]
    assert out[0].delta == (5, 2)  # relative to B's last position
    assert types(rec.pointer_up(B, (305, 102), 50)) == [G.GESTURE_END]


def test_pan_then_second_finger_does_not_restart(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_move(A, (150, 100), 10)
    out = rec.pointer_down(B, (300, 300), 20)
    assert G.GESTURE_START not in types(out)  # still the same manipulation


def test_three_fingers_use_first_two_then_promote_third(rec):
    start_two(rec)
    rec.pointer_down(C, (500, 500), 20)
    assert rec.pointer_move(C, (600, 600), 30) == []  # ignored
    rec.pointer_up(A, (100, 100), 40)  # pair becomes (B, C)
    out = rec.pointer_move(C, (610, 600), 50)
    assert G.PAN in types(out)


def test_unknown_pointer_ignored(rec):
    assert rec.pointer_move("ghost", (1, 1), 0) == []
    assert rec.pointer_up("ghost", (1, 1), 0) == []


# ----------------------------------------------------------------------
# Cancellation
# ----------------------------------------------------------------------
def test_cancel_during_pan_ends_gesture_and_resets(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_move(A, (200, 100), 10)
    assert types(rec.cancel()) == [G.GESTURE_END]
    assert not rec.active and not rec.manipulating
    # Stale events for the cancelled finger are ignored.
    assert rec.pointer_move(A, (250, 100), 20) == []
    assert rec.pointer_up(A, (250, 100), 30) == []


def test_cancel_drops_pending_tap(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_up(A, (100, 100), 50)
    assert rec.cancel() == []
    assert rec.update(1000) == []


def test_cancel_preserves_config():
    cfg = GestureConfig(long_press_ms=123)
    rec = GestureRecognizer(cfg)
    rec.cancel()
    assert rec.config.long_press_ms == 123


def test_pointer_cancel_suppresses_tap(rec):
    rec.pointer_down(A, (100, 100), 0)
    rec.pointer_cancel(A)
    assert rec.update(1000) == []
    assert not rec.active
