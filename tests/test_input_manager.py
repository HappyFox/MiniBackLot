"""Touch/mouse normalization and duplicate-event suppression."""

import pygame

from minibacklot.events.input_manager import (
    MOUSE_POINTER_ID,
    InputManager,
    PointerKind,
    PointerSource,
)


def finger(kind, x, y, finger_id=0, touch_id=7):
    return pygame.event.Event(kind, x=x, y=y, dx=0.0, dy=0.0, finger_id=finger_id, touch_id=touch_id)


def mouse(kind, pos, button=1, touch=False):
    if kind == pygame.MOUSEMOTION:
        return pygame.event.Event(kind, pos=pos, rel=(0, 0), buttons=(0, 0, 0), touch=touch)
    return pygame.event.Event(kind, pos=pos, button=button, touch=touch)


def test_finger_coordinates_are_scaled_to_window():
    im = InputManager((1000, 500))
    (pe,) = im.translate(finger(pygame.FINGERDOWN, 0.5, 0.25, finger_id=3), 0)
    assert pe.kind == PointerKind.DOWN
    assert pe.pos == (500, 125)
    assert pe.source == PointerSource.TOUCH
    assert pe.pid == ("touch", 7, 3)


def test_simultaneous_fingers_have_distinct_ids():
    im = InputManager((100, 100))
    a = im.translate(finger(pygame.FINGERDOWN, 0.1, 0.1, finger_id=1), 0)[0]
    b = im.translate(finger(pygame.FINGERDOWN, 0.9, 0.9, finger_id=2), 0)[0]
    assert a.pid != b.pid


def test_touch_generated_mouse_events_are_dropped():
    im = InputManager((100, 100), mouse_suppress_ms=0)
    assert im.translate(mouse(pygame.MOUSEBUTTONDOWN, (10, 10), touch=True), 0) == []
    assert im.translate(mouse(pygame.MOUSEMOTION, (12, 10), touch=True), 0) == []
    assert im.translate(mouse(pygame.MOUSEBUTTONUP, (12, 10), touch=True), 0) == []


def test_mouse_events_shortly_after_touch_are_dropped_even_without_flag():
    im = InputManager((100, 100), mouse_suppress_ms=500)
    im.translate(finger(pygame.FINGERDOWN, 0.1, 0.1), 1000)
    assert im.translate(mouse(pygame.MOUSEBUTTONDOWN, (10, 10)), 1100) == []
    # Long after touch, real mouse works again.
    (pe,) = im.translate(mouse(pygame.MOUSEBUTTONDOWN, (10, 10)), 2000)
    assert pe.kind == PointerKind.DOWN and pe.pid == MOUSE_POINTER_ID


def test_mouse_synthesized_touch_is_dropped():
    im = InputManager((100, 100))
    assert im.translate(finger(pygame.FINGERDOWN, 0.5, 0.5, touch_id=-1), 0) == []


def test_mouse_drag_sequence():
    im = InputManager((100, 100))
    assert im.translate(mouse(pygame.MOUSEMOTION, (5, 5)), 0)[0].kind == PointerKind.HOVER
    assert im.translate(mouse(pygame.MOUSEBUTTONDOWN, (5, 5)), 1)[0].kind == PointerKind.DOWN
    assert im.translate(mouse(pygame.MOUSEMOTION, (9, 5)), 2)[0].kind == PointerKind.MOVE
    assert im.translate(mouse(pygame.MOUSEBUTTONUP, (9, 5)), 3)[0].kind == PointerKind.UP
    assert im.translate(mouse(pygame.MOUSEBUTTONUP, (9, 5)), 4) == []  # no double release


def test_mouse_release_delivered_even_if_touch_occurs_mid_drag():
    im = InputManager((100, 100), mouse_suppress_ms=500)
    im.translate(mouse(pygame.MOUSEBUTTONDOWN, (5, 5)), 0)
    im.translate(finger(pygame.FINGERDOWN, 0.1, 0.1), 10)
    (pe,) = im.translate(mouse(pygame.MOUSEBUTTONUP, (5, 5)), 20)
    assert pe.kind == PointerKind.UP


def test_non_left_buttons_ignored():
    im = InputManager((100, 100))
    assert im.translate(mouse(pygame.MOUSEBUTTONDOWN, (5, 5), button=3), 0) == []
