import math

import pytest

from minibacklot.display.transforms import (
    ViewTransform,
    fill_scale,
    fit_scale,
    normalize_angle,
)

IMG = (400, 300)


def approx_vec(v, expected, tol=1e-6):
    return v[0] == pytest.approx(expected[0], abs=tol) and v[1] == pytest.approx(expected[1], abs=tol)


@pytest.mark.parametrize(
    "t",
    [
        ViewTransform(),
        ViewTransform(cx=100, cy=50, scale=2.5),
        ViewTransform(cx=-30, cy=900, scale=0.3, rotation=37),
        ViewTransform(cx=10, cy=20, scale=1.7, rotation=-123, flip_h=True),
        ViewTransform(cx=10, cy=20, scale=0.9, rotation=90, flip_h=True, flip_v=True),
    ],
)
def test_screen_image_roundtrip(t):
    for p in [(0, 0), (400, 300), (123.4, 56.7), (-50, 1000)]:
        q = t.image_to_screen(p, IMG)
        assert approx_vec(t.screen_to_image(q, IMG), p)


def test_image_center_maps_to_transform_center():
    t = ViewTransform(cx=321, cy=123, scale=3, rotation=45, flip_v=True)
    assert approx_vec(t.image_to_screen((200, 150), IMG), (321, 123))


def test_rotation_is_clockwise_on_screen():
    t = ViewTransform(rotation=90)
    # Image point to the right of center should end up below center (y-down).
    q = t.image_to_screen((300, 150), IMG)
    assert approx_vec(q, (0, 100))


def test_flip_h_mirrors_x():
    t = ViewTransform(flip_h=True)
    assert approx_vec(t.image_to_screen((300, 150), IMG), (-100, 0))


@pytest.mark.parametrize("factor", [0.25, 0.9, 1.5, 7.0])
@pytest.mark.parametrize("rotation", [0, 30, -170])
def test_zoom_keeps_anchor_fixed(factor, rotation):
    t = ViewTransform(cx=500, cy=400, scale=1.2, rotation=rotation, flip_h=True)
    anchor = (123.0, 456.0)
    before = t.screen_to_image(anchor, IMG)
    t2 = t.zoomed_about(anchor, factor)
    assert t2.scale == pytest.approx(1.2 * factor)
    assert approx_vec(t2.image_to_screen(before, IMG), anchor)


def test_zoom_clamped_still_keeps_anchor():
    t = ViewTransform(cx=500, cy=400, scale=1.0)
    anchor = (100.0, 100.0)
    before = t.screen_to_image(anchor, IMG)
    t2 = t.zoomed_about(anchor, 1000.0, max_scale=4.0)
    assert t2.scale == 4.0
    assert approx_vec(t2.image_to_screen(before, IMG), anchor)


def test_zoom_ignores_invalid_factor():
    t = ViewTransform(scale=2)
    assert t.zoomed_about((0, 0), 0) is t
    assert t.zoomed_about((0, 0), float("nan")) is t


@pytest.mark.parametrize("deg", [5, -45, 90, 179])
def test_rotate_keeps_anchor_fixed(deg):
    t = ViewTransform(cx=500, cy=400, scale=0.8, rotation=10)
    anchor = (700.0, 100.0)
    before = t.screen_to_image(anchor, IMG)
    t2 = t.rotated_about(anchor, deg)
    assert t2.rotation == pytest.approx(normalize_angle(10 + deg))
    assert approx_vec(t2.image_to_screen(before, IMG), anchor)


def test_composed_pinch_matches_closed_form():
    """Pan + zoom + rotate about the new midpoint == one similarity transform."""
    t = ViewTransform(cx=300, cy=200, scale=1.0, rotation=0)
    m0, m1, k, a = (100.0, 100.0), (140.0, 90.0), 1.6, 25.0
    p_img = t.screen_to_image(m0, IMG)
    t2 = t.translated(m1[0] - m0[0], m1[1] - m0[1]).zoomed_about(m1, k).rotated_about(m1, a)
    # The image point that was under the old midpoint is under the new one.
    assert approx_vec(t2.image_to_screen(p_img, IMG), m1)
    r = math.radians(a)
    vx, vy = t.cx - m0[0], t.cy - m0[1]
    expected = (m1[0] + k * (math.cos(r) * vx - math.sin(r) * vy), m1[1] + k * (math.sin(r) * vx + math.cos(r) * vy))
    assert approx_vec((t2.cx, t2.cy), expected)


def test_mirror_on_screen_when_rotated():
    t = ViewTransform(cx=0, cy=0, rotation=30)
    m = t.mirrored_horizontally()
    for p in [(0, 0), (400, 0), (123, 277)]:
        a = t.image_to_screen(p, IMG)
        b = m.image_to_screen(p, IMG)
        assert approx_vec(b, (-a[0], a[1]))
    v = t.mirrored_vertically()
    for p in [(0, 0), (400, 0), (123, 277)]:
        a = t.image_to_screen(p, IMG)
        b = v.image_to_screen(p, IMG)
        assert approx_vec(b, (a[0], -a[1]))


def test_double_mirror_is_identity():
    t = ViewTransform(cx=5, cy=6, scale=2, rotation=33)
    assert t.mirrored_horizontally().mirrored_horizontally().is_close(t)


def test_resize_preserves_relative_framing():
    t = ViewTransform(cx=1000, cy=500, scale=0.5)
    t2 = t.resized((1920, 1080), (960, 540))
    assert t2.scale == pytest.approx(0.25)
    assert (t2.cx, t2.cy) == pytest.approx((500, 250))
    # Round trip back.
    assert t2.resized((960, 540), (1920, 1080)).is_close(t)


def test_normalize_angle():
    assert normalize_angle(190) == pytest.approx(-170)
    assert normalize_angle(-180) == pytest.approx(180)
    assert normalize_angle(720) == pytest.approx(0)


def test_fit_and_fill_unrotated():
    assert fit_scale((4000, 3000), (1920, 1080)) == pytest.approx(0.36)
    assert fill_scale((4000, 3000), (1920, 1080)) == pytest.approx(0.48)


def test_fit_rotated_90_swaps_dimensions():
    assert fit_scale((4000, 3000), (1920, 1080), 90) == pytest.approx(min(1920 / 3000, 1080 / 4000))


def test_fill_rotated_covers_screen():
    img, scr, rot = (4000, 3000), (1920, 1080), 20.0
    s = fill_scale(img, scr, rot)
    t = ViewTransform(cx=960, cy=540, scale=s, rotation=rot)
    for corner in [(0, 0), (1920, 0), (0, 1080), (1920, 1080)]:
        x, y = t.screen_to_image(corner, img)
        assert -1e-6 <= x <= 4000 + 1e-6 and -1e-6 <= y <= 3000 + 1e-6
