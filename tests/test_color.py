"""Colour mathematics."""

import numpy as np
import pytest

from minibacklot.imaging import color


def test_srgb_roundtrip_all_8bit_values():
    v = np.arange(256) / 255.0
    assert np.allclose(color.linear_to_srgb(color.srgb_to_linear(v)), v, atol=1e-12)


def test_srgb_known_values():
    assert color.srgb_to_linear(0.0) == 0.0
    assert color.srgb_to_linear(1.0) == pytest.approx(1.0)
    # sRGB 0.5 is ~21.4 % linear light; mid grey (18 %) encodes to ~0.4614.
    assert color.srgb_to_linear(0.5) == pytest.approx(0.21404, abs=1e-5)
    assert color.MID_GREY == pytest.approx(0.46135, abs=1e-4)


def test_srgb_transfer_is_continuous_at_breakpoint():
    eps = 1e-9
    assert color.srgb_to_linear(0.04045 - eps) == pytest.approx(color.srgb_to_linear(0.04045 + eps), abs=1e-7)
    assert color.linear_to_srgb(0.0031308 - eps) == pytest.approx(color.linear_to_srgb(0.0031308 + eps), abs=1e-6)


def test_linear_to_srgb_extends_above_one_monotonically():
    x = np.linspace(0, 8, 1000)
    v = color.linear_to_srgb(x)
    assert np.all(np.diff(v) > 0)
    assert v[-1] > 1.0


def test_negative_inputs_clip_to_zero():
    assert color.srgb_to_linear(-0.5) == 0.0
    assert color.linear_to_srgb(-0.5) == 0.0


def test_neutral_white_balance_is_identity():
    assert np.allclose(color.white_balance_gains(0, 0), 1.0)


@pytest.mark.parametrize("t,m", [(100, 0), (-100, 0), (0, 100), (0, -100), (40, -30)])
def test_white_balance_preserves_grey_luminance(t, m):
    g = color.white_balance_gains(t, m)
    assert float(color.LUMA @ g) == pytest.approx(1.0)


def test_warm_is_red_cool_is_blue_magenta_has_less_green():
    warm = color.white_balance_gains(50, 0)
    cool = color.white_balance_gains(-50, 0)
    assert warm[0] > warm[2] and cool[2] > cool[0]
    mag = color.white_balance_gains(0, 50)
    assert mag[1] < mag[0] and mag[1] < mag[2]


def test_white_balance_extreme_is_about_one_stop():
    g = color.white_balance_gains(100, 0)
    assert g[0] / g[2] == pytest.approx(2.0)


V = np.linspace(0, 1, 2001)


@pytest.mark.parametrize(
    "fn,amounts",
    [
        (color.shadows_curve, [-1, -0.5, 0.5, 1]),
        (color.highlights_curve, [-1, -0.5, 0.5, 1]),
        (color.contrast_curve, [-1, -0.3, 0.3, 1]),
        (color.black_level_curve, [-1, 0.5, 1]),
    ],
)
def test_tone_components_are_monotonic(fn, amounts):
    for a in amounts:
        out = fn(V, a)
        assert np.all(np.diff(out) >= -1e-12), (fn.__name__, a)


@pytest.mark.parametrize("fn", [color.shadows_curve, color.highlights_curve, color.contrast_curve])
def test_zero_amount_is_identity(fn):
    assert np.allclose(fn(V, 0.0), V)


def test_shadows_mostly_affect_dark_tones():
    out = color.shadows_curve(V, 1.0)
    lift = out - V
    assert lift[500] > 0.2  # v=0.25 strongly lifted
    assert lift[1800] < 0.01  # v=0.9 barely touched
    assert out[0] == 0 and out[-1] == pytest.approx(1)


def test_highlights_mostly_affect_bright_tones():
    out = color.highlights_curve(V, -1.0)
    drop = V - out
    assert drop[1500] > 0.2  # v=0.75
    assert drop[200] < 0.01  # v=0.1


def test_contrast_keeps_pivot_and_endpoints():
    for a in (-1, 0.5, 1):
        out = color.contrast_curve(np.array([0.0, color.MID_GREY, 1.0]), a)
        assert out == pytest.approx([0.0, color.MID_GREY, 1.0])


def test_contrast_increases_slope_at_pivot():
    p = color.MID_GREY
    d = 1e-4
    hi = color.contrast_curve(np.array([p - d, p + d]), 1.0)
    lo = color.contrast_curve(np.array([p - d, p + d]), -1.0)
    assert (hi[1] - hi[0]) / (2 * d) == pytest.approx(2 ** 1.25, rel=1e-2)
    assert (lo[1] - lo[0]) / (2 * d) == pytest.approx(2 ** -1.25, rel=1e-2)


def test_gamma_brightens_midtones_keeps_endpoints():
    out = color.gamma_curve(np.array([0.0, 0.5, 1.0]), 2.0)
    assert out[0] == 0 and out[2] == 1 and out[1] == pytest.approx(0.5**0.5)


def test_black_level_lift_and_crush():
    assert color.black_level_curve(np.array([0.0, 1.0]), 1.0) == pytest.approx([0.2, 1.0])
    crushed = color.black_level_curve(np.array([0.1, 0.15, 1.0]), -1.0)
    assert crushed == pytest.approx([0.0, 0.0, 1.0])


def test_highlight_rolloff_compresses_overrange_smoothly():
    v = np.linspace(0, 3, 3001)
    out = color.highlight_rolloff(v, 1.0)
    assert np.all(out < 1.0 + 1e-12)
    assert np.allclose(out[v <= 0.75], v[v <= 0.75])
    slope = np.diff(out) / np.diff(v)
    assert np.all(slope > 0)
    assert np.max(np.abs(np.diff(slope))) < 0.01  # no kink at the knee
