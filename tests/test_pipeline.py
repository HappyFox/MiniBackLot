"""Image-processing pipeline and adjustment state."""

import numpy as np
import pytest

from minibacklot.imaging import color
from minibacklot.imaging.pipeline import (
    ADJUSTMENT_SPECS,
    Adjustments,
    AdjustmentState,
    Cancelled,
    ProcessingPlan,
    process_array,
    tone_curve,
)


def ramp_image():
    """Every 8-bit value in every channel, plus some colours."""
    v = np.arange(256, dtype=np.uint8)
    grey = np.stack([v, v, v], axis=-1)[None, :, :]
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, (31, 256, 3), dtype=np.uint8)
    return np.concatenate([grey, noise], axis=0)


def grey(value):
    return np.full((2, 2, 3), value, dtype=np.uint8)


def test_identity_is_exact():
    img = ramp_image()
    assert np.array_equal(ProcessingPlan(Adjustments()).apply(img), img)


def test_identity_float_path_is_near_exact():
    # Saturation 1.0 via the float path: force it with a tiny non-zero value.
    img = ramp_image()
    out = ProcessingPlan(Adjustments(saturation=1e-9)).apply(img)
    assert np.max(np.abs(out.astype(int) - img.astype(int))) <= 1


def test_source_is_not_modified():
    img = ramp_image()
    before = img.copy()
    process_array(img, Adjustments(exposure=1, saturation=50, contrast=30))
    assert np.array_equal(img, before)


def test_exposure_plus_one_stop_doubles_linear_light():
    out = process_array(grey(100), Adjustments(exposure=1.0))
    lin_in = color.srgb_to_linear(100 / 255)
    lin_out = color.srgb_to_linear(out[0, 0, 0] / 255)
    assert lin_out == pytest.approx(2 * lin_in, rel=0.02)


def test_exposure_minus_one_stop_halves_linear_light():
    out = process_array(grey(200), Adjustments(exposure=-1.0))
    assert color.srgb_to_linear(out[0, 0, 0] / 255) == pytest.approx(
        color.srgb_to_linear(200 / 255) / 2, rel=0.02
    )


def test_exposure_clips_at_white_without_wrapping():
    out = process_array(grey(250), Adjustments(exposure=3.0))
    assert (out == 255).all()


def test_highlight_recovery_keeps_overexposed_detail():
    img = np.array([[[200, 200, 200], [230, 230, 230]]], dtype=np.uint8)
    clipped = process_array(img, Adjustments(exposure=1.5))
    recovered = process_array(img, Adjustments(exposure=1.5, highlights=-100))
    assert clipped[0, 0, 0] == clipped[0, 1, 0] == 255  # detail lost
    assert recovered[0, 0, 0] < recovered[0, 1, 0] < 255  # detail kept


def test_temperature_warms_neutral_grey():
    out = process_array(grey(128), Adjustments(temperature=60))[0, 0]
    assert out[0] > out[1] > out[2]
    out = process_array(grey(128), Adjustments(temperature=-60))[0, 0]
    assert out[2] > out[1] > out[0]


def test_tint_magenta_reduces_green():
    out = process_array(grey(128), Adjustments(tint=60))[0, 0]
    assert out[1] < out[0] and out[1] < out[2]


def test_full_desaturation_gives_equal_channels():
    img = ramp_image()
    out = process_array(img, Adjustments(saturation=-100))
    assert np.max(np.abs(out[..., 0].astype(int) - out[..., 1])) <= 1
    assert np.max(np.abs(out[..., 1].astype(int) - out[..., 2])) <= 1


def test_saturation_preserves_neutrals():
    out = process_array(grey(90), Adjustments(saturation=80))
    assert np.max(np.ptp(out, axis=-1)) <= 1


def test_saturation_increases_colourfulness():
    img = np.array([[[180, 120, 100]]], dtype=np.uint8)
    out = process_array(img, Adjustments(saturation=60))[0, 0].astype(int)
    assert np.ptp(out) > np.ptp(img[0, 0].astype(int))


def test_contrast_darkens_shadows_and_brightens_highlights():
    img = np.array([[[50, 50, 50], [220, 220, 220]]], dtype=np.uint8)
    out = process_array(img, Adjustments(contrast=60))
    assert out[0, 0, 0] < 50 and out[0, 1, 0] > 220


def test_black_level_raises_floor():
    out = process_array(grey(0), Adjustments(black_level=100))
    assert out[0, 0, 0] == round(0.2 * 255)


def test_gamma_brightens_midtones():
    out = process_array(grey(128), Adjustments(gamma=1.8))
    assert out[0, 0, 0] > 128


@pytest.mark.parametrize(
    "adj",
    [
        Adjustments(exposure=0.7, contrast=40, shadows=50, highlights=-60, gamma=1.2),
        Adjustments(exposure=-0.4, contrast=-50, black_level=-40),
        Adjustments(exposure=2.0, highlights=-100, shadows=-100),
    ],
)
def test_tone_curve_is_monotonic_no_inversions(adj):
    v = np.linspace(0, 4, 5000)
    out = tone_curve(v, adj)
    assert np.all(np.diff(out) >= -1e-12)
    # And the 8-bit grey ramp stays ordered after processing.
    ramp = ramp_image()[:1]
    res = process_array(ramp, adj)[0, :, 0].astype(int)
    assert np.all(np.diff(res) >= 0)


def test_no_banding_from_moderate_adjustments():
    """Moderate edits shouldn't merge many adjacent levels in the midtones."""
    ramp = ramp_image()[:1]
    out = process_array(ramp, Adjustments(exposure=-0.3, contrast=10))[0, 64:200, 0]
    assert len(np.unique(out)) > 0.85 * len(out)


def test_fast_and_float_paths_agree():
    img = ramp_image()
    adj = Adjustments(exposure=0.4, temperature=25, tint=-10, contrast=20, gamma=1.1, shadows=20)
    fast = ProcessingPlan(adj)
    assert fast.fast_luts is not None
    slow = ProcessingPlan(Adjustments(**{**adj.to_dict(), "saturation": 1e-9}))
    assert slow.fast_luts is None
    diff = np.abs(fast.apply(img).astype(int) - slow.apply(img).astype(int))
    assert diff.max() <= 1


def test_strip_processing_matches_and_can_cancel():
    img = np.random.default_rng(2).integers(0, 256, (300, 50, 3), dtype=np.uint8)
    plan = ProcessingPlan(Adjustments(saturation=30))
    assert np.array_equal(plan.apply(img, strip_rows=7), plan.apply(img, strip_rows=1000))
    calls = []

    def cancel():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(Cancelled):
        plan.apply(img, should_cancel=cancel, strip_rows=10)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        ProcessingPlan(Adjustments(exposure=1)).apply(np.zeros((4, 4), np.uint8))


# ----------------------------------------------------------------------
# Adjustments value object
# ----------------------------------------------------------------------
def test_specs_cover_all_fields():
    assert {s.name for s in ADJUSTMENT_SPECS} == set(Adjustments().to_dict())
    for s in ADJUSTMENT_SPECS:
        assert getattr(Adjustments(), s.name) == s.default
        assert s.minimum < s.default < s.maximum


def test_with_value_clamps_and_snaps():
    a = Adjustments().with_value("exposure", 99)
    assert a.exposure == 3.0
    assert Adjustments().with_value("gamma", 1.0 + 1e-9).gamma == 1.0
    assert Adjustments().with_value("gamma", 1.0 + 1e-9).is_identity()


def test_adjustments_dict_roundtrip_and_tolerance():
    a = Adjustments(exposure=1.25, tint=-30, gamma=1.4)
    assert Adjustments.from_dict(a.to_dict()) == a
    bad = Adjustments.from_dict({"exposure": "x", "contrast": 1e9, "gamma": float("nan"), "nope": 1})
    assert bad.exposure == 0 and bad.contrast == 100 and bad.gamma == 1.0
    assert Adjustments.from_dict(None) == Adjustments()


def test_format():
    from minibacklot.imaging.pipeline import SPECS_BY_NAME

    assert SPECS_BY_NAME["exposure"].format(0.5) == "+0.50 EV"
    assert SPECS_BY_NAME["gamma"].format(1.0) == "1.00"
    assert SPECS_BY_NAME["contrast"].format(-12) == "-12"


# ----------------------------------------------------------------------
# AdjustmentState
# ----------------------------------------------------------------------
def test_state_drag_then_undo():
    st = AdjustmentState()
    pushes = []
    st.on_history_push = lambda: pushes.append(1)
    st.push_history()  # drag start
    for v in (0.1, 0.5, 0.9):
        st.set_value("exposure", v)
    assert st.current.exposure == 0.9
    assert pushes == [1]
    assert st.undo()
    assert st.current.is_identity()
    assert not st.undo()


def test_state_step_reset_and_reset_all():
    st = AdjustmentState()
    st.step("exposure", 3)
    assert st.current.exposure == pytest.approx(0.15)
    st.step("contrast", -5)
    assert st.current.contrast == -5
    assert st.reset("exposure")
    assert st.current.exposure == 0 and st.current.contrast == -5
    assert not st.reset("exposure")  # already default: no history entry
    assert st.reset_all()
    assert st.current.is_identity()
    assert st.undo() and st.current.contrast == -5


def test_state_revision_only_changes_on_real_change():
    st = AdjustmentState()
    r = st.revision
    st.set_value("exposure", 0)
    assert st.revision == r
    st.set_value("exposure", 0.5)
    assert st.revision == r + 1
