"""Colour mathematics: sRGB transfer functions, white balance and tone curves.

Conventions
-----------
* *Linear* values are scene-referred linear-light RGB (sRGB primaries,
  D65). 1.0 is diffuse white; values above 1.0 are allowed until the final
  tone stage.
* *Perceptual* values ``v`` are sRGB-encoded (``linear_to_srgb``). Tonal
  controls (shadows, highlights, contrast, gamma, black level) operate in
  this domain, where equal steps look roughly equally large.

All functions accept scalars or NumPy arrays and are pure, so the same maths
can later be ported to a GPU shader (the pipeline only needs per-channel
gains, a saturation factor and a 1-D tone lookup table).

Temperature and tint are *relative* creative adjustments, not calibrated
Kelvin or Duv values.
"""

from __future__ import annotations

import numpy as np

# Rec. 709 / sRGB luminance weights for linear RGB.
LUMA = np.array([0.2126, 0.7152, 0.0722])

# Mid grey (18 % linear reflectance) in the perceptual domain, ~0.4614.
MID_GREY_LINEAR = 0.18


def srgb_to_linear(v):
    """sRGB electro-optical transfer function (IEC 61966-2-1). Input clipped at 0."""
    v = np.maximum(np.asarray(v, dtype=np.float64), 0.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """Inverse of :func:`srgb_to_linear`; extends smoothly above 1.0."""
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


MID_GREY = float(linear_to_srgb(MID_GREY_LINEAR))


def white_balance_gains(temperature: float, tint: float) -> np.ndarray:
    """Relative per-channel linear gains for temperature/tint in [-100, 100].

    Positive temperature warms (more red, less blue); positive tint shifts
    towards magenta (less green). At the extremes red and blue differ by about
    one stop. Gains are normalized so the luminance of neutral grey is
    unchanged, so white balance does not alter overall brightness.
    """
    t = float(np.clip(temperature, -100, 100)) / 100.0
    m = float(np.clip(tint, -100, 100)) / 100.0
    gains = np.array([2.0 ** (0.5 * t), 2.0 ** (-0.5 * m), 2.0 ** (-0.5 * t)])
    return gains / float(LUMA @ gains)


# ----------------------------------------------------------------------
# Tone curve components (perceptual domain)
# ----------------------------------------------------------------------
_BUMP_PEAK = 0.25 * 0.75**3  # max of v * (1 - v)^3, at v = 0.25


def _shadow_bump(v):
    """Smooth bump: 0 at v=0 and v=1, peak 1.0 at v=0.25."""
    return v * (1.0 - v) ** 3 / _BUMP_PEAK


def shadows_curve(v, amount: float):
    """Lift (amount > 0) or deepen (amount < 0) shadows; amount in [-1, 1].

    The strength limits keep the curve monotonic, so no tonal inversions.
    Peak effect at v = 0.25: +0.25 when lifting, -0.10 when deepening.
    """
    k = 0.25 * amount if amount > 0 else 0.10 * amount
    return v + k * _shadow_bump(v)


def highlights_curve(v, amount: float):
    """Brighten (amount > 0) or tame (amount < 0) highlights; amount in [-1, 1].

    The mirror image of :func:`shadows_curve` (peak at v = 0.75).
    """
    k = 0.10 * amount if amount > 0 else 0.25 * amount
    return v + k * _shadow_bump(1.0 - v)


def highlight_rolloff(v, amount: float, knee: float = 0.75):
    """Softly compress values above ``knee`` so over-range values approach 1.

    ``amount`` in [0, 1] blends between hard clipping (0) and a tanh
    shoulder (1). The shoulder has slope 1 at the knee, so there is no kink.
    Used to recover detail pushed above white by exposure.
    """
    if amount <= 0:
        return v
    w = 1.0 - knee
    rolled = np.where(v > knee, knee + w * np.tanh((v - knee) / w), v)
    return v + amount * (rolled - v)


def contrast_curve(v, amount: float, pivot: float = MID_GREY):
    """S-curve around ``pivot`` that keeps 0 and 1 fixed; amount in [-1, 1].

    The slope at the pivot is ``2 ** (1.25 * amount)`` (about 0.42 to 2.4).
    Being a power curve on each side, it never clips.
    """
    if amount == 0:
        return v
    k = 2.0 ** (1.25 * amount)
    v = np.clip(v, 0.0, 1.0)
    lo = pivot * np.power(v / pivot, k)
    hi = 1.0 - (1.0 - pivot) * np.power(np.maximum(1.0 - v, 0.0) / (1.0 - pivot), k)
    return np.where(v < pivot, lo, hi)


def gamma_curve(v, gamma: float):
    """Midtone brightness: ``v ** (1 / gamma)``. gamma > 1 brightens midtones."""
    if gamma == 1.0:
        return v
    return np.power(np.clip(v, 0.0, 1.0), 1.0 / gamma)


def black_level_curve(v, amount: float):
    """Raise (amount > 0) or crush (amount < 0) the shadow floor; amount in [-1, 1].

    +1 lifts black to 0.20 (perceptual); -1 maps everything below 0.15 to black.
    """
    if amount > 0:
        floor = 0.20 * amount
        return floor + (1.0 - floor) * v
    if amount < 0:
        cut = -0.15 * amount
        return np.clip((v - cut) / (1.0 - cut), 0.0, None)
    return v
