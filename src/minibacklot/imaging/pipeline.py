"""Nondestructive lighting/colour pipeline.

:class:`Adjustments` is an immutable description of the user's settings. A
:class:`ProcessingPlan` compiles it into lookup tables that can be applied
quickly to 8-bit sRGB pixels; the source pixels are never modified.

Processing order::

    8-bit sRGB --LUT--> linear light x white-balance gains x 2**exposure
               --(per pixel)--> saturation (mix with linear luminance)
               --LUT--> sRGB encode -> highlight roll-off -> shadows
                        -> highlights -> contrast -> gamma -> black level
               --> 8-bit sRGB

Exposure and white balance are applied in linear light, as a camera or raw
converter would. The tonal stages run in the perceptual (sRGB-encoded)
domain. All maths is float; values are only quantized once, at the end,
which avoids banding from repeated 8-bit rounding.

When saturation is neutral, every stage is per-channel, so the whole
pipeline collapses to three 256-entry 8-bit lookup tables (very fast).
Otherwise a float path handles the cross-channel saturation step, followed
by a 1-D tone LUT sampled on a square-root grid over linear [0, 16].

The plan is just gains, a saturation factor and LUTs, so a future GPU path
can upload the same tables as 1-D textures.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Callable

import numpy as np

from . import color

# ----------------------------------------------------------------------
# Adjustment definitions (single source of truth for UI, clamping, I/O)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class AdjustmentSpec:
    name: str
    label: str
    category: str
    minimum: float
    maximum: float
    default: float
    step: float  # increment for the -/+ buttons and keyboard
    fmt: str  # format for display, applied to the value

    def clamp(self, value: float) -> float:
        return min(self.maximum, max(self.minimum, float(value)))

    def format(self, value: float) -> str:
        return self.fmt.format(value)


ADJUSTMENT_SPECS: tuple[AdjustmentSpec, ...] = (
    AdjustmentSpec("exposure", "Exposure", "Light", -3.0, 3.0, 0.0, 0.05, "{:+.2f} EV"),
    AdjustmentSpec("contrast", "Contrast", "Light", -100, 100, 0, 1, "{:+.0f}"),
    AdjustmentSpec("gamma", "Gamma", "Light", 0.4, 2.5, 1.0, 0.01, "{:.2f}"),
    AdjustmentSpec("black_level", "Black level", "Light", -100, 100, 0, 1, "{:+.0f}"),
    AdjustmentSpec("shadows", "Shadows", "Light", -100, 100, 0, 1, "{:+.0f}"),
    AdjustmentSpec("highlights", "Highlights", "Light", -100, 100, 0, 1, "{:+.0f}"),
    AdjustmentSpec("temperature", "Temperature", "Color", -100, 100, 0, 1, "{:+.0f}"),
    AdjustmentSpec("tint", "Tint", "Color", -100, 100, 0, 1, "{:+.0f}"),
    AdjustmentSpec("saturation", "Saturation", "Color", -100, 100, 0, 1, "{:+.0f}"),
)
SPECS_BY_NAME = {s.name: s for s in ADJUSTMENT_SPECS}
CATEGORIES: tuple[str, ...] = tuple(dict.fromkeys(s.category for s in ADJUSTMENT_SPECS))


@dataclass(frozen=True)
class Adjustments:
    exposure: float = 0.0  # EV stops
    contrast: float = 0.0  # -100..100
    gamma: float = 1.0  # 0.4..2.5
    black_level: float = 0.0  # -100..100
    shadows: float = 0.0  # -100..100
    highlights: float = 0.0  # -100..100
    temperature: float = 0.0  # -100 (cool) .. 100 (warm), relative
    tint: float = 0.0  # -100 (green) .. 100 (magenta), relative
    saturation: float = 0.0  # -100 (greyscale) .. 100

    def is_identity(self) -> bool:
        return all(getattr(self, s.name) == s.default for s in ADJUSTMENT_SPECS)

    def get(self, name: str) -> float:
        return getattr(self, name)

    def with_value(self, name: str, value: float) -> Adjustments:
        spec = SPECS_BY_NAME[name]
        v = spec.clamp(value)
        # Snap tiny float drift to the default so "reset" states compare equal.
        if abs(v - spec.default) < spec.step * 1e-3:
            v = spec.default
        return replace(self, **{name: v})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Adjustments:
        if not isinstance(data, dict):
            return cls()
        adj = cls()
        for f in fields(cls):
            v = data.get(f.name)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                adj = adj.with_value(f.name, v)
        return adj


# ----------------------------------------------------------------------
# Tone curve
# ----------------------------------------------------------------------
def tone_curve(v: np.ndarray, adj: Adjustments) -> np.ndarray:
    """Map perceptual values (may exceed 1) to output values in [0, 1]."""
    v = np.asarray(v, dtype=np.float64)
    h = adj.highlights / 100.0
    v = color.highlight_rolloff(v, max(0.0, -h))
    v = np.clip(v, 0.0, 1.0)
    if adj.shadows:
        v = color.shadows_curve(v, adj.shadows / 100.0)
    if adj.highlights:
        v = color.highlights_curve(v, h)
    v = color.contrast_curve(v, adj.contrast / 100.0)
    v = color.gamma_curve(v, adj.gamma)
    v = color.black_level_curve(v, adj.black_level / 100.0)
    return np.clip(v, 0.0, 1.0)


# ----------------------------------------------------------------------
# Compiled plan
# ----------------------------------------------------------------------
TONE_LUT_SIZE = 16384
TONE_LUT_MAX = 16.0  # linear input range covered by the tone LUT
STRIP_ROWS = 128  # rows per processing strip (memory + cancellation)

_SRGB8_TO_LINEAR = color.srgb_to_linear(np.arange(256) / 255.0)


class Cancelled(Exception):
    pass


class ProcessingPlan:
    def __init__(self, adj: Adjustments) -> None:
        self.adjustments = adj
        gains = color.white_balance_gains(adj.temperature, adj.tint) * (2.0**adj.exposure)
        self.gains = gains
        in_lut = _SRGB8_TO_LINEAR[None, :] * gains[:, None]  # (3, 256) float64
        self.saturation = 1.0 + adj.saturation / 100.0
        self.fast_luts: np.ndarray | None = None
        if self.saturation == 1.0:
            out = tone_curve(color.linear_to_srgb(in_lut), adj)
            self.fast_luts = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        else:
            self.in_lut = in_lut.astype(np.float32)
            grid = np.linspace(0.0, 1.0, TONE_LUT_SIZE) ** 2 * TONE_LUT_MAX
            out = tone_curve(color.linear_to_srgb(grid), adj)
            self.tone_lut = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def apply(
        self,
        rgb: np.ndarray,
        should_cancel: Callable[[], bool] | None = None,
        strip_rows: int = STRIP_ROWS,
    ) -> np.ndarray:
        """Return a new (H, W, 3) uint8 array; ``rgb`` is not modified.

        Raises :class:`Cancelled` if ``should_cancel()`` becomes true
        between strips.
        """
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected an (H, W, 3) uint8 array")
        out = np.empty_like(rgb)
        for y0 in range(0, rgb.shape[0], strip_rows):
            if should_cancel is not None and should_cancel():
                raise Cancelled
            src = rgb[y0 : y0 + strip_rows]
            dst = out[y0 : y0 + strip_rows]
            if self.fast_luts is not None:
                for c in range(3):
                    dst[..., c] = self.fast_luts[c][src[..., c]]
            else:
                self._apply_float(src, dst)
        return out

    def _apply_float(self, src: np.ndarray, dst: np.ndarray) -> None:
        r = self.in_lut[0][src[..., 0]]
        g = self.in_lut[1][src[..., 1]]
        b = self.in_lut[2][src[..., 2]]
        y = color.LUMA[0] * r + color.LUMA[1] * g + color.LUMA[2] * b
        s = np.float32(self.saturation)
        scale = np.float32(1.0 / TONE_LUT_MAX)
        n = np.float32(TONE_LUT_SIZE - 1)
        for c, ch in enumerate((r, g, b)):
            ch -= y
            ch *= s
            ch += y
            np.clip(ch, 0.0, TONE_LUT_MAX, out=ch)
            ch *= scale
            np.sqrt(ch, out=ch)
            ch *= n
            ch += np.float32(0.5)
            dst[..., c] = self.tone_lut[ch.astype(np.intp)]


def process_array(rgb: np.ndarray, adj: Adjustments) -> np.ndarray:
    """Convenience wrapper: apply ``adj`` to an 8-bit sRGB array."""
    if adj.is_identity():
        return rgb.copy()
    return ProcessingPlan(adj).apply(rgb)


# ----------------------------------------------------------------------
# Editable state with undo
# ----------------------------------------------------------------------
class AdjustmentState:
    """Current adjustments plus undo history.

    ``on_history_push`` is called whenever a snapshot is recorded, so the
    application can interleave adjustment and view undo in one stack.
    """

    def __init__(self, initial: Adjustments | None = None, history_limit: int = 100) -> None:
        self.current = initial or Adjustments()
        self.revision = 0
        self._history: deque[Adjustments] = deque(maxlen=history_limit)
        self.on_history_push: Callable[[], None] | None = None

    def push_history(self) -> None:
        if not self._history or self._history[-1] != self.current:
            self._history.append(self.current)
            if self.on_history_push:
                self.on_history_push()

    def _set(self, adj: Adjustments) -> bool:
        if adj == self.current:
            return False
        self.current = adj
        self.revision += 1
        return True

    def set_value(self, name: str, value: float) -> bool:
        """Continuous change (slider drag); call push_history() at drag start."""
        return self._set(self.current.with_value(name, value))

    def step(self, name: str, steps: int) -> bool:
        spec = SPECS_BY_NAME[name]
        self.push_history()
        v = round((self.current.get(name) + steps * spec.step) / spec.step) * spec.step
        return self._set(self.current.with_value(name, v))

    def reset(self, name: str) -> bool:
        if self.current.get(name) == SPECS_BY_NAME[name].default:
            return False
        self.push_history()
        return self._set(self.current.with_value(name, SPECS_BY_NAME[name].default))

    def reset_all(self) -> bool:
        if self.current.is_identity():
            return False
        self.push_history()
        return self._set(Adjustments())

    def replace_all(self, adj: Adjustments) -> bool:
        self.push_history()
        return self._set(adj)

    @property
    def can_undo(self) -> bool:
        return any(a != self.current for a in self._history)

    def undo(self) -> bool:
        while self._history:
            prev = self._history.pop()
            if prev != self.current:
                return self._set(prev)
        return False
