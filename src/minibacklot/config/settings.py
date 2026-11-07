"""Application settings persisted as JSON in the XDG config directory.

Loading is tolerant: missing keys get defaults, unknown keys are ignored and
values of the wrong type are replaced by defaults, so a hand-edited or older
config file never prevents the application from starting.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_NAME = "minibacklot"
SETTINGS_VERSION = 1


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or ""
    root = Path(base) if base and Path(base).is_absolute() else Path.home() / ".config"
    return root / APP_NAME


def default_settings_path() -> Path:
    return config_dir() / "settings.json"


@dataclass
class TouchSettings:
    tap_slop: float = 12.0  # logical px
    tap_max_ms: int = 300
    double_tap_ms: int = 350
    double_tap_slop: float = 48.0  # logical px
    long_press_ms: int = 600
    pinch_threshold: float = 0.03
    rotation_threshold_deg: float = 8.0
    rotation_enabled: bool = True
    rotation_snap_deg: float = 4.0  # sticky snap to 0/90/180/270; 0 disables
    mouse_suppress_ms: int = 500
    wheel_zoom_step: float = 1.1


@dataclass
class Settings:
    last_directory: str | None = None
    last_image: str | None = None
    start_fullscreen: bool = True
    window_size: tuple[int, int] = (1280, 800)
    background_color: tuple[int, int, int] = (0, 0, 0)
    ui_scale: float = 0.0  # 0 = automatic
    controls_autohide_s: float = 5.0
    cursor_autohide_s: float = 2.0
    default_display_mode: str = "fit"
    position_locked: bool = False
    # Re-render settled frames with Lanczos/bicubic on a worker thread.
    high_quality_render: bool = True
    # Current lighting adjustments (see imaging.pipeline.Adjustments).
    adjustments: dict[str, float] = field(default_factory=dict)
    touch: TouchSettings = field(default_factory=TouchSettings)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["version"] = SETTINGS_VERSION
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        if not isinstance(data, dict):
            return cls()
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            default = getattr(defaults, f.name)
            value = data[f.name]
            if f.name == "touch":
                kwargs["touch"] = _coerce_dataclass(TouchSettings, value)
            else:
                coerced = _coerce(value, default, f.name)
                if coerced is not _INVALID:
                    kwargs[f.name] = coerced
        return cls(**kwargs)


_INVALID = object()


def _coerce(value: Any, default: Any, name: str) -> Any:
    if name in ("last_directory", "last_image"):
        return value if value is None or isinstance(value, str) else _INVALID
    if name == "default_display_mode":
        return value if value in ("fit", "fill", "actual") else _INVALID
    if isinstance(default, dict):
        return dict(value) if isinstance(value, dict) else _INVALID
    if isinstance(default, bool):
        return value if isinstance(value, bool) else _INVALID
    if isinstance(default, int):
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else _INVALID
    if isinstance(default, float):
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else _INVALID
    if isinstance(default, tuple):
        if (
            isinstance(value, (list, tuple))
            and len(value) == len(default)
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        ):
            vals = tuple(int(v) for v in value)
            if name == "background_color":
                vals = tuple(max(0, min(255, v)) for v in vals)
            return vals
        return _INVALID
    return value


def _coerce_dataclass(cls: type, data: Any) -> Any:
    defaults = cls()
    if not isinstance(data, dict):
        return defaults
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            v = _coerce(data[f.name], getattr(defaults, f.name), f.name)
            if v is not _INVALID:
                kwargs[f.name] = v
    return cls(**kwargs)


def load_settings(path: Path | None = None) -> Settings:
    path = path or default_settings_path()
    try:
        with open(path, encoding="utf-8") as fh:
            return Settings.from_dict(json.load(fh))
    except FileNotFoundError:
        return Settings()
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Ignoring unreadable settings file %s: %s", path, exc)
        return Settings()


def save_settings(settings: Settings, path: Path | None = None) -> bool:
    """Atomically write settings. Returns False (and logs) on failure."""
    path = path or default_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".settings-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(settings.to_dict(), fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return True
    except OSError as exc:
        log.warning("Could not save settings to %s: %s", path, exc)
        return False
