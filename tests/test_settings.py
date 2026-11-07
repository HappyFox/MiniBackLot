import json

from minibacklot.config.settings import (
    Settings,
    TouchSettings,
    config_dir,
    load_settings,
    save_settings,
)


def test_roundtrip(tmp_path):
    path = tmp_path / "sub" / "settings.json"
    s = Settings(
        last_directory="/photos",
        background_color=(10, 20, 30),
        ui_scale=1.5,
        position_locked=True,
        touch=TouchSettings(long_press_ms=800, rotation_enabled=False),
    )
    assert save_settings(s, path)
    loaded = load_settings(path)
    assert loaded == s


def test_missing_file_gives_defaults(tmp_path):
    assert load_settings(tmp_path / "nope.json") == Settings()


def test_corrupt_file_gives_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{not json")
    assert load_settings(p) == Settings()


def test_invalid_values_fall_back_per_field(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(
        json.dumps(
            {
                "ui_scale": "big",
                "start_fullscreen": "yes",
                "background_color": [300, -5, 20],
                "default_display_mode": "zoomy",
                "controls_autohide_s": 9,
                "unknown_key": 1,
                "touch": {"long_press_ms": 750, "tap_slop": None},
            }
        )
    )
    s = load_settings(p)
    d = Settings()
    assert s.ui_scale == d.ui_scale
    assert s.start_fullscreen == d.start_fullscreen
    assert s.background_color == (255, 0, 20)
    assert s.default_display_mode == "fit"
    assert s.controls_autohide_s == 9.0
    assert s.touch.long_press_ms == 750
    assert s.touch.tap_slop == TouchSettings().tap_slop


def test_non_dict_json_gives_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("[1, 2, 3]")
    assert load_settings(p) == Settings()


def test_xdg_config_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_dir() == tmp_path / "minibacklot"
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/path")  # spec: must be absolute
    assert config_dir().name == "minibacklot" and config_dir().parent.name == ".config"


def test_save_failure_returns_false(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert not save_settings(Settings(), blocker / "settings.json")


def test_adjustments_persist_and_tolerate_garbage(tmp_path):
    from minibacklot.imaging.pipeline import Adjustments

    p = tmp_path / "settings.json"
    adj = Adjustments(exposure=0.75, temperature=-20)
    save_settings(Settings(adjustments=adj.to_dict()), p)
    assert Adjustments.from_dict(load_settings(p).adjustments) == adj
    p.write_text(json.dumps({"adjustments": "nope"}))
    assert Adjustments.from_dict(load_settings(p).adjustments).is_identity()
