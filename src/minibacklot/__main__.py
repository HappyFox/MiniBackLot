"""Command-line entry point: ``minibacklot [PATH]`` or ``python -m minibacklot``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from . import __version__  # noqa: E402
from .config.settings import default_settings_path, load_settings  # noqa: E402


def _parse_color(text: str) -> tuple[int, int, int]:
    t = text.strip().lstrip("#")
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    if len(t) != 6:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB, got {text!r}")
    try:
        return (int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected #RRGGBB, got {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="minibacklot",
        description="Digital backdrops for miniature photography.",
    )
    p.add_argument("path", nargs="?", type=Path, help="image file or directory to show")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fullscreen", dest="fullscreen", action="store_true", default=None, help="start fullscreen")
    mode.add_argument("--windowed", dest="fullscreen", action="store_false", help="start in a window")
    p.add_argument("--ui-scale", type=float, default=None, help="UI scale factor (default: automatic)")
    p.add_argument("--background", type=_parse_color, default=None, help="background color, e.g. #202020")
    p.add_argument("--config", type=Path, default=None, help=f"settings file (default: {default_settings_path()})")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings_path = args.config or default_settings_path()
    settings = load_settings(settings_path)
    if args.background is not None:
        settings.background_color = args.background

    from .application import Application  # imports pygame

    app = Application(
        settings,
        settings_path=settings_path,
        start_path=args.path,
        fullscreen=args.fullscreen,
        ui_scale=args.ui_scale,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
