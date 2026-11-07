# MiniBacklot

**Digital backdrops for miniature photography.**

MiniBacklot shows a photograph fullscreen on a monitor placed behind a toy or
miniature scene, so it can be photographed as a background. It is built for
touchscreens first: everything can be done with your fingers, while mouse and
keyboard keep working.

Status: **Phase 2 — touch-first viewer with lighting and colour adjustments.**
Filters, presets and calibration screens are planned (see *Roadmap*).

## Install (development)

Requires Python 3.12+ on Linux.

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'
# optional, for 16-bit RGB TIFFs and later phases:
uv pip install --python .venv/bin/python -e '.[opencv]'
```

(Plain `python -m venv .venv && .venv/bin/pip install -e '.[dev]'` works too.)

## Run

```sh
.venv/bin/minibacklot ~/Pictures/backdrops          # a folder
.venv/bin/minibacklot ~/Pictures/backdrops/sky.jpg  # start at a file
.venv/bin/minibacklot --windowed                    # reopen the last folder in a window
```

Options: `--fullscreen` / `--windowed`, `--ui-scale 1.5`,
`--background '#202020'`, `--config PATH`, `-v`.

Supported formats: JPEG, PNG, TIFF, WebP. EXIF orientation is applied and
embedded ICC profiles are converted to sRGB. Source images are only ever
read, never written.

## Touch controls

| Gesture | Action |
|---|---|
| Tap | Show / hide the controls |
| One-finger drag | Pan |
| Two-finger pinch | Zoom around the point between your fingers |
| Two-finger twist | Rotate (engages after ~8° so pinching doesn't wobble; gently snaps to 0°/90°/180°/270°) |
| Double-tap | Fit the image again (keeps rotation and flips) |
| Long-press on image | Show image info |
| Long-press on a button | Explain what the button does |

The toolbar contains Prev/Next, Fit / Fill / 1:1, Flip H / Flip V, 0° (remove
rotation), **Adjust**, Undo, Reset, **Lock**, Window/Full, **Hide** and Quit
(tap twice). Undo steps back through position *and* adjustment changes in the
order you made them.

**Lock** freezes the image position: pan, zoom, rotate, flip and the display
mode buttons are disabled until you unlock it. Taps still show and hide the
controls. The lock state is remembered between sessions, and switching to
another image keeps a manually framed view.

Rotation snapping is *sticky*: within ±4° of a right angle the image locks
exactly onto 0°, 90°, 180° or 270°, and releases as soon as you twist past
that window. Set `touch.rotation_snap_deg` to change the window, or to `0`
to turn snapping off. Keyboard `[`/`]` always rotates in exact steps.

Controls hide themselves after 5 seconds without interaction, but never
while a finger is on them or while the Adjust panel is open.

## Lighting and colour adjustments

**Adjust** opens a side panel with two tabs:

| Light | Color |
|---|---|
| Exposure (EV stops, applied in linear light) | Temperature (warmer / cooler) |
| Contrast | Tint (green / magenta) |
| Gamma (midtone brightness) | Saturation |
| Black level (lift or crush the shadow floor) | |
| Shadows | |
| Highlights (negative values also recover detail pushed past white) | |

Each row shows its value (amber when changed) and has large **−/+** buttons
(hold to repeat), a slider and a **reset** button. Touching the slider thumb
grabs it without jumping, so you can fine-tune with your finger on the thumb.
Touching elsewhere on the track jumps there. The slider snaps to its default
as you drag past it. Drag the label area to scroll the panel. **Reset all**
returns everything to neutral, and long-pressing any control explains it.

Adjustments never modify your files. They apply to whichever image is shown
and are remembered between sessions (per-image adjustments and named presets
come in Phase 4). Temperature and tint are *relative* creative controls, not
calibrated Kelvin values. Shadows and highlights are global tone-curve
adjustments, not local (spatially-aware) tone mapping.

While an adjustment is being processed, the previous picture stays on
screen; processing never blocks touch input.

## Mouse and keyboard

* Left-drag: pan. Wheel: zoom at the cursor. Ctrl/Shift + wheel: rotate (with snapping).
* `←`/`→`, PgUp/PgDn, Space, `N`/`P`: previous/next image. Home/End: first/last.
* `0` fit, `1` actual size, `2` fill, `+`/`-` zoom, `[`/`]` rotate 1°
  (Shift: 15°), `R` remove rotation, `H`/`V` flip, Backspace reset.
* Shift + arrows: pan. `L` lock. `U` or Ctrl+Z undo. `I` info. `A` adjust panel.
* Mouse wheel over the Adjust panel scrolls it.
* Tab or `C`: toggle controls. `F`/F11: fullscreen. Esc: hide controls, then
  leave fullscreen. `Q`: quit.

## Configuration

Settings are stored as JSON in `$XDG_CONFIG_HOME/minibacklot/settings.json`
(usually `~/.config/minibacklot/settings.json`): last folder and image,
background colour, UI scale, auto-hide delays, default display mode, the
current lighting adjustments and touch sensitivity (`touch.tap_slop`, `touch.long_press_ms`,
`touch.double_tap_ms`, `touch.rotation_threshold_deg`,
`touch.rotation_enabled`, `touch.rotation_snap_deg`, ...). Invalid or unknown entries are ignored.

## Tests

```sh
.venv/bin/pytest
```

Gesture recognition, input normalization, transforms, rendering and image
loading are all tested without a display or touch hardware (SDL's `dummy`
video driver is used for the renderer tests).

## Architecture

```
src/minibacklot/
  application.py          window, event loop, wiring
  events/input_manager.py  touch/mouse -> pointer events, duplicate suppression
  events/gesture_recognizer.py  pan / pinch / rotate / tap / double-tap / long-press
  display/transforms.py   immutable image<->screen transform, fit/fill math
  display/view.py         view controller: modes, lock, undo
  display/renderer.py     pyramid + visible-crop rendering
  imaging/loader.py       decoding, EXIF/ICC, pyramid, async loader + LRU cache
  imaging/color.py        sRGB transfer, white balance, tone-curve maths
  imaging/pipeline.py     Adjustments, compiled ProcessingPlan, undoable state
  imaging/processor.py    background worker: processes pyramid levels
  ui/                     theme, widget, button, slider, toolbar, side panel, toast
  config/settings.py      XDG JSON settings
```

Pointer events go to the toolbar first; a widget that accepts a touch
*captures* that finger until it lifts, so dragging on a control never moves
the image. Everything else goes to the gesture recognizer, whose output is
applied through the view controller (which enforces the lock).

Images are decoded on a worker thread into a pyramid of half-size levels,
each made with a Lanczos filter (a plain 2×2 average would bake moiré into
fine textures such as fabric, foliage or brickwork). Each frame scales and
rotates only the visible part of the most suitable level, so 24 MP+ images
stay fluid. Rendering happens in three passes:

1. **During a gesture:** nearest-neighbour, about 4 ms per frame.
2. **120 ms after release:** `smoothscale`/`rotozoom`, about 12 ms.
3. **Then a high-quality pass on a worker thread** (`display/hq_render.py`).
   It takes the visible area from a level *finer* than needed, shrinks it
   with Lanczos, then applies rotation, flips and *sub-pixel* positioning in
   one bicubic affine transform. It costs about 100–140 ms per 1080p frame,
   is ready about a quarter of a second after release, and replaces the
   pass-2 frame when done. Measured against a direct Lanczos resize of the
   full image, its error is 2–3× lower in normal views, and more than 10×
   lower near 1:1, where `smoothscale` aliases. Set `"high_quality_render":
   false` in the settings to skip it on slow hardware. The
loop blocks in `pygame.event.wait()`, so an idle backdrop uses essentially no
CPU.

### Processing pipeline

```
8-bit sRGB ─LUT→ linear × white-balance gains × 2^exposure
           ─→ saturation (mix with linear luminance)
           ─LUT→ sRGB encode → highlight roll-off → shadows → highlights
                 → contrast (S-curve about mid grey) → gamma → black level
           ─→ 8-bit sRGB
```

Everything runs in floating point and is quantized once, at the end. Each
tonal component is designed to be monotonic, so no tones are inverted.
Adjustments are compiled into a `ProcessingPlan`: per-channel gains, a
saturation factor and lookup tables. When saturation is neutral, the whole
pipeline collapses to three 8-bit LUTs (about 13 ms for a 2 MP screen-size
preview and about 95 ms for 24 MP on the development machine). Otherwise a
float path with a 16 k-entry tone LUT is used (about 38 ms / 410 ms). These
same tables are what a future GPU shader would upload as textures.

Processing runs on a worker thread, level by level, starting with the
pyramid level currently on screen, then coarser and finer levels. New slider
values supersede old work, but the first level always finishes, so the
preview keeps updating during a continuous drag. Until the full-resolution
level is ready, the renderer uses the nearest processed level, so colours
are always correct, if briefly softer when zoomed in. Finished results are
cached for the last two image/adjustment combinations.

## Platform notes and known limitations

* **Wayland vs X11:** SDL2 picks X11 (XWayland) by default. To run natively,
  use `SDL_VIDEODRIVER=wayland minibacklot ...`. Both have been launched
  successfully. Real multitouch hardware has not been tested yet.
* **Touch → mouse duplicates:** MiniBacklot sets `SDL_TOUCH_MOUSE_EVENTS=0`
  and `SDL_MOUSE_TOUCH_EVENTS=0`, ignores mouse events flagged
  `event.touch`, and also ignores mouse events within 500 ms of a touch
  (`touch.mouse_suppress_ms`), because some drivers don't flag them.
* **Touchpad pinch** gestures are not exposed by SDL2, so touchpads can only
  scroll-zoom.
* **DPI:** SDL2 doesn't report reliable physical DPI on Linux, so the UI
  scale is guessed from the window height (1.0 at 1080p). Use `--ui-scale`
  or the `ui_scale` setting to override.
* **Screen blanking** is inhibited with `pygame.display.set_allow_screensaver(False)`.
  On Wayland this needs compositor support for the idle-inhibit protocol.
  Some desktop power settings may still blank the screen.
* **Colour management:** embedded ICC profiles are converted to sRGB. The
  display's own profile is not applied.
* Files Pillow can't decode (e.g. some 16-bit RGB TIFFs) are retried with
  OpenCV if the `opencv` extra is installed.

## Roadmap

1. ~~Touch-first viewer~~
2. ~~Lighting: exposure (linear light), temperature, tint, contrast,
   saturation, gamma, black level, shadows/highlights, with touch sliders~~
3. Filters: Gaussian blur, luma/chroma denoise, anti-moiré, dithering
4. Workflow: presets, calibration screens, solid colours, a thumbnail
   browser and folder picker (so opening folders doesn't need the
   command line)
5. Optimization, optional GPU shaders, packaging, desktop integration

## License

MIT — see [LICENSE](LICENSE).
