"""High-quality settle rendering and anti-aliased pyramid."""

import time

import numpy as np
import pytest
from PIL import Image

from minibacklot.display.hq_render import (
    HQRenderer,
    available_arrays,
    choose_level,
    level_widths,
    render_hq,
)
from minibacklot.display.transforms import ViewTransform
from minibacklot.imaging.loader import ImagePyramid, ProcessedPyramid


def quadrants(w=200, h=100):
    img = Image.new("RGB", (w, h), (255, 0, 0))
    img.paste((0, 0, 255), (w // 2, 0, w, h))
    img.paste((0, 255, 0), (w // 2, h // 2, w, h))
    return ImagePyramid.from_pil(img, min_side=16)


def hq(pyr, t, screen=(300, 200), bg=(9, 9, 9)):
    return np.asarray(render_hq(available_arrays(pyr), level_widths(pyr), pyr.base_size, t, screen, bg))


def stripes(w=1200, h=800, period=2.6, angle=0.1):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 127.5 + 127.5 * np.cos(2 * np.pi * (x * np.cos(angle) + y * np.sin(angle)) / period)
    return Image.fromarray(np.stack([v.astype(np.uint8)] * 3, -1))


def test_unrotated_placement_and_background():
    out = hq(quadrants(), ViewTransform(cx=150, cy=100, scale=1.0))
    assert tuple(out[70, 100]) == (255, 0, 0)
    assert tuple(out[70, 200]) == (0, 0, 255)
    assert tuple(out[130, 200]) == (0, 255, 0)
    assert tuple(out[10, 10]) == (9, 9, 9)


@pytest.mark.parametrize("rotation,flip_h", [(45, False), (-30, True), (90, False)])
def test_rotation_and_flip_match_transform(rotation, flip_h):
    pyr = quadrants()
    t = ViewTransform(cx=200, cy=200, scale=1.3, rotation=rotation, flip_h=flip_h)
    out = hq(pyr, t, screen=(400, 400))
    for img_pt, expected in [((50, 25), (255, 0, 0)), ((150, 25), (0, 0, 255)), ((150, 75), (0, 255, 0))]:
        sx, sy = t.image_to_screen(img_pt, pyr.base_size)
        assert tuple(out[round(sy), round(sx)]) == expected


def test_subpixel_positioning():
    """A half-pixel shift moves an edge by half a pixel (blended value)."""
    img = Image.new("RGB", (100, 100), (0, 0, 0))
    img.paste((255, 255, 255), (50, 0, 100, 100))  # edge at x=50
    pyr = ImagePyramid.from_pil(img, min_side=1000)
    a = hq(pyr, ViewTransform(cx=50.0, cy=50, scale=1.0), screen=(100, 100), bg=(0, 0, 0))
    b = hq(pyr, ViewTransform(cx=50.5, cy=50, scale=1.0), screen=(100, 100), bg=(0, 0, 0))
    assert a[50, 49, 0] < 10 and a[50, 50, 0] > 245  # crisp edge
    assert 80 < b[50, 50, 0] < 175  # pixel 50 now half covered


def test_downscale_has_no_moire():
    pyr = ImagePyramid.from_pil(stripes(), min_side=64)
    # Displayed period ~0.7 px: far beyond Nyquist, should render as flat grey.
    for rot in (0, 7, 33):
        out = hq(pyr, ViewTransform(cx=200, cy=150, scale=0.27, rotation=rot), screen=(400, 300))
        centre = out[110:190, 150:250, 0].astype(float)
        assert centre.std() < 3.0, rot
        assert abs(centre.mean() - 127.5) < 6


def test_pyramid_levels_are_antialiased():
    pyr = ImagePyramid.from_pil(stripes(), min_side=64)
    # Stripes (period 2.6 px) are beyond level 1's Nyquist limit: should be ~flat.
    assert pyr.arrays[1][50:350, 50:550, 0].astype(float).std() < 3.0


def test_upscale_is_sharper_than_bilinear():
    img = Image.new("RGB", (40, 40), (0, 0, 0))
    img.paste((255, 255, 255), (20, 0, 40, 40))
    pyr = ImagePyramid.from_pil(img, min_side=1000)
    out = hq(pyr, ViewTransform(cx=200, cy=200, scale=10.0), screen=(400, 400), bg=(0, 0, 0))
    row = out[200, :, 0].astype(int)
    transition = np.count_nonzero((row > 25) & (row < 230))
    assert transition <= 12  # bilinear at 10x would blur over ~10+ px with soft tails


def test_choose_level_prefers_one_finer_when_affordable():
    pyr = ImagePyramid.from_pil(Image.new("RGB", (4000, 2000)), min_side=64)
    arrays, widths = available_arrays(pyr), level_widths(pyr)
    t = ViewTransform(cx=400, cy=300, scale=0.2)  # ideal level 2 (0.25)
    assert choose_level(arrays, widths, t, pyr.base_size, (800, 600)) == 1


def test_choose_level_respects_cost_limit(monkeypatch):
    import minibacklot.display.hq_render as m

    pyr = ImagePyramid.from_pil(Image.new("RGB", (4000, 2000)), min_side=64)
    monkeypatch.setattr(m, "MAX_CROP_PIXELS", 600_000)
    t = ViewTransform(cx=400, cy=300, scale=0.2)  # level 1 crop = 2000x1000 > limit
    assert m.choose_level(available_arrays(pyr), level_widths(pyr), t, pyr.base_size, (800, 600)) == 2


def test_works_with_partial_processed_pyramid():
    src = quadrants(400, 200)
    pp = ProcessedPyramid(src, key="k")
    pp.add(2, src.arrays[2])  # only a coarse level available
    out = np.asarray(
        render_hq(available_arrays(pp), level_widths(pp), pp.base_size, ViewTransform(150, 100, 0.5), (300, 200), (0, 0, 0))
    )
    assert tuple(out[60, 110]) == (255, 0, 0)


def test_offscreen_gives_background():
    out = hq(quadrants(), ViewTransform(cx=9000, cy=9000, scale=1.0), bg=(1, 2, 3))
    assert (out == (1, 2, 3)).all()


def test_worker_latest_request_wins(pg_display):
    pyr = quadrants()
    r = HQRenderer()
    try:
        r.submit("a", pyr, ViewTransform(150, 100, 1.0), (300, 200), (0, 0, 0))
        r.submit("b", pyr, ViewTransform(150, 100, 2.0), (300, 200), (0, 0, 0))
        results = []
        end = time.monotonic() + 5
        while not results and time.monotonic() < end:
            results = r.poll()
            time.sleep(0.01)
        assert results and all(x.key == "b" for x in results)
        assert results[-1].surface.get_size() == (300, 200)
        r.submit("c", pyr, ViewTransform(150, 100, 1.0), (300, 200), (0, 0, 0))
        r.cancel()
        time.sleep(0.2)
        assert r.poll() == []
    finally:
        r.shutdown()
