"""Image decoding edge cases, pyramid and async loader."""

import time

import numpy as np
import pygame
import pytest
from PIL import Image, ImageCms

from minibacklot.imaging.loader import (
    AsyncImageLoader,
    ImageLoadError,
    ImagePyramid,
    decode_image,
    list_images,
    load_image,
)


def save_rgb(path, size=(40, 20), color=(200, 100, 50), **kw):
    Image.new("RGB", size, color).save(path, **kw)
    return path


def test_list_images_filters_and_sorts_naturally(tmp_path):
    for name in ["img10.jpg", "img2.JPG", "img1.png", "notes.txt", ".hidden.jpg", "a.webp", "b.tiff"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "dir.jpg").mkdir()
    names = [p.name for p in list_images(tmp_path)]
    assert names == ["a.webp", "b.tiff", "img1.png", "img2.JPG", "img10.jpg"]


def test_list_images_missing_dir(tmp_path):
    assert list_images(tmp_path / "missing") == []


@pytest.mark.parametrize("ext", ["jpg", "png", "tiff", "webp"])
def test_decode_formats(tmp_path, ext):
    p = save_rgb(tmp_path / f"x.{ext}")
    img = decode_image(p)
    assert img.mode == "RGB" and img.size == (40, 20)


def test_exif_orientation_is_applied(tmp_path):
    p = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    Image.new("RGB", (40, 20)).save(p, exif=exif)
    assert decode_image(p).size == (20, 40)


def test_transparency_is_flattened_onto_background(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 0)).save(p)
    img = decode_image(p, background=(10, 20, 30))
    assert img.getpixel((0, 0)) == (10, 20, 30)


def test_grayscale_and_16bit(tmp_path):
    p = tmp_path / "g.png"
    Image.new("L", (4, 4), 128).save(p)
    assert decode_image(p).getpixel((0, 0)) == (128, 128, 128)

    p16 = tmp_path / "g16.png"
    Image.fromarray(np.full((4, 4), 65535, dtype=np.uint16)).save(p16)
    assert decode_image(p16).getpixel((0, 0)) == (255, 255, 255)


def test_icc_profile_does_not_break_loading(tmp_path):
    p = tmp_path / "icc.jpg"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    save_rgb(p, icc_profile=icc)
    img = decode_image(p)
    r, g, b = img.getpixel((5, 5))
    assert abs(r - 200) < 6 and abs(g - 100) < 6 and abs(b - 50) < 6


def test_corrupt_file_raises_load_error(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"\xff\xd8 definitely not a jpeg")
    with pytest.raises(ImageLoadError):
        decode_image(p)


def test_missing_file_raises_load_error(tmp_path):
    with pytest.raises(ImageLoadError):
        decode_image(tmp_path / "gone.jpg")


def test_source_file_is_not_modified(tmp_path):
    p = save_rgb(tmp_path / "x.jpg")
    before = (p.read_bytes(), p.stat().st_mtime_ns)
    load_image(p)
    assert (p.read_bytes(), p.stat().st_mtime_ns) == before


def test_pyramid_levels_and_selection():
    pyr = ImagePyramid.from_pil(Image.new("RGB", (2000, 1000)), min_side=256)
    sizes = [s.get_size() for s in pyr.levels]
    assert sizes == [(2000, 1000), (1000, 500), (500, 250), (250, 125)][: len(sizes)]
    assert max(sizes[-1]) >= 256 or len(sizes) == 1
    assert pyr.level_for_scale(1.5)[0] == 0
    assert pyr.level_for_scale(0.6)[0] == 0
    assert pyr.level_for_scale(0.5)[0] == 1
    assert pyr.level_for_scale(0.3)[0] == 1
    assert pyr.level_for_scale(0.01)[0] == len(sizes) - 1


def test_one_pixel_image(tmp_path):
    p = save_rgb(tmp_path / "1.png", size=(1, 1))
    li = load_image(p)
    assert li.size == (1, 1) and len(li.pyramid.levels) == 1


def wait_results(loader, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        res = loader.poll()
        if res:
            return res
        time.sleep(0.01)
    raise AssertionError("loader timed out")


def test_async_loader_success_cache_and_errors(tmp_path):
    good = save_rgb(tmp_path / "good.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"garbage")
    loader = AsyncImageLoader(cache_size=2)
    try:
        assert loader.request(good) is None
        (res,) = wait_results(loader)
        assert res.path == good and res.image is not None and res.error is None
        assert loader.request(good) is res.image  # cached now

        assert loader.request(bad) is None
        (res,) = wait_results(loader)
        assert res.image is None and res.error
    finally:
        loader.shutdown()


def test_prefetch_is_cached_but_not_reported(tmp_path):
    p = save_rgb(tmp_path / "p.png")
    loader = AsyncImageLoader()
    try:
        loader.request(p, prefetch=True)
        end = time.monotonic() + 5
        while loader.cached(p) is None and time.monotonic() < end:
            assert loader.poll() == []
            time.sleep(0.01)
        assert loader.cached(p) is not None
    finally:
        loader.shutdown()


def test_async_loader_posts_wake_event(tmp_path):
    pygame.display.init()
    try:
        ev_type = pygame.event.custom_type()
        p = save_rgb(tmp_path / "w.png")
        loader = AsyncImageLoader(wake_event_type=ev_type)
        try:
            loader.request(p)
            wait_results(loader)
            assert any(e.type == ev_type for e in pygame.event.get())
        finally:
            loader.shutdown()
    finally:
        pygame.display.quit()
