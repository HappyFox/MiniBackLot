"""Renderer places pixels where the transform says (headless SDL)."""

import pytest
from PIL import Image

from minibacklot.display.renderer import Renderer
from minibacklot.display.transforms import ViewTransform
from minibacklot.imaging.loader import ImagePyramid


def quadrant_image(w=200, h=100):
    """Red left half, blue right half; green bottom-right quarter."""
    img = Image.new("RGB", (w, h), (255, 0, 0))
    img.paste((0, 0, 255), (w // 2, 0, w, h))
    img.paste((0, 255, 0), (w // 2, h // 2, w, h))
    return ImagePyramid.from_pil(img, min_side=16)


def color_at(surface, pos):
    return tuple(surface.get_at(pos))[:3]


@pytest.mark.parametrize("smooth", [True, False])
def test_unrotated_render(pg_display, smooth):
    import pygame

    target = pygame.Surface((300, 200))
    r = Renderer(background=(9, 9, 9))
    t = ViewTransform(cx=150, cy=100, scale=1.0)
    r.render(target, quadrant_image(), t, smooth)
    assert color_at(target, (100, 70)) == (255, 0, 0)
    assert color_at(target, (200, 70)) == (0, 0, 255)
    assert color_at(target, (200, 130)) == (0, 255, 0)
    assert color_at(target, (10, 10)) == (9, 9, 9)  # background outside image


def test_flip_h(pg_display):
    import pygame

    target = pygame.Surface((300, 200))
    t = ViewTransform(cx=150, cy=100, scale=1.0, flip_h=True)
    Renderer().render(target, quadrant_image(), t)
    assert color_at(target, (100, 70)) == (0, 0, 255)
    assert color_at(target, (200, 70)) == (255, 0, 0)


@pytest.mark.parametrize("smooth", [True, False])
def test_rotated_45_has_background_corners_and_correct_colors(pg_display, smooth):
    import pygame

    target = pygame.Surface((400, 400))
    pyr = quadrant_image()
    t = ViewTransform(cx=200, cy=200, scale=1.0, rotation=45)
    Renderer(background=(9, 9, 9)).render(target, pyr, t, smooth)
    assert color_at(target, (2, 2)) == (9, 9, 9)  # no top-left-pixel padding colour
    for img_pt, expected in [((50, 25), (255, 0, 0)), ((150, 25), (0, 0, 255)), ((150, 75), (0, 255, 0))]:
        sx, sy = t.image_to_screen(img_pt, pyr.base_size)
        assert color_at(target, (round(sx), round(sy))) == expected


def test_zoomed_in_uses_crop_and_maps_correctly(pg_display):
    import pygame

    target = pygame.Surface((300, 200))
    pyr = quadrant_image()
    t = ViewTransform(cx=150, cy=100, scale=1.0).zoomed_about((150, 100), 8.0)
    Renderer().render(target, pyr, t)
    # Center of the screen is the image center: left of it red, right blue.
    assert color_at(target, (140, 90)) == (255, 0, 0)
    assert color_at(target, (160, 90)) == (0, 0, 255)


def test_zoomed_out_uses_pyramid_level(pg_display):
    import pygame

    target = pygame.Surface((300, 200))
    pyr = quadrant_image(800, 400)
    t = ViewTransform(cx=150, cy=100, scale=0.25)
    Renderer().render(target, pyr, t)
    assert color_at(target, (125, 90)) == (255, 0, 0)
    assert color_at(target, (175, 110)) == (0, 255, 0)


def test_offscreen_image_draws_background(pg_display):
    import pygame

    target = pygame.Surface((100, 100))
    t = ViewTransform(cx=5000, cy=5000, scale=1.0)
    Renderer(background=(1, 2, 3)).render(target, quadrant_image(), t)
    assert color_at(target, (50, 50)) == (1, 2, 3)


def test_frame_cache_reuses_surface(pg_display):
    r = Renderer()
    pyr = quadrant_image()
    t = ViewTransform(cx=50, cy=50)
    f1 = r.frame((100, 100), pyr, t, True)
    key = r._frame_key
    f2 = r.frame((100, 100), pyr, t, True)
    assert f1 is f2 and r._frame_key == key
