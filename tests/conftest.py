import os

# Headless SDL so tests run without a display or touch hardware.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def pg_display():
    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((320, 240))
    yield screen
    pygame.display.quit()
