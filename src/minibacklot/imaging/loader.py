"""Image discovery, decoding and background loading.

Decoding uses Pillow, which handles JPEG, PNG, TIFF and WebP, applies EXIF
orientation and lets us convert embedded ICC profiles to sRGB. OpenCV is used
as an optional fallback for files Pillow cannot decode well (e.g. 16-bit RGB
TIFFs).

Loaded images are turned into an :class:`ImagePyramid` of pygame surfaces
(full resolution plus successive half-size levels) on a worker thread, so the
renderer can draw zoomed-out views of 24+ MP images cheaply. Only
``pygame.image.frombytes`` is used off the main thread; it does not touch the
display.
"""

from __future__ import annotations

import io
import logging
import queue
import re
import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pygame
from PIL import Image, ImageCms, ImageOps

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})

# Allow large panoramas but keep Pillow's decompression-bomb protection.
Image.MAX_IMAGE_PIXELS = 300_000_000

# Smallest pyramid level (longest side) worth keeping.
PYRAMID_MIN_SIDE = 256


class ImageLoadError(Exception):
    pass


def _natural_key(path: Path) -> list[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", path.name)]


def list_images(directory: Path) -> list[Path]:
    """Supported images in ``directory`` (non-recursive), naturally sorted."""
    try:
        entries = [
            p
            for p in directory.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file() and not p.name.startswith(".")
        ]
    except OSError as exc:
        log.warning("Cannot list %s: %s", directory, exc)
        return []
    return sorted(entries, key=_natural_key)


# ----------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------
_SRGB_PROFILE = None


def _srgb_profile():
    global _SRGB_PROFILE
    if _SRGB_PROFILE is None:
        _SRGB_PROFILE = ImageCms.createProfile("sRGB")
    return _SRGB_PROFILE


def _to_srgb(img: Image.Image) -> Image.Image:
    """Convert using the embedded ICC profile, if any. Never raises."""
    icc = img.info.get("icc_profile")
    if not icc:
        return img
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        mode = "RGBA" if "A" in img.getbands() else "RGB"
        if img.mode not in ("RGB", "RGBA", "CMYK", "L", "LA"):
            return img
        if img.mode == "LA":
            img = img.convert("RGBA")
        out = ImageCms.profileToProfile(img, src, _srgb_profile(), outputMode=mode)
        return out if out is not None else img
    except Exception as exc:  # broken/unsupported profiles are common
        log.info("Ignoring ICC profile: %s", exc)
        return img


def _high_bit_to_8bit(img: Image.Image) -> Image.Image:
    """Scale 16/32-bit integer or float grayscale images to 8-bit."""
    arr = np.asarray(img)
    if arr.dtype.kind == "f":
        lo, hi = 0.0, 1.0 if arr.max() <= 1.0 else float(arr.max())
    else:
        lo, hi = 0.0, 65535.0 if arr.max() > 255 else 255.0
    out = np.clip((arr.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1e-6)) + 0.5, 0, 255)
    return Image.fromarray(out.astype(np.uint8), "L")


def _load_with_opencv(path: Path) -> Image.Image | None:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return None
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYCOLOR)
    if data is None:
        return None
    if data.dtype == np.uint16:
        data = (data / 257.0 + 0.5).astype(np.uint8)
    elif data.dtype.kind == "f":
        data = (np.clip(data, 0, 1) * 255 + 0.5).astype(np.uint8)
    if data.ndim == 2:
        return Image.fromarray(data, "L")
    if data.shape[2] == 4:
        return Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGRA2RGBA), "RGBA")
    return Image.fromarray(cv2.cvtColor(data, cv2.COLOR_BGR2RGB), "RGB")


def decode_image(path: Path, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Decode ``path`` to an 8-bit sRGB ``RGB`` Pillow image.

    EXIF orientation is applied, ICC profiles are converted to sRGB, and
    transparency is flattened onto ``background``. The file is opened
    read-only and never modified.
    """
    try:
        with Image.open(path) as src:
            src.load()
            img = src
            img = ImageOps.exif_transpose(img) or img
            if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                img = _high_bit_to_8bit(img)
            else:
                img = _to_srgb(img)
    except Image.DecompressionBombError as exc:
        raise ImageLoadError(f"image too large: {exc}") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        fallback = _load_with_opencv(path)
        if fallback is None:
            raise ImageLoadError(str(exc) or exc.__class__.__name__) from exc
        img = fallback

    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, background)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        img = flat
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if img.width < 1 or img.height < 1:
        raise ImageLoadError("image has no pixels")
    return img


# ----------------------------------------------------------------------
# Pyramid
# ----------------------------------------------------------------------
def surface_from_array(arr: np.ndarray) -> pygame.Surface:
    """Wrap an (H, W, 3) uint8 C-contiguous array without copying.

    The surface shares memory with ``arr``; keep the array alive as long as
    the surface is used. Safe to call off the main thread.
    """
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return pygame.image.frombuffer(arr, (arr.shape[1], arr.shape[0]), "RGB")


def level_index_for_scale(widths: list[int], scale: float) -> int:
    """Smallest level whose resolution is >= the displayed resolution."""
    best = 0
    for i, w in enumerate(widths):
        if w / widths[0] >= scale:
            best = i
        else:
            break
    return best


@dataclass
class ImagePyramid:
    """Full-resolution image plus successive half-resolution levels.

    ``arrays`` hold the pixels (H, W, 3 uint8 sRGB); ``levels`` are pygame
    surfaces sharing the same memory. Source pixels are treated as
    read-only: processing always writes new arrays.
    """

    levels: list[pygame.Surface]
    arrays: list[np.ndarray] = field(default_factory=list)
    version: int = 0  # renderer cache key; source pyramids never change

    @property
    def base_size(self) -> tuple[int, int]:
        return self.levels[0].get_size()

    @property
    def widths(self) -> list[int]:
        return [s.get_width() for s in self.levels]

    def index_for_scale(self, scale: float) -> int:
        return level_index_for_scale(self.widths, scale)

    def level_for_scale(self, scale: float) -> tuple[int, pygame.Surface]:
        i = self.index_for_scale(scale)
        return i, self.levels[i]

    @classmethod
    def from_pil(cls, img: Image.Image, min_side: int = PYRAMID_MIN_SIDE) -> ImagePyramid:
        arrays = [np.array(img, dtype=np.uint8)]
        cur = img
        while max(cur.size) // 2 >= min_side:
            # Lanczos, not reduce(2): a 2x2 box average lets detail finer than
            # the new level's Nyquist limit alias into moire, which no later
            # resampling can remove. Costs ~0.25 s per 24 MP on the loader
            # thread, once per image.
            size = (max(1, (cur.width + 1) // 2), max(1, (cur.height + 1) // 2))
            cur = cur.resize(size, Image.Resampling.LANCZOS)
            arrays.append(np.array(cur, dtype=np.uint8))
        for a in arrays:
            a.flags.writeable = False
        return cls([surface_from_array(a) for a in arrays], arrays)


class ProcessedPyramid:
    """Processed versions of a source pyramid's levels for one adjustment set.

    Levels arrive one at a time from the processing worker. Until all are
    present, :meth:`level_for_scale` falls back to the nearest available
    level (preferring coarser ones, which are cheap to draw), so the view
    always shows the *current* adjustments, just possibly softer for a
    moment.
    """

    def __init__(self, source: ImagePyramid, key: object) -> None:
        self.source = source
        self.key = key
        self._levels: dict[int, pygame.Surface] = {}
        self._arrays: dict[int, np.ndarray] = {}
        self.version = 0

    @property
    def base_size(self) -> tuple[int, int]:
        return self.source.base_size

    @property
    def has_any(self) -> bool:
        return bool(self._levels)

    @property
    def complete(self) -> bool:
        return len(self._levels) == len(self.source.levels)

    @property
    def arrays_by_level(self) -> dict[int, np.ndarray]:
        return dict(self._arrays)

    def missing(self) -> list[int]:
        return [i for i in range(len(self.source.levels)) if i not in self._levels]

    def add(self, index: int, array: np.ndarray, surface: pygame.Surface | None = None) -> None:
        self._arrays[index] = array
        self._levels[index] = surface if surface is not None else surface_from_array(array)
        self.version += 1

    def level_for_scale(self, scale: float) -> tuple[int, pygame.Surface]:
        if not self._levels:
            raise LookupError("no processed levels yet")
        ideal = self.source.index_for_scale(scale)
        if ideal in self._levels:
            return ideal, self._levels[ideal]
        coarser = [i for i in self._levels if i > ideal]
        i = min(coarser) if coarser else max(i for i in self._levels if i < ideal)
        return i, self._levels[i]


@dataclass
class LoadedImage:
    path: Path
    pyramid: ImagePyramid
    file_size: int = 0

    @property
    def size(self) -> tuple[int, int]:
        return self.pyramid.base_size


def load_image(path: Path, background: tuple[int, int, int] = (0, 0, 0)) -> LoadedImage:
    img = decode_image(path, background)
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0
    return LoadedImage(path=path, pyramid=ImagePyramid.from_pil(img), file_size=file_size)


# ----------------------------------------------------------------------
# Asynchronous loader with LRU cache
# ----------------------------------------------------------------------
@dataclass
class LoadResult:
    path: Path
    image: LoadedImage | None = None
    error: str | None = None


@dataclass
class AsyncImageLoader:
    """Loads images on a worker thread and caches recent results.

    Results are delivered through :meth:`poll` on the main thread. If
    ``wake_event_type`` is given, a pygame event of that type is posted when a
    result is ready so a blocking ``pygame.event.wait()`` wakes up.
    """

    background: tuple[int, int, int] = (0, 0, 0)
    cache_size: int = 3
    wake_event_type: int | None = None
    _executor: ThreadPoolExecutor = field(init=False)
    _results: "queue.Queue[LoadResult]" = field(init=False)
    _cache: "OrderedDict[Path, LoadedImage]" = field(init=False)
    _inflight: dict[Path, Future] = field(init=False)
    _lock: threading.Lock = field(init=False)

    def __post_init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minibacklot-loader")
        self._results = queue.Queue()
        self._cache = OrderedDict()
        self._inflight = {}
        self._lock = threading.Lock()

    def cached(self, path: Path) -> LoadedImage | None:
        img = self._cache.get(path)
        if img is not None:
            self._cache.move_to_end(path)
        return img

    def request(self, path: Path, *, prefetch: bool = False) -> LoadedImage | None:
        """Return the cached image, or start loading it and return None.

        Prefetch requests are not reported through :meth:`poll` unless a
        normal request for the same path is made later.
        """
        img = self.cached(path)
        if img is not None:
            return img
        with self._lock:
            if path in self._inflight:
                if not prefetch:
                    self._inflight[path].wanted = True  # type: ignore[attr-defined]
                return None
            fut = self._executor.submit(self._work, path)
            fut.wanted = not prefetch  # type: ignore[attr-defined]
            self._inflight[path] = fut
        return None

    def _work(self, path: Path) -> LoadResult:
        try:
            result = LoadResult(path, image=load_image(path, self.background))
        except ImageLoadError as exc:
            result = LoadResult(path, error=str(exc))
        except Exception as exc:  # never let the worker die silently
            log.exception("Unexpected error loading %s", path)
            result = LoadResult(path, error=f"{exc.__class__.__name__}: {exc}")
        self._results.put(result)
        if self.wake_event_type is not None:
            try:
                pygame.event.post(pygame.event.Event(self.wake_event_type))
            except pygame.error:
                pass
        return result

    def poll(self) -> list[LoadResult]:
        """Collect finished loads (main thread). Returns wanted results only."""
        out: list[LoadResult] = []
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                fut = self._inflight.pop(res.path, None)
            if res.image is not None:
                self._cache[res.path] = res.image
                self._cache.move_to_end(res.path)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
            if fut is not None and getattr(fut, "wanted", True):
                out.append(res)
        return out

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(getattr(f, "wanted", False) for f in self._inflight.values())

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
