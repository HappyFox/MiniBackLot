"""Background processing of pyramid levels with latest-request-wins semantics.

The main thread calls :meth:`ImageProcessor.submit` whenever the image or
the adjustments change. A single worker thread processes the most recent
request, level by level, starting with the level the renderer needs right
now, so the visible preview updates first and the other levels follow.

Superseded requests are abandoned between strips, but the *first* level of
a request always runs to completion. This keeps the preview moving during a
continuous slider drag, instead of restarting forever.

NumPy releases the GIL for the heavy operations, so the UI thread stays
responsive.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np
import pygame

from .loader import ImagePyramid, surface_from_array
from .pipeline import Adjustments, Cancelled, ProcessingPlan

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessResult:
    generation: int
    key: object
    level: int
    array: np.ndarray
    surface: pygame.Surface


@dataclass
class _Request:
    generation: int
    key: object
    pyramid: ImagePyramid
    adjustments: Adjustments
    order: list[int]


def level_order(n_levels: int, first: int, skip: set[int] | frozenset[int] = frozenset()) -> list[int]:
    """``first``, then coarser levels (cheap), then finer ones up to full res."""
    first = max(0, min(first, n_levels - 1))
    order = [first] + list(range(first + 1, n_levels)) + list(range(first - 1, -1, -1))
    return [i for i in order if i not in skip]


class ImageProcessor:
    def __init__(self, wake_event_type: int | None = None) -> None:
        self.wake_event_type = wake_event_type
        self._cond = threading.Condition()
        self._request: _Request | None = None
        self._generation = 0
        self._busy = False
        self._stop = False
        self._results: queue.Queue[ProcessResult] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="minibacklot-processor", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Main-thread API
    # ------------------------------------------------------------------
    def submit(
        self,
        key: object,
        pyramid: ImagePyramid,
        adjustments: Adjustments,
        first_level: int = 0,
        skip: set[int] | frozenset[int] = frozenset(),
    ) -> int:
        """Queue processing; returns the request's generation number."""
        order = level_order(len(pyramid.levels), first_level, skip)
        with self._cond:
            self._generation += 1
            self._request = _Request(self._generation, key, pyramid, adjustments, order)
            self._cond.notify()
            return self._generation

    def cancel(self) -> None:
        with self._cond:
            self._generation += 1
            self._request = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def busy(self) -> bool:
        with self._cond:
            return self._busy or self._request is not None

    def poll(self) -> list[ProcessResult]:
        """Finished levels of the *current* generation (main thread)."""
        out = []
        while True:
            try:
                res = self._results.get_nowait()
            except queue.Empty:
                return out
            if res.generation == self._generation:
                out.append(res)

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop = True
            self._request = None
            self._cond.notify()
        self._thread.join(timeout)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._cond:
                while self._request is None and not self._stop:
                    self._busy = False
                    self._cond.wait()
                if self._stop:
                    return
                req, self._request = self._request, None
                self._busy = True
            try:
                self._process(req)
            except Exception:
                log.exception("Processing failed")

    def _superseded(self, req: _Request) -> bool:
        return self._generation != req.generation or self._stop

    def _process(self, req: _Request) -> None:
        plan = ProcessingPlan(req.adjustments)
        for n, level in enumerate(req.order):
            if n > 0 and self._superseded(req):
                return
            check = None if n == 0 else (lambda: self._superseded(req))
            try:
                arr = plan.apply(req.pyramid.arrays[level], should_cancel=check)
            except Cancelled:
                return
            arr.flags.writeable = False
            self._results.put(ProcessResult(req.generation, req.key, level, arr, surface_from_array(arr)))
            if self.wake_event_type is not None:
                try:
                    pygame.event.post(pygame.event.Event(self.wake_event_type))
                except pygame.error:
                    pass
