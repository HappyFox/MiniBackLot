"""Background processing worker and processed pyramids."""

import time

import numpy as np
import pytest
from PIL import Image

from minibacklot.imaging.loader import ImagePyramid, ProcessedPyramid
from minibacklot.imaging.pipeline import Adjustments, process_array
from minibacklot.imaging.processor import ImageProcessor, level_order


def make_pyramid(w=1024, h=512):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return ImagePyramid.from_pil(Image.fromarray(arr), min_side=64)


def collect(proc, n, timeout=10.0):
    out = []
    end = time.monotonic() + timeout
    while len(out) < n and time.monotonic() < end:
        out += proc.poll()
        time.sleep(0.005)
    return out


def test_level_order():
    assert level_order(5, 2) == [2, 3, 4, 1, 0]
    assert level_order(3, 0) == [0, 1, 2]
    assert level_order(3, 9) == [2, 1, 0]
    assert level_order(4, 1, skip={0, 3}) == [1, 2]


def test_processes_all_levels_needed_first():
    pyr = make_pyramid()
    adj = Adjustments(exposure=0.5, saturation=20)
    proc = ImageProcessor()
    try:
        gen = proc.submit("k", pyr, adj, first_level=2)
        results = collect(proc, len(pyr.levels))
        assert [r.level for r in results] == level_order(len(pyr.levels), 2)
        assert all(r.generation == gen and r.key == "k" for r in results)
        for r in results:
            assert np.array_equal(r.array, process_array(pyr.arrays[r.level], adj))
            assert r.surface.get_size() == pyr.levels[r.level].get_size()
    finally:
        proc.shutdown()


def test_superseded_requests_are_dropped():
    pyr = make_pyramid()
    proc = ImageProcessor()
    try:
        proc.submit("old", pyr, Adjustments(exposure=1))
        gen = proc.submit("new", pyr, Adjustments(exposure=-1))
        results = collect(proc, len(pyr.levels))
        time.sleep(0.05)
        results += proc.poll()
        assert results and all(r.key == "new" and r.generation == gen for r in results)
        assert len(results) == len(pyr.levels)
    finally:
        proc.shutdown()


def test_cancel_discards_results():
    pyr = make_pyramid()
    proc = ImageProcessor()
    try:
        proc.submit("k", pyr, Adjustments(exposure=1))
        proc.cancel()
        time.sleep(0.3)
        assert proc.poll() == []
        assert not proc.busy
    finally:
        proc.shutdown()


def test_processed_pyramid_fallback_prefers_coarser():
    pyr = make_pyramid()  # levels: 1024, 512, 256, 128, 64
    pp = ProcessedPyramid(pyr, key="k")
    assert not pp.has_any
    with pytest.raises(LookupError):
        pp.level_for_scale(1.0)
    pp.add(3, pyr.arrays[3])
    assert pp.level_for_scale(1.0)[0] == 3  # only one available
    pp.add(1, pyr.arrays[1])
    assert pp.level_for_scale(0.25)[0] == 3  # ideal 2 missing -> coarser 3
    assert pp.level_for_scale(0.5)[0] == 1  # exact
    assert pp.level_for_scale(1.0)[0] == 1  # ideal 0 missing -> nearest coarser
    v = pp.version
    pp.add(0, pyr.arrays[0])
    assert pp.version == v + 1
    assert pp.level_for_scale(1.0)[0] == 0
    assert not pp.complete
    for i in pp.missing():
        pp.add(i, pyr.arrays[i])
    assert pp.complete
