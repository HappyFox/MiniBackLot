"""Centralized touch gesture recognizer.

The recognizer is a pure state machine: it is fed normalized pointer events
(``pointer_down`` / ``pointer_move`` / ``pointer_up``) with explicit
millisecond timestamps, plus periodic ``update(now)`` calls for timers. It
returns lists of :class:`Gesture` objects. It has no pygame dependency, so it
can be driven entirely by synthetic events in tests.

Gestures produced:

* ``GESTURE_START`` / ``GESTURE_END`` bracket every continuous manipulation
  (single-finger pan or two-finger pinch/rotate). Use them for undo
  snapshots and to switch between fast and high-quality rendering.
* ``PAN`` — translation delta in screen pixels (single finger, or the
  midpoint of two fingers).
* ``PINCH`` — incremental scale factor around ``pos`` (two-finger midpoint).
* ``ROTATE`` — incremental clockwise angle in degrees around ``pos``.
* ``TAP``, ``DOUBLE_TAP``, ``LONG_PRESS`` — discrete gestures at ``pos``.

Design notes:

* A single finger can only ever pan; zoom requires two fingers, so panning
  can never accidentally zoom.
* Pinch and rotation each have an activation threshold measured from the
  start of the two-finger gesture, so a pure pinch does not wobble the
  rotation and vice versa. Activation is continuous: the threshold amount is
  absorbed rather than applied as a jump.
* A single ``TAP`` is delayed until the double-tap window expires, so that a
  double-tap does not also produce two single taps.
* When one finger of a pinch lifts, the remaining finger continues panning
  from its current position without a jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Hashable

Vec = tuple[float, float]
PointerId = Hashable


class GestureType(Enum):
    GESTURE_START = auto()
    GESTURE_END = auto()
    PAN = auto()
    PINCH = auto()
    ROTATE = auto()
    TAP = auto()
    DOUBLE_TAP = auto()
    LONG_PRESS = auto()


@dataclass(frozen=True)
class Gesture:
    type: GestureType
    pos: Vec = (0.0, 0.0)
    delta: Vec = (0.0, 0.0)
    scale: float = 1.0
    angle: float = 0.0  # degrees, clockwise on screen


@dataclass
class GestureConfig:
    tap_slop: float = 12.0  # px a finger may move and still count as a tap
    tap_max_ms: int = 300  # max press duration for a tap
    double_tap_ms: int = 350  # max gap between first release and second press
    double_tap_slop: float = 48.0  # max distance between the two taps
    long_press_ms: int = 600
    pinch_threshold: float = 0.03  # relative scale change before pinch engages
    rotation_threshold_deg: float = 8.0  # angle before rotation engages
    rotation_enabled: bool = True


class _Phase(Enum):
    IDLE = auto()
    PRESSING = auto()  # one finger down, not yet moved beyond slop
    PANNING = auto()  # one finger dragging
    MULTI = auto()  # two or more fingers


@dataclass
class _Contact:
    start: Vec
    pos: Vec
    t_down: int


@dataclass
class _PendingTap:
    pos: Vec
    t_up: int


def _dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a: Vec, b: Vec) -> Vec:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _angle(a: Vec, b: Vec) -> float:
    # y-down screen coordinates: increasing atan2 is clockwise on screen.
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class GestureRecognizer:
    config: GestureConfig = field(default_factory=GestureConfig)

    def __post_init__(self) -> None:
        self._contacts: dict[PointerId, _Contact] = {}
        self._order: list[PointerId] = []  # pointer ids in press order
        self._phase = _Phase.IDLE
        self._started = False  # GESTURE_START emitted, GESTURE_END pending
        self._long_press_fired = False
        self._used_multi = False  # touch sequence involved 2+ fingers
        self._pending_tap: _PendingTap | None = None
        self._double_candidate = False
        # Two-finger tracking
        self._pair: tuple[PointerId, PointerId] | None = None
        self._cum_log_scale = 0.0
        self._cum_angle = 0.0
        self._pinch_active = False
        self._rotate_active = False

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """True while any finger is down."""
        return bool(self._contacts)

    @property
    def manipulating(self) -> bool:
        """True between GESTURE_START and GESTURE_END."""
        return self._started

    @property
    def needs_timer(self) -> bool:
        """True if ``update()`` must be called soon (long-press / tap timers)."""
        return self._phase == _Phase.PRESSING or self._pending_tap is not None

    @property
    def contact_count(self) -> int:
        return len(self._contacts)

    # ------------------------------------------------------------------
    # Event input
    # ------------------------------------------------------------------
    def pointer_down(self, pid: PointerId, pos: Vec, t: int) -> list[Gesture]:
        out: list[Gesture] = []
        if pid in self._contacts:  # duplicate down: treat as move
            return self.pointer_move(pid, pos, t)
        self._contacts[pid] = _Contact(start=pos, pos=pos, t_down=t)
        self._order.append(pid)
        n = len(self._contacts)

        if n == 1:
            self._used_multi = False
            self._long_press_fired = False
            self._phase = _Phase.PRESSING
            self._double_candidate = False
            pending = self._pending_tap
            if pending is not None:
                if (
                    t - pending.t_up <= self.config.double_tap_ms
                    and _dist(pos, pending.pos) <= self.config.double_tap_slop
                ):
                    self._double_candidate = True
                else:
                    out += self._flush_pending_tap()
        else:
            out += self._flush_pending_tap()
            self._double_candidate = False
            self._used_multi = True
            if not self._started:
                self._started = True
                out.append(Gesture(GestureType.GESTURE_START, pos=pos))
            self._phase = _Phase.MULTI
            self._select_pair(reset_thresholds=True)
        return out

    def pointer_move(self, pid: PointerId, pos: Vec, t: int) -> list[Gesture]:
        contact = self._contacts.get(pid)
        if contact is None:
            return []
        out: list[Gesture] = []
        prev = contact.pos
        if self._phase == _Phase.PRESSING:
            contact.pos = pos
            if _dist(pos, contact.start) > self.config.tap_slop:
                out += self._flush_pending_tap()
                self._double_candidate = False
                self._phase = _Phase.PANNING
                if not self._started:
                    self._started = True
                    out.append(Gesture(GestureType.GESTURE_START, pos=contact.start))
                # Include the slop distance so the image stays under the finger.
                d = (pos[0] - contact.start[0], pos[1] - contact.start[1])
                out.append(Gesture(GestureType.PAN, pos=pos, delta=d))
        elif self._phase == _Phase.PANNING:
            contact.pos = pos
            d = (pos[0] - prev[0], pos[1] - prev[1])
            if d != (0.0, 0.0):
                out.append(Gesture(GestureType.PAN, pos=pos, delta=d))
        elif self._phase == _Phase.MULTI:
            if self._pair is None or pid not in self._pair:
                contact.pos = pos  # extra finger: track but ignore
                return out
            a_id, b_id = self._pair
            a0, b0 = self._contacts[a_id].pos, self._contacts[b_id].pos
            contact.pos = pos
            a1, b1 = self._contacts[a_id].pos, self._contacts[b_id].pos
            out += self._two_finger_update(a0, b0, a1, b1)
        return out

    def pointer_up(self, pid: PointerId, pos: Vec, t: int) -> list[Gesture]:
        contact = self._contacts.pop(pid, None)
        if contact is None:
            return []
        self._order.remove(pid)
        out: list[Gesture] = []
        n = len(self._contacts)

        if self._phase == _Phase.PRESSING:
            # Released without moving beyond slop.
            if not self._long_press_fired and not self._used_multi:
                if t - contact.t_down <= self.config.tap_max_ms:
                    if self._double_candidate and self._pending_tap is not None:
                        self._pending_tap = None
                        out.append(Gesture(GestureType.DOUBLE_TAP, pos=contact.start))
                    else:
                        out += self._flush_pending_tap()
                        self._pending_tap = _PendingTap(pos=contact.start, t_up=t)
                else:
                    out += self._flush_pending_tap()
            self._double_candidate = False
            self._phase = _Phase.IDLE
        elif self._phase == _Phase.PANNING:
            if n == 0:
                out += self._end_manipulation(pos)
        elif self._phase == _Phase.MULTI:
            if n == 0:
                out += self._end_manipulation(pos)
            elif n == 1:
                # Continue panning with the remaining finger, no jump: its
                # current position becomes the new reference.
                self._pair = None
                self._phase = _Phase.PANNING
            else:
                self._select_pair(reset_thresholds=False)
        return out

    def pointer_cancel(self, pid: PointerId) -> list[Gesture]:
        """Forget a single contact without producing a tap."""
        if pid not in self._contacts:
            return []
        contact = self._contacts[pid]
        self._long_press_fired = True  # suppress tap on the synthetic release
        return self.pointer_up(pid, contact.pos, contact.t_down)

    def update(self, t: int) -> list[Gesture]:
        """Drive timers. Call regularly while ``needs_timer`` is True."""
        out: list[Gesture] = []
        if (
            self._phase == _Phase.PRESSING
            and not self._long_press_fired
            and not self._used_multi
            and len(self._contacts) == 1
        ):
            (contact,) = self._contacts.values()
            if t - contact.t_down >= self.config.long_press_ms:
                out += self._flush_pending_tap()
                self._double_candidate = False
                self._long_press_fired = True
                out.append(Gesture(GestureType.LONG_PRESS, pos=contact.start))
        pending = self._pending_tap
        if (
            pending is not None
            and not self._double_candidate
            and t - pending.t_up > self.config.double_tap_ms
        ):
            out += self._flush_pending_tap()
        return out

    def cancel(self) -> list[Gesture]:
        """Abort everything (focus loss, lock, UI capture). No taps emitted."""
        out: list[Gesture] = []
        if self._started:
            out.append(Gesture(GestureType.GESTURE_END))
        self.__post_init__()
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _flush_pending_tap(self) -> list[Gesture]:
        if self._pending_tap is None:
            return []
        pos = self._pending_tap.pos
        self._pending_tap = None
        return [Gesture(GestureType.TAP, pos=pos)]

    def _end_manipulation(self, pos: Vec) -> list[Gesture]:
        self._phase = _Phase.IDLE
        self._pair = None
        if self._started:
            self._started = False
            return [Gesture(GestureType.GESTURE_END, pos=pos)]
        return []

    def _select_pair(self, reset_thresholds: bool) -> None:
        self._pair = (self._order[0], self._order[1])
        if reset_thresholds:
            self._cum_log_scale = 0.0
            self._cum_angle = 0.0
            self._pinch_active = False
            self._rotate_active = False

    def _two_finger_update(self, a0: Vec, b0: Vec, a1: Vec, b1: Vec) -> list[Gesture]:
        out: list[Gesture] = []
        m0, m1 = _mid(a0, b0), _mid(a1, b1)
        d = (m1[0] - m0[0], m1[1] - m0[1])
        if d != (0.0, 0.0):
            out.append(Gesture(GestureType.PAN, pos=m1, delta=d))

        dist0, dist1 = _dist(a0, b0), _dist(a1, b1)
        if dist0 > 1e-3 and dist1 > 1e-3:
            step = math.log(dist1 / dist0)
            self._cum_log_scale += step
            threshold = math.log1p(self.config.pinch_threshold)
            if self._pinch_active:
                apply = step
            elif abs(self._cum_log_scale) > threshold:
                self._pinch_active = True
                apply = self._cum_log_scale - math.copysign(threshold, self._cum_log_scale)
            else:
                apply = 0.0
            if apply != 0.0:
                out.append(Gesture(GestureType.PINCH, pos=m1, scale=math.exp(apply)))

            if self.config.rotation_enabled:
                step_a = _wrap(_angle(a1, b1) - _angle(a0, b0))
                self._cum_angle += step_a
                thr = self.config.rotation_threshold_deg
                if self._rotate_active:
                    apply_a = step_a
                elif abs(self._cum_angle) > thr:
                    self._rotate_active = True
                    apply_a = self._cum_angle - math.copysign(thr, self._cum_angle)
                else:
                    apply_a = 0.0
                if apply_a != 0.0:
                    out.append(Gesture(GestureType.ROTATE, pos=m1, angle=apply_a))
        return out
