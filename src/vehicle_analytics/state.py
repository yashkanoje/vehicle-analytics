"""What the pipeline remembers about one tracked vehicle."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from vehicle_analytics.crossing import CrossingEvent, detect_crossing
from vehicle_analytics.geometry import Line, Point, Zone
from vehicle_analytics.speed import SpeedEstimate

APPEARANCE_SAMPLES = 5


@dataclass(frozen=True)
class VehicleRecord:
    """The finished result for one counted vehicle."""

    track_id: str
    cls_name: str
    direction: str | None
    counted_at_s: float | None
    speed_kmph: float | None
    speed_err_kmph: float | None
    speed_mode: str | None
    colour: str | None = None
    lane: str | None = None
    first_seen_s: float | None = None
    last_seen_s: float | None = None
    snapshot_path: str | None = None


@dataclass
class TrackState:
    track_id: str
    history: list[tuple[int, Point]] = field(default_factory=list)
    ground_history: list[tuple[int, Point]] = field(default_factory=list)
    class_votes: Counter[str] = field(default_factory=Counter)
    crossings: dict[str, CrossingEvent] = field(default_factory=dict)
    zone_entered: dict[str, int] = field(default_factory=dict)
    direction: str | None = None
    counted_at_s: float | None = None
    lane: str | None = None
    speed: SpeedEstimate | None = None
    colour_samples: list[tuple[float, str]] = field(default_factory=list)
    snapshot: np.ndarray | None = None
    snapshot_area: float = 0.0

    @property
    def first_frame(self) -> int:
        return self.history[0][0]

    @property
    def last_frame(self) -> int:
        return self.history[-1][0]

    @property
    def cls_name(self) -> str:
        return self.class_votes.most_common(1)[0][0]

    @property
    def counted(self) -> bool:
        return self.direction is not None

    @property
    def colour(self) -> str | None:
        """Majority colour over the largest views held."""
        if not self.colour_samples:
            return None
        return Counter(name for _, name in self.colour_samples).most_common(1)[0][0]

    def observe(
        self,
        frame_index: int,
        point: Point,
        cls_name: str,
        lines: Iterable[Line],
        to_ground: Callable[[Point], Point] | None = None,
        zones: Iterable[Zone] = (),
        visible: bool = True,
    ) -> list[CrossingEvent]:
        """Add one observation and return the lines newly crossed by it.

        `visible` is False for a box cut by the frame edge. Its reference point is
        pinned to that edge, so it is kept out of the ground history and cannot give
        a live speed.
        """
        new_events: list[CrossingEvent] = []
        if self.history:
            k_prev, p_prev = self.history[-1]
            for line in lines:
                if line.name in self.crossings:
                    continue
                event = detect_crossing(
                    line, self.track_id, p_prev, point, k_prev, frame_index, to_ground
                )
                if event is not None:
                    self.crossings[line.name] = event
                    new_events.append(event)
        self.history.append((frame_index, point))
        self.class_votes[cls_name] += 1
        if to_ground is not None and visible:
            self.ground_history.append((frame_index, to_ground(point)))
        for zone in zones:
            if zone.contains(point):
                self.zone_entered.setdefault(zone.name, frame_index)
            else:
                self.zone_entered.pop(zone.name, None)
        return new_events

    def live_speed_kmph(self, fps: float, window_s: float) -> float | None:
        """Ground speed over the last `window_s` seconds, or None if it cannot be measured."""
        if len(self.ground_history) < 2:
            return None
        k_now, ground_now = self.ground_history[-1]
        if k_now != self.last_frame:
            return None  # not fully visible now: the last ground point is out of date
        k_then, ground_then = k_now, ground_now
        for k, ground in reversed(self.ground_history[:-1]):
            if k_now - k > window_s * fps:
                break
            k_then, ground_then = k, ground
        if k_then == k_now:
            return None
        return math.dist(ground_now, ground_then) / ((k_now - k_then) / fps) * 3.6

    def wants_appearance(self, area: float) -> bool:
        """Whether a view of this size would be among the largest held."""
        full = len(self.colour_samples) >= APPEARANCE_SAMPLES
        return not full or area > self.colour_samples[-1][0]

    def add_appearance(self, area: float, colour: str, crop: np.ndarray) -> None:
        self.colour_samples.append((area, colour))
        self.colour_samples.sort(reverse=True)
        del self.colour_samples[APPEARANCE_SAMPLES:]
        if area > self.snapshot_area:
            self.snapshot = crop
            self.snapshot_area = area

    def mark_counted(self, direction: str, at_s: float, lane: str | None = None) -> None:
        self.direction = direction
        self.counted_at_s = at_s
        self.lane = lane

    def to_record(
        self,
        fps: float | None = None,
        snapshot_path: str | None = None,
        include_uncounted: bool = False,
    ) -> VehicleRecord | None:
        """The vehicle's record, or None if it was not counted and `include_uncounted` is off."""
        if not self.counted and not include_uncounted:
            return None
        speed = self.speed
        return VehicleRecord(
            track_id=self.track_id,
            cls_name=self.cls_name,
            direction=self.direction,
            counted_at_s=self.counted_at_s,
            speed_kmph=speed.kmph if speed else None,
            speed_err_kmph=speed.err_kmph if speed else None,
            speed_mode=speed.mode if speed else None,
            colour=self.colour,
            lane=self.lane,
            first_seen_s=self.first_frame / fps if fps else None,
            last_seen_s=self.last_frame / fps if fps else None,
            snapshot_path=snapshot_path,
        )
