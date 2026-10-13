"""What the pipeline remembers about one tracked vehicle."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from vehicle_analytics.crossing import CrossingEvent, detect_crossing
from vehicle_analytics.geometry import Line, Point
from vehicle_analytics.speed import SpeedEstimate


@dataclass(frozen=True)
class VehicleRecord:
    """The finished result for one counted vehicle."""

    track_id: str
    cls_name: str
    direction: str
    counted_at_s: float
    speed_kmph: float | None
    speed_err_kmph: float | None
    speed_mode: str | None


@dataclass
class TrackState:
    track_id: str
    history: list[tuple[int, Point]] = field(default_factory=list)
    class_votes: Counter[str] = field(default_factory=Counter)
    crossings: dict[str, CrossingEvent] = field(default_factory=dict)
    direction: str | None = None
    counted_at_s: float | None = None
    speed: SpeedEstimate | None = None

    @property
    def last_frame(self) -> int:
        return self.history[-1][0]

    @property
    def cls_name(self) -> str:
        return self.class_votes.most_common(1)[0][0]

    @property
    def counted(self) -> bool:
        return self.direction is not None

    def observe(
        self, frame_index: int, point: Point, cls_name: str, lines: Iterable[Line]
    ) -> list[CrossingEvent]:
        """Add one observation and return the lines newly crossed by it."""
        new_events: list[CrossingEvent] = []
        if self.history:
            k_prev, p_prev = self.history[-1]
            for line in lines:
                if line.name in self.crossings:
                    continue
                event = detect_crossing(line, self.track_id, p_prev, point, k_prev, frame_index)
                if event is not None:
                    self.crossings[line.name] = event
                    new_events.append(event)
        self.history.append((frame_index, point))
        self.class_votes[cls_name] += 1
        return new_events

    def mark_counted(self, direction: str, at_s: float) -> None:
        self.direction = direction
        self.counted_at_s = at_s

    def to_record(self) -> VehicleRecord | None:
        if self.direction is None or self.counted_at_s is None:
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
        )
