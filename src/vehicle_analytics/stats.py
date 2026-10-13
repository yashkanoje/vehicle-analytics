"""Running counts, flow rate and speed aggregates for one run."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict

from vehicle_analytics.state import VehicleRecord


class Stats:
    def __init__(self) -> None:
        self.counts_by_direction: Counter[str] = Counter()
        self.counts_by_class: Counter[str] = Counter()
        self._speeds: defaultdict[str, list[float]] = defaultdict(list)

    @property
    def total(self) -> int:
        return sum(self.counts_by_direction.values())

    def record_count(self, direction: str) -> None:
        """A vehicle crossed the count line."""
        self.counts_by_direction[direction] += 1

    def record_vehicle(self, record: VehicleRecord) -> None:
        """A counted vehicle is finished: its class and speed are now final."""
        self.counts_by_class[record.cls_name] += 1
        if record.speed_kmph is not None:
            self._speeds[record.direction].append(record.speed_kmph)

    def flow_per_hour(self, elapsed_s: float) -> dict[str, float]:
        """Vehicles per hour by direction, over the video time elapsed so far."""
        if elapsed_s <= 0.0:
            return {direction: 0.0 for direction in self.counts_by_direction}
        return {
            direction: count / elapsed_s * 3600.0
            for direction, count in self.counts_by_direction.items()
        }

    def speed_summary(self) -> dict[str, dict[str, float]]:
        return {
            direction: {
                "n": len(speeds),
                "mean": round(statistics.fmean(speeds), 2),
                "median": round(statistics.median(speeds), 2),
                "max": round(max(speeds), 2),
            }
            for direction, speeds in self._speeds.items()
        }

    def summary(self, elapsed_s: float) -> dict:
        return {
            "total": self.total,
            "counts_by_direction": dict(self.counts_by_direction),
            "counts_by_class": dict(self.counts_by_class),
            "flow_per_hour": {
                direction: round(rate, 1)
                for direction, rate in self.flow_per_hour(elapsed_s).items()
            },
            "speed_kmph": self.speed_summary(),
        }
