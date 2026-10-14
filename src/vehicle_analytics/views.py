"""What the dashboard shows, worked out from database rows.

No Streamlit here, so these functions are tested with the core install.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Row = Mapping[str, Any]

DB_ENV = "VEHICLE_ANALYTICS_DB"  # how the CLI tells the Streamlit script what to open
ROOT_ENV = "VEHICLE_ANALYTICS_ROOT"
NOT_RECORDED = "(not recorded)"


@dataclass(frozen=True)
class RunOverview:
    duration_s: float | None  # None when the run did not record its length
    counted: int
    uncounted: int  # stored because they broke a rule, but never crossed the count line
    by_direction: dict[str, int]
    by_class: dict[str, int]
    flow_per_hour: dict[str, float]
    speeds_kmph: list[float]
    violations_by_rule: dict[str, int]


def run_overview(run: Row, vehicles: Sequence[Row], violations: Sequence[Row]) -> RunOverview:
    """The overview numbers of one run, counted the way summary.json counts them."""
    counted = [v for v in vehicles if v["counted_at_s"] is not None]
    frames, fps = run["frame_count"], run["fps"]
    duration = frames / fps if frames and fps else None
    by_direction = dict(Counter(v["direction"] for v in counted))
    return RunOverview(
        duration_s=duration,
        counted=len(counted),
        uncounted=len(vehicles) - len(counted),
        by_direction=by_direction,
        by_class=dict(Counter(v["class"] for v in counted)),
        flow_per_hour=(
            {d: n / duration * 3600 for d, n in by_direction.items()} if duration else {}
        ),
        speeds_kmph=[v["speed_kmph"] for v in counted if v["speed_kmph"] is not None],
        violations_by_rule=dict(Counter(v["rule"] for v in violations)),
    )


def speed_histogram(speeds: Sequence[float], band_kmph: float = 10.0) -> dict[float, int]:
    """Vehicles per speed band, keyed by the band's lower edge, empty bands included."""
    if not speeds:
        return {}
    bands = Counter(math.floor(s / band_kmph) for s in speeds)
    return {b * band_kmph: bands[b] for b in range(min(bands), max(bands) + 1)}


def _label(value: Any) -> str:
    return NOT_RECORDED if value is None else str(value)


def options(rows: Sequence[Row], key: str) -> list[str]:
    """The distinct values of a column, for a filter. A missing value is NOT_RECORDED."""
    return sorted({_label(row[key]) for row in rows})


def filter_vehicles(
    rows: Sequence[Row],
    *,
    classes: Sequence[str] | None = None,
    colours: Sequence[str] | None = None,
    lanes: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    speed_range: tuple[float, float] | None = None,
) -> list[Row]:
    """The rows that match every filter given; a filter left as None does not filter.

    A speed range keeps only vehicles with a measured speed inside it, so the caller leaves
    it None while the range is untouched, to keep the vehicles that have no speed.
    """
    chosen = {"class": classes, "colour": colours, "lane": lanes, "direction": directions}
    kept = []
    for row in rows:
        if any(
            allowed is not None and _label(row[key]) not in allowed
            for key, allowed in chosen.items()
        ):
            continue
        if speed_range is not None:
            speed = row["speed_kmph"]
            if speed is None or not speed_range[0] <= speed <= speed_range[1]:
                continue
        kept.append(row)
    return kept


def vehicle_table(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """One display row per vehicle, in the columns and units the dashboard shows."""

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, 1)

    return [
        {
            "record": row["id"],
            "track": row["track_id"],
            "class": row["class"],
            "colour (unverified)": row["colour"],
            "direction": row["direction"],
            "lane": row["lane"],
            "speed km/h": rounded(row["speed_kmph"]),
            "± km/h": rounded(row["speed_err_kmph"]),
            "counted at s": rounded(row["counted_at_s"]),
        }
        for row in rows
    ]
