"""Overlay drawing. `track_label` and `panel_rows` are pure; `draw_*` change the frame."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from vehicle_analytics.geometry import Line, Zone
from vehicle_analytics.speed import SpeedEstimate
from vehicle_analytics.stats import Stats

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_COLOURS = {"a": (0, 0, 255), "b": (255, 0, 0)}
COUNTED_COLOUR = (0, 255, 0)
TRACKING_COLOUR = (255, 160, 0)


ZONE_COLOURS = {"lane": (90, 90, 90), "shoulder": (0, 140, 255), "restricted": (0, 0, 200)}


def track_label(
    track_id: str, cls_name: str, speed: SpeedEstimate | None, live_kmph: float | None = None
) -> str:
    """The measured zone speed when there is one; until then, the live speed marked with ~."""
    label = f"#{track_id} {cls_name}"
    if speed is not None:
        label += f" {speed.kmph:.1f} +/- {speed.err_kmph:.1f} km/h"
    elif live_kmph is not None:
        label += f" ~{live_kmph:.0f} km/h"
    return label


def draw_zones(frame: np.ndarray, zones: Iterable[Zone]) -> None:
    for zone in zones:
        outline = np.array(zone.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [outline], True, ZONE_COLOURS.get(zone.kind, (90, 90, 90)), 1)


def panel_rows(stats: Stats, elapsed_s: float) -> list[str]:
    flow = stats.flow_per_hour(elapsed_s)
    rows = [f"Total: {stats.total}"]
    for direction in sorted(stats.counts_by_direction):
        count = stats.counts_by_direction[direction]
        rows.append(f"{direction}: {count}  ({flow[direction]:.0f} veh/h)")
    return rows


def draw_lines(frame: np.ndarray, lines: Iterable[Line]) -> None:
    for line in lines:
        p1 = (round(line.p1[0]), round(line.p1[1]))
        p2 = (round(line.p2[0]), round(line.p2[1]))
        cv2.line(frame, p1, p2, LINE_COLOURS.get(line.name, (0, 255, 255)), 2)


def draw_track(frame: np.ndarray, box_xyxy: Sequence[float], label: str, counted: bool) -> None:
    x1, y1, x2, y2 = (round(v) for v in box_xyxy)
    colour = COUNTED_COLOUR if counted else TRACKING_COLOUR
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.circle(frame, ((x1 + x2) // 2, y2), 4, (0, 255, 255), -1)
    cv2.putText(frame, label, (x1, max(y1 - 6, 12)), FONT, 0.5, colour, 2)


def draw_panel(frame: np.ndarray, rows: Sequence[str]) -> None:
    cv2.rectangle(frame, (8, 8), (300, 22 + 24 * len(rows)), (0, 0, 0), -1)
    for i, row in enumerate(rows):
        cv2.putText(frame, row, (16, 32 + 24 * i), FONT, 0.6, (255, 255, 255), 2)
