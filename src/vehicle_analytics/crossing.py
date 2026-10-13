"""Detect when a moving point crosses a line, and when within the frame interval."""

from __future__ import annotations

from dataclasses import dataclass

from vehicle_analytics.geometry import Line, Point


@dataclass(frozen=True)
class CrossingEvent:
    track_id: str
    line_name: str
    frame_time: float
    point_px: Point
    side_from: int
    side_to: int


def _side(distance: float) -> int:
    # A point exactly on the line belongs to the positive side, so a transit
    # through the line produces exactly one side change.
    return 1 if distance >= 0.0 else -1


def detect_crossing(
    line: Line,
    track_id: str,
    p_prev: Point,
    p_cur: Point,
    k_prev: float,
    k_cur: float,
) -> CrossingEvent | None:
    """Return the crossing of `line` between two observations, or None.

    The crossing instant is found by linear interpolation of the signed
    distance, so it does not depend on either observation lying near the line.
    """
    d_prev = line.signed_distance(p_prev)
    d_cur = line.signed_distance(p_cur)
    side_from, side_to = _side(d_prev), _side(d_cur)
    if side_from == side_to:
        return None
    t = d_prev / (d_prev - d_cur)
    point = (
        p_prev[0] + t * (p_cur[0] - p_prev[0]),
        p_prev[1] + t * (p_cur[1] - p_prev[1]),
    )
    if not line.within_segment(point):
        return None
    return CrossingEvent(
        track_id=track_id,
        line_name=line.name,
        frame_time=k_prev + t * (k_cur - k_prev),
        point_px=point,
        side_from=side_from,
        side_to=side_to,
    )
