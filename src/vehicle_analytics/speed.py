"""Speed between two line crossings, with a one-sigma error bar."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vehicle_analytics.crossing import CrossingEvent
from vehicle_analytics.geometry import Homography


@dataclass(frozen=True)
class SpeedEstimate:
    kmph: float
    err_kmph: float
    mode: str


def zone_speed(
    event_a: CrossingEvent,
    event_b: CrossingEvent,
    fps: float,
    *,
    meters_between_lines: float | None = None,
    homography: Homography | None = None,
    jitter_px: float = 2.0,
    distance_rel_uncertainty: float = 0.05,
) -> SpeedEstimate | None:
    """Speed over the zone between two crossings.

    With a homography the distance is measured on the ground plane between the
    two crossing points. Without one it is the configured distance between the
    lines. The error bar combines two independent relative terms in quadrature:
    box localisation jitter over the pixel travel, and the uncertainty of the
    real-world scale.
    """
    elapsed_s = abs(event_b.frame_time - event_a.frame_time) / fps
    travel_px = math.dist(event_a.point_px, event_b.point_px)
    if elapsed_s <= 0.0 or travel_px <= 0.0:
        return None

    if homography is not None:
        distance_m = math.dist(
            homography.to_ground(event_a.point_px), homography.to_ground(event_b.point_px)
        )
        mode = "homography"
    elif meters_between_lines is not None:
        distance_m = meters_between_lines
        mode = "two_line"
    else:
        raise ValueError("zone_speed needs meters_between_lines or a homography")

    kmph = distance_m / elapsed_s * 3.6
    localisation = math.sqrt(2.0) * jitter_px / travel_px
    relative = math.hypot(localisation, distance_rel_uncertainty)
    return SpeedEstimate(kmph=kmph, err_kmph=kmph * relative, mode=mode)
