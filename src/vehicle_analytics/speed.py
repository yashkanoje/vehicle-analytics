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
    lines.

    The error bar combines two independent relative terms in quadrature: box
    localisation and the uncertainty of the real-world scale. A box-edge error
    of jitter_px across a line moves the crossing along the vehicle's path, by
    more when the path meets the line at a slant. With a homography that shift
    is converted to metres at each crossing point separately, because a pixel
    covers more road far from the camera; without one the image scale is taken
    as uniform between the lines.
    """
    elapsed_s = abs(event_b.frame_time - event_a.frame_time) / fps
    travel_px = math.dist(event_a.point_px, event_b.point_px)
    if elapsed_s <= 0.0 or travel_px <= 0.0:
        return None

    ux = (event_b.point_px[0] - event_a.point_px[0]) / travel_px
    uy = (event_b.point_px[1] - event_a.point_px[1]) / travel_px
    shifts_px = []
    for event in (event_a, event_b):
        across = abs(ux * event.line_normal[0] + uy * event.line_normal[1])
        if across < 1e-9:  # the path runs along this line; no crossing time can be trusted
            return None
        shifts_px.append(jitter_px / across)

    if homography is not None:
        ground_a = homography.to_ground(event_a.point_px)
        ground_b = homography.to_ground(event_b.point_px)
        distance_m = math.dist(ground_a, ground_b)
        if distance_m <= 0.0:
            return None
        shifts_m = [
            math.dist(ground, homography.to_ground((p[0] + shift * ux, p[1] + shift * uy)))
            for p, ground, shift in (
                (event_a.point_px, ground_a, shifts_px[0]),
                (event_b.point_px, ground_b, shifts_px[1]),
            )
        ]
        localisation = math.hypot(*shifts_m) / distance_m
        mode = "homography"
    elif meters_between_lines is not None:
        distance_m = meters_between_lines
        localisation = math.hypot(*shifts_px) / travel_px
        mode = "two_line"
    else:
        raise ValueError("zone_speed needs meters_between_lines or a homography")

    kmph = distance_m / elapsed_s * 3.6
    relative = math.hypot(localisation, distance_rel_uncertainty)
    return SpeedEstimate(kmph=kmph, err_kmph=kmph * relative, mode=mode)
