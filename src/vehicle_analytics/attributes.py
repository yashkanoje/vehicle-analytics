"""Per-vehicle attributes read from the image: colour, lane, and crop helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from vehicle_analytics.geometry import Point, Zone

DARK_VALUE = 60
PALE_SATURATION = 45
BRIGHT_VALUE = 170


def colour_name(crop_bgr: np.ndarray) -> str:
    """Name the colour of the central half of a crop.

    A coarse label from median hue, saturation and value. It describes what the camera
    saw, so shadow, glare and low resolution all move it.
    """
    if crop_bgr.size == 0:
        return "other"
    height, width = crop_bgr.shape[:2]
    centre = crop_bgr[height // 4 : height - height // 4, width // 4 : width - width // 4]
    if centre.size == 0:
        centre = crop_bgr
    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    saturation = float(np.median(hsv[:, 1]))
    value = float(np.median(hsv[:, 2]))
    if value < DARK_VALUE:
        return "black"
    if saturation < PALE_SATURATION:
        return "white" if value > BRIGHT_VALUE else "grey"
    # Red sits on both sides of the 0/180 wrap; shift it to the middle before the median.
    hue = (float(np.median((hsv[:, 0] + 90.0) % 180.0)) - 90.0) % 180.0
    if hue < 10.0 or hue >= 170.0:
        return "red"
    if 20.0 <= hue < 35.0:
        return "yellow"
    if 35.0 <= hue < 85.0:
        return "green"
    if 85.0 <= hue < 130.0:
        return "blue"
    return "other"


def lane_at(point: Point, zones: Iterable[Zone]) -> str | None:
    """Name of the first lane zone containing the point."""
    for zone in zones:
        if zone.kind == "lane" and zone.contains(point):
            return zone.name
    return None


def box_inside_frame(box_xyxy: Sequence[float], width: int, height: int) -> bool:
    """True when the whole box is visible: a box cut by the frame edge shows part of a vehicle."""
    x1, y1, x2, y2 = box_xyxy
    return x1 >= 1.0 and y1 >= 1.0 and x2 <= width - 1.0 and y2 <= height - 1.0


def box_area(box_xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = box_xyxy
    return max(x2 - x1, 0.0) * max(y2 - y1, 0.0)


def crop_box(frame: np.ndarray, box_xyxy: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    return frame[max(y1, 0) : y2, max(x1, 0) : x2].copy()
