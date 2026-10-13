"""Per-vehicle attributes read from the image: colour, lane, and crop helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from vehicle_analytics.geometry import Point, Zone

DARK_VALUE = 60
PALE_SATURATION = 45
BRIGHT_VALUE = 170
# Brightness relative to the road, in the camera's gamma-encoded values. Asphalt reflects
# roughly a tenth of the light: black paint no more than that, white paint about eight times
# as much, which is about 2.5 times after a gamma of 2.2. Silver falls in between.
BLACK_ROAD_RATIO = 1.2
WHITE_ROAD_RATIO = 2.5


def zone_mask(zones: Iterable[Zone], width: int, height: int) -> np.ndarray | None:
    """Pixels that belong to any zone, i.e. the road surface. None when there are no zones."""
    zones = list(zones)
    if not zones:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for zone in zones:
        cv2.fillPoly(mask, [np.array(zone.polygon, dtype=np.int32)], 1)
    return mask.astype(bool)


def road_colour(frame: np.ndarray, mask: np.ndarray | None) -> tuple[float, float, float] | None:
    """Median colour of the road in this frame: the neutral reference for vehicle colours."""
    if mask is None:
        return None
    pixels = frame[mask][::8]
    if len(pixels) == 0:
        return None
    blue, green, red = np.median(pixels, axis=0)
    return (float(blue), float(green), float(red))


def colour_name(
    crop_bgr: np.ndarray, road_bgr: tuple[float, float, float] | None = None
) -> str:
    """Name the paint colour of the vehicle in a crop. A coarse label.

    Two things would otherwise decide the answer instead of the paint. The colour of the
    light: asphalt is close to neutral grey, so when the road's colour in the same frame is
    given, the crop is white-balanced against it, and black and white are judged by
    brightness relative to the road. And the glass: seen head-on, the middle of a vehicle is
    windscreen and shadow, so the label is taken from the brighter half of the pixels.
    Without a road reference the label describes what the camera recorded, cast included.
    """
    if crop_bgr.size == 0:
        return "other"
    height, width = crop_bgr.shape[:2]
    centre = crop_bgr[height // 6 : height - height // 6, width // 6 : width - width // 6]
    if centre.size == 0:
        centre = crop_bgr
    road_value = None
    if road_bgr is not None and min(road_bgr) >= 1.0:
        road_value = sum(road_bgr) / 3.0
        gains = np.array([road_value / channel for channel in road_bgr], dtype=np.float32)
        centre = np.clip(centre.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hsv = hsv[hsv[:, 2] >= np.median(hsv[:, 2])]  # paint, not glass, tyres or shadow
    saturation = float(np.median(hsv[:, 1]))
    value = float(np.median(hsv[:, 2]))
    if road_value is None:
        dark, bright = value < DARK_VALUE, value > BRIGHT_VALUE
    else:
        dark = value < DARK_VALUE or value < BLACK_ROAD_RATIO * road_value
        bright = value > BRIGHT_VALUE or value >= WHITE_ROAD_RATIO * road_value
    if dark:
        return "black"
    if saturation < PALE_SATURATION:
        return "white" if bright else "grey"
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
