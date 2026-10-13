"""Turn four points on a road rectangle of known size into a calibration block.

Point order, for clicks and for --points alike:
near-left, near-right, far-right, far-left.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from vehicle_analytics.geometry import Point
from vehicle_analytics.pipeline import VideoOpenError

WINDOW_TITLE = "calibrate: near-left, near-right, far-right, far-left (Esc cancels)"
ESC_KEY = 27


class CalibrationError(ValueError):
    """The calibration input is malformed or the user cancelled."""


def parse_points(text: str) -> list[Point]:
    try:
        points = [
            (float(x), float(y)) for x, y in (pair.split(",") for pair in text.split())
        ]
    except ValueError:
        points = []
    if len(points) != 4:
        raise CalibrationError("--points needs four 'x,y' pairs separated by spaces")
    return points


def calibration_block(
    points_px: Sequence[Point], width: int, height: int, width_m: float, length_m: float
) -> dict:
    if len(points_px) != 4:
        raise CalibrationError("calibration needs exactly four points")
    if width_m <= 0.0 or length_m <= 0.0:
        raise CalibrationError("--width-m and --length-m must be greater than 0")
    return {
        "calibration": {
            "image_points": [[round(x / width, 4), round(y / height, 4)] for x, y in points_px],
            "ground_points_m": [
                [0.0, 0.0],
                [float(width_m), 0.0],
                [float(width_m), float(length_m)],
                [0.0, float(length_m)],
            ],
        }
    }


def first_frame(video: str | Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise VideoOpenError(f"cannot open video: {video}")
        ok, frame = capture.read()
        if not ok:
            raise VideoOpenError(f"video has no frames: {video}")
        return frame
    finally:
        capture.release()


def pick_points(frame: np.ndarray) -> list[Point]:
    points: list[Point] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))

    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, on_mouse)
    try:
        while len(points) < 4:
            view = frame.copy()
            for index, (x, y) in enumerate(points, start=1):
                cv2.circle(view, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(
                    view, str(index), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )
            cv2.imshow(WINDOW_TITLE, view)
            if cv2.waitKey(20) & 0xFF == ESC_KEY:
                raise CalibrationError("calibration cancelled")
    finally:
        cv2.destroyWindow(WINDOW_TITLE)
    return points


def calibration_yaml(
    video: str | Path, width_m: float, length_m: float, points_text: str | None = None
) -> str:
    frame = first_frame(video)
    height, width = frame.shape[:2]
    points = parse_points(points_text) if points_text is not None else pick_points(frame)
    block = calibration_block(points, width, height, width_m, length_m)
    return yaml.safe_dump(block, sort_keys=False, default_flow_style=None)
