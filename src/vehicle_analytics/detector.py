"""YOLO wrapper. Ultralytics is imported only when a detector is built."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    box_xyxy: Box
    conf: float
    cls_name: str


def detections_from_result(
    xyxy: Sequence[Sequence[float]],
    confs: Sequence[float],
    cls_ids: Sequence[float],
    names: Mapping[int, str],
    allowed: Collection[str],
) -> list[Detection]:
    """Turn raw model output into detections, keeping only the allowed classes."""
    detections: list[Detection] = []
    for box, conf, cls_id in zip(xyxy, confs, cls_ids, strict=True):
        cls_name = names[int(cls_id)]
        if cls_name not in allowed:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        detections.append(Detection((x1, y1, x2, y2), float(conf), cls_name))
    return detections


class YoloDetector:
    def __init__(
        self, weights: str, conf: float, classes: Collection[str], device: str | None = None
    ) -> None:
        import torch
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self.conf = conf
        self.classes = frozenset(classes)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self._model(frame, conf=self.conf, device=self.device, verbose=False)[0]
        boxes = result.boxes
        return detections_from_result(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), result.names, self.classes
        )
