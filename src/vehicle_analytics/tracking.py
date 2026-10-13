"""DeepSORT wrapper. deep-sort-realtime is imported only when a tracker is built."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vehicle_analytics.detector import Box, Detection


@dataclass(frozen=True)
class Track:
    track_id: str
    box_xyxy: Box
    cls_name: str


def to_deepsort_input(detections: Sequence[Detection]) -> list[tuple[list[float], float, str]]:
    """DeepSORT takes ([left, top, width, height], confidence, class)."""
    raw = []
    for detection in detections:
        x1, y1, x2, y2 = detection.box_xyxy
        raw.append(([x1, y1, x2 - x1, y2 - y1], detection.conf, detection.cls_name))
    return raw


def tracks_from_deepsort(raw_tracks: Iterable[Any]) -> list[Track]:
    """Keep tracks that are confirmed and were matched to a detection this frame.

    A coasting track (time_since_update > 0) carries a box predicted by the
    motion model, not a measurement. Using it would let a vehicle that is no
    longer detected drift across a line.
    """
    tracks: list[Track] = []
    for raw in raw_tracks:
        if not raw.is_confirmed() or raw.time_since_update != 0:
            continue
        x1, y1, x2, y2 = (float(v) for v in raw.to_ltrb())
        tracks.append(Track(str(raw.track_id), (x1, y1, x2, y2), raw.get_det_class() or "vehicle"))
    return tracks


class DeepSortTracker:
    def __init__(self, max_age: int, n_init: int, max_cosine_distance: float) -> None:
        import torch
        from deep_sort_realtime.deepsort_tracker import DeepSort

        use_gpu = torch.cuda.is_available()
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",
            half=use_gpu,
            embedder_gpu=use_gpu,
        )

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        raw_tracks = self._tracker.update_tracks(to_deepsort_input(detections), frame=frame)
        return tracks_from_deepsort(raw_tracks)
