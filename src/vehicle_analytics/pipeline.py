"""Frame loop: detection -> tracking -> crossings -> speed -> stats -> output files."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from vehicle_analytics import render
from vehicle_analytics.config import Config
from vehicle_analytics.detector import Detection
from vehicle_analytics.geometry import reference_point
from vehicle_analytics.speed import zone_speed
from vehicle_analytics.state import TrackState, VehicleRecord
from vehicle_analytics.stats import Stats
from vehicle_analytics.tracking import Track

WINDOW_TITLE = "vehicle-analytics (Esc to stop)"
ESC_KEY = 27
EVENT_COLUMNS = [
    "track_id",
    "class",
    "direction",
    "counted_at_s",
    "speed_kmph",
    "speed_err_kmph",
    "speed_mode",
]


class VideoOpenError(RuntimeError):
    """The video file could not be opened or holds no frames."""


class FpsError(RuntimeError):
    """The video reports no usable frame rate and the config gives no override."""


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class Tracker(Protocol):
    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]: ...


class FrameSource(Protocol):
    fps: float
    width: int
    height: int

    def __iter__(self) -> Iterator[np.ndarray]: ...


class VideoFileSource:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise VideoOpenError(f"cannot open video: {self.path}")
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterator[np.ndarray]:
        try:
            while True:
                ok, frame = self._capture.read()
                if not ok:
                    break
                yield frame
        finally:
            self._capture.release()


class ArraySource:
    """Frames held in memory. Used by tests."""

    def __init__(self, frames: Sequence[np.ndarray], fps: float, width: int, height: int) -> None:
        self._frames = list(frames)
        self.fps = fps
        self.width = width
        self.height = height

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self._frames)


@dataclass
class RunResult:
    records: list[VehicleRecord]
    summary: dict[str, Any]
    out_dir: Path
    frames_processed: int


def _cell(value: float | None, digits: int) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def write_outputs(out_dir: Path, records: Sequence[VehicleRecord], summary: dict) -> None:
    with (out_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EVENT_COLUMNS)
        for record in records:
            writer.writerow(
                [
                    record.track_id,
                    record.cls_name,
                    record.direction,
                    _cell(record.counted_at_s, 3),
                    _cell(record.speed_kmph, 2),
                    _cell(record.speed_err_kmph, 2),
                    record.speed_mode or "",
                ]
            )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run(
    source: FrameSource,
    config: Config,
    detector: Detector,
    tracker: Tracker,
    out_dir: str | Path,
    *,
    video_name: str = "video",
    display: bool = False,
) -> RunResult:
    fps = config.fps_override if config.fps_override is not None else source.fps
    if not fps or fps <= 0.0:
        raise FpsError("video reports no frame rate; set 'fps_override' in the config")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = config.lines_px(source.width, source.height)
    homography = config.homography_px(source.width, source.height)
    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (source.width, source.height),
    )

    states: dict[str, TrackState] = {}
    records: list[VehicleRecord] = []
    stats = Stats()
    frame_index = 0

    def finalise(track_id: str) -> None:
        record = states.pop(track_id).to_record()
        if record is not None:
            records.append(record)
            stats.record_vehicle(record)

    try:
        for frame_index, frame in enumerate(source, start=1):
            tracks = tracker.update(detector.detect(frame), frame)

            for track in tracks:
                state = states.setdefault(track.track_id, TrackState(track.track_id))
                point = reference_point(track.box_xyxy)
                for event in state.observe(frame_index, point, track.cls_name, lines.values()):
                    if event.line_name == config.count_line:
                        direction = config.direction_label(event.side_to)
                        state.mark_counted(direction, event.frame_time / fps)
                        stats.record_count(direction)
                if state.speed is None and len(state.crossings) == len(lines):
                    state.speed = zone_speed(
                        state.crossings["a"],
                        state.crossings["b"],
                        fps,
                        meters_between_lines=config.speed.meters_between_lines,
                        homography=homography,
                        jitter_px=config.speed.jitter_px,
                        distance_rel_uncertainty=config.speed.distance_rel_uncertainty,
                    )
                label = render.track_label(track.track_id, state.cls_name, state.speed)
                render.draw_track(frame, track.box_xyxy, label, state.counted)

            # The tracker forgets a vehicle after max_age unmatched frames; so do we.
            lost = [
                track_id
                for track_id, state in states.items()
                if frame_index - state.last_frame > config.tracker.max_age
            ]
            for track_id in lost:
                finalise(track_id)

            render.draw_lines(frame, lines.values())
            render.draw_panel(frame, render.panel_rows(stats, frame_index / fps))
            writer.write(frame)

            if display:
                cv2.imshow(WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == ESC_KEY:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        # Reached on normal end, Esc, Ctrl-C and errors alike: whatever was
        # processed is finalised and written.
        writer.release()
        if display:
            cv2.destroyAllWindows()
        for track_id in list(states):
            finalise(track_id)
        elapsed_s = frame_index / fps
        summary = {
            "video": video_name,
            "fps": fps,
            "frames_processed": frame_index,
            "elapsed_s": round(elapsed_s, 3),
            **stats.summary(elapsed_s),
            "config": config.raw,
        }
        write_outputs(out_dir, records, summary)

    return RunResult(
        records=records, summary=summary, out_dir=out_dir, frames_processed=frame_index
    )
