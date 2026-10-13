import csv
import json

import numpy as np
import pytest

from vehicle_analytics.config import parse_config
from vehicle_analytics.pipeline import ArraySource, FpsError, VideoFileSource, VideoOpenError, run
from vehicle_analytics.tracking import Track

SIZE = 200
FPS = 10.0


def config(**changes):
    data = {
        "lines": {"a": [[0.0, 0.4], [1.0, 0.4]], "b": [[0.0, 0.6], [1.0, 0.6]]},
        "speed": {"meters_between_lines": 10.0},
        "tracker": {"max_age": 2},
    }
    data.update(changes)
    return parse_config(data)


def source(n_frames, fps=FPS):
    frames = [np.zeros((SIZE, SIZE, 3), dtype=np.uint8) for _ in range(n_frames)]
    return ArraySource(frames, fps, SIZE, SIZE)


def box(y_bottom, x=100.0):
    return (x - 20.0, y_bottom - 30.0, x + 20.0, y_bottom)


class NoDetector:
    def detect(self, frame):
        return []


class ScriptedTracker:
    """Replays {frame_index: [Track, ...]}; frames not listed have no tracks."""

    def __init__(self, script):
        self.script = script
        self.frame = 0

    def update(self, detections, frame):
        self.frame += 1
        return self.script.get(self.frame, [])


def path_of(track_id, ys, cls_names=None, start=1):
    """Script for one track: its bottom-centre y per frame from `start`."""
    cls_names = cls_names or ["car"] * len(ys)
    return {
        start + i: [Track(track_id, box(y), cls_name)]
        for i, (y, cls_name) in enumerate(zip(ys, cls_names, strict=True))
    }


def run_script(script, n_frames, tmp_path, cfg=None, src=None):
    return run(
        src or source(n_frames),
        cfg or config(),
        NoDetector(),
        ScriptedTracker(script),
        tmp_path / "out",
        video_name="test.mp4",
    )


def test_one_vehicle_is_counted_timed_and_measured(tmp_path):
    result = run_script(path_of("1", [60, 75, 85, 105, 115, 125]), 6, tmp_path)
    (record,) = result.records
    assert record.track_id == "1"
    assert record.cls_name == "car"
    assert record.direction == "down"
    assert record.counted_at_s == pytest.approx(0.55)
    assert record.speed_kmph == pytest.approx(120.0)
    assert record.speed_err_kmph == pytest.approx(10.3923, abs=1e-3)
    assert record.speed_mode == "two_line"
    assert result.frames_processed == 6
    assert result.summary["total"] == 1
    assert result.summary["flow_per_hour"] == {"down": 6000.0}


def test_fast_vehicle_crossing_both_lines_in_one_step_is_counted(tmp_path):
    # Regression for defect 1.
    result = run_script(path_of("2", [50, 150]), 2, tmp_path)
    assert result.summary["counts_by_direction"] == {"down": 1}
    assert result.records[0].speed_kmph is not None


def test_upward_vehicle_gets_the_other_direction(tmp_path):
    result = run_script(path_of("3", [140, 100, 60]), 3, tmp_path)
    assert result.records[0].direction == "up"
    assert result.summary["counts_by_direction"] == {"up": 1}


def test_id_switch_between_the_lines_counts_once_without_speed(tmp_path):
    script = {**path_of("5", [70, 90]), **path_of("6", [100, 130], start=4)}
    result = run_script(script, 8, tmp_path)
    (record,) = result.records
    assert record.track_id == "6"
    assert record.speed_kmph is None
    assert result.summary["total"] == 1


def test_jitter_across_count_line_counts_once(tmp_path):
    result = run_script(path_of("7", [110, 130, 110, 130]), 4, tmp_path)
    assert result.summary["total"] == 1


def test_class_is_the_majority_over_the_track(tmp_path):
    script = path_of("8", [110, 130, 135], ["car", "car", "truck"])
    result = run_script(script, 3, tmp_path)
    assert result.records[0].cls_name == "car"
    assert result.summary["counts_by_class"] == {"car": 1}


def test_vehicle_that_never_reaches_the_count_line_is_not_recorded(tmp_path):
    result = run_script(path_of("9", [60, 90]), 2, tmp_path)
    assert result.records == []
    assert result.summary["total"] == 0


def test_track_lost_mid_video_is_still_recorded(tmp_path):
    result = run_script(path_of("10", [110, 130]), 12, tmp_path)
    assert [r.track_id for r in result.records] == ["10"]


def test_output_files(tmp_path):
    result = run_script(path_of("1", [60, 75, 85, 105, 115, 125]), 6, tmp_path)
    out = result.out_dir
    assert (out / "annotated.mp4").stat().st_size > 0

    with (out / "events.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "track_id": "1",
            "class": "car",
            "direction": "down",
            "counted_at_s": "0.550",
            "speed_kmph": "120.00",
            "speed_err_kmph": "10.39",
            "speed_mode": "two_line",
        }
    ]

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["video"] == "test.mp4"
    assert summary["fps"] == 10.0
    assert summary["frames_processed"] == 6
    assert summary["elapsed_s"] == 0.6
    assert summary["total"] == 1
    assert summary["config"]["speed"]["meters_between_lines"] == 10.0


def test_missing_speed_is_written_as_an_empty_cell(tmp_path):
    result = run_script(path_of("6", [100, 130]), 2, tmp_path)
    text = (result.out_dir / "events.csv").read_text(encoding="utf-8")
    assert text.splitlines()[1] == "6,car,down,0.167,,,"


def test_empty_source_still_writes_outputs(tmp_path):
    result = run_script({}, 0, tmp_path)
    assert result.records == []
    assert result.frames_processed == 0
    assert result.summary["flow_per_hour"] == {}
    assert (result.out_dir / "events.csv").read_text(encoding="utf-8").startswith("track_id,")
    assert json.loads((result.out_dir / "summary.json").read_text(encoding="utf-8"))["total"] == 0


class InterruptingTracker(ScriptedTracker):
    def __init__(self, script, stop_at):
        super().__init__(script)
        self.stop_at = stop_at

    def update(self, detections, frame):
        if self.frame + 1 == self.stop_at:
            raise KeyboardInterrupt
        return super().update(detections, frame)


def test_interrupt_still_writes_what_was_processed(tmp_path):
    tracker = InterruptingTracker(path_of("1", [110, 130]), stop_at=4)
    result = run(source(10), config(), NoDetector(), tracker, tmp_path / "out")
    assert [r.track_id for r in result.records] == ["1"]
    assert (result.out_dir / "events.csv").is_file()
    assert json.loads((result.out_dir / "summary.json").read_text(encoding="utf-8"))["total"] == 1


def test_missing_frame_rate_is_an_error(tmp_path):
    with pytest.raises(FpsError, match="fps_override"):
        run_script({}, 2, tmp_path, src=source(2, fps=0.0))


def test_fps_override_replaces_the_reported_frame_rate(tmp_path):
    script = path_of("1", [60, 75, 85, 105, 115, 125])
    result = run_script(script, 6, tmp_path, cfg=config(fps_override=20), src=source(6, fps=0.0))
    assert result.records[0].speed_kmph == pytest.approx(240.0)


def test_calibration_switches_speed_to_homography_mode(tmp_path):
    # Whole frame is a 20 m x 20 m square: 0.1 m per pixel, so 40 px is 4 m.
    calibration = {
        "image_points": [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
        "ground_points_m": [[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]],
    }
    script = path_of("1", [60, 75, 85, 105, 115, 125])
    result = run_script(script, 6, tmp_path, cfg=config(calibration=calibration))
    assert result.records[0].speed_mode == "homography"
    assert result.records[0].speed_kmph == pytest.approx(48.0)


def test_unreadable_video_raises(tmp_path):
    with pytest.raises(VideoOpenError, match="cannot open video"):
        VideoFileSource(tmp_path / "missing.mp4")
