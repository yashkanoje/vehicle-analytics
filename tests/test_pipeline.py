import csv
import datetime
import json

import cv2
import numpy as np
import pytest

from vehicle_analytics.config import parse_config
from vehicle_analytics.pipeline import (
    ArraySource,
    FpsError,
    OutputError,
    VideoFileSource,
    VideoOpenError,
    run,
)
from vehicle_analytics.store import Store
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
        (row,) = list(csv.DictReader(handle))
    snapshot = row.pop("snapshot_path")
    assert row == {
        "track_id": "1",
        "class": "car",
        "direction": "down",
        "counted_at_s": "0.550",
        "speed_kmph": "120.00",
        "speed_err_kmph": "10.39",
        "speed_mode": "two_line",
        "colour": "black",
        "lane": "",
        "first_seen_s": "0.100",
        "last_seen_s": "0.600",
    }
    assert snapshot.endswith("out/snapshots/1.jpg")
    assert (out / "snapshots" / "1.jpg").stat().st_size > 0

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
    assert text.splitlines()[1].split(",")[:7] == ["6", "car", "down", "0.167", "", "", ""]


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


def test_constant_ground_speed_is_measured_across_tracking_gaps(tmp_path):
    # Perspective scene: 60 m of road, its far edge a quarter of the way down the frame.
    calibration = {
        "image_points": [[0.0, 1.0], [1.0, 1.0], [0.75, 0.25], [0.25, 0.25]],
        "ground_points_m": [[0.0, 0.0], [10.0, 0.0], [10.0, 60.0], [0.0, 60.0]],
    }
    cfg = config(calibration=calibration, tracker={"max_age": 15})
    to_image = np.linalg.inv(cfg.homography_px(SIZE, SIZE).matrix)

    def image_y(ground_y):
        _, y, w = to_image @ np.array([5.0, ground_y, 1.0])
        return y / w

    # 1.4 m per frame towards the camera, but seen only every tenth frame.
    script = {
        1 + 10 * i: [Track("1", box(image_y(ground_y)), "car")]
        for i, ground_y in enumerate([58.0, 44.0, 30.0, 16.0, 2.0])
    }
    result = run_script(script, 41, tmp_path, cfg=cfg)
    assert result.records[0].speed_kmph == pytest.approx(1.4 * FPS * 3.6, rel=0.005)


def test_summary_survives_non_json_values_in_the_config(tmp_path):
    # YAML turns `recorded: 2025-12-08` into a date object.
    result = run_script({}, 1, tmp_path, cfg=config(recorded=datetime.date(2025, 12, 8)))
    summary = json.loads((result.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["config"]["recorded"] == "2025-12-08"


def test_out_path_that_is_a_file_is_an_output_error(tmp_path):
    (tmp_path / "out").write_text("not a folder", encoding="utf-8")
    with pytest.raises(OutputError, match="output folder"):
        run_script({}, 1, tmp_path)


def test_video_writer_that_cannot_open_is_an_output_error(tmp_path, monkeypatch):
    # Whether a writer opens depends on the codecs installed, so the failure is simulated.
    class ClosedWriter:
        def __init__(self, *args):
            pass

        def isOpened(self):  # noqa: N802 - OpenCV's name
            return False

        def release(self):
            pass

    monkeypatch.setattr("vehicle_analytics.pipeline.cv2.VideoWriter", ClosedWriter)
    with pytest.raises(OutputError, match="annotated.mp4"):
        run_script({}, 1, tmp_path)


LANES = [
    {
        "name": "left",
        "kind": "lane",
        "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
        "allowed_direction": "down",
    },
    {
        "name": "right",
        "kind": "lane",
        "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
        "allowed_direction": "down",
    },
]
DOWN = [60, 75, 85, 105, 115, 125]


def path_at(track_id, ys, x, start=1):
    return {start + i: [Track(track_id, box(y, x), "car")] for i, y in enumerate(ys)}


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "db" / "analytics.db") as opened:
        yield opened


def run_with_store(script, n_frames, tmp_path, store, cfg=None, tracker=None):
    return run(
        source(n_frames),
        cfg or config(zones=LANES),
        NoDetector(),
        tracker or ScriptedTracker(script),
        tmp_path / "out",
        video_name="test.mp4",
        store=store,
    )


def test_vehicle_row_is_written_with_lane_colour_and_snapshot(tmp_path, store):
    result = run_with_store(path_at("1", DOWN, x=150.0), 6, tmp_path, store)
    run_id = result.summary["run_id"]
    (run_row,) = store.runs()
    assert (run_row["id"], run_row["video"], run_row["frame_count"]) == (run_id, "test.mp4", 6)
    (row,) = store.vehicles(run_id)
    assert (row["track_id"], row["class"], row["direction"]) == ("1", "car", "down")
    assert (row["lane"], row["colour"]) == ("right", "black")
    assert row["speed_kmph"] == pytest.approx(120.0)
    assert (row["first_seen_s"], row["last_seen_s"]) == (0.1, 0.6)
    assert row["snapshot_path"].endswith("out/snapshots/1.jpg")
    assert result.records[0].lane == "right"


def test_lane_follows_the_side_of_the_road(tmp_path, store):
    script = {**path_at("1", DOWN, x=50.0), **path_at("2", DOWN, x=150.0, start=7)}
    result = run_with_store(script, 14, tmp_path, store)
    lanes = {row["track_id"]: row["lane"] for row in store.vehicles(result.summary["run_id"])}
    assert lanes == {"1": "left", "2": "right"}


def test_lane_is_empty_outside_every_zone(tmp_path, store):
    one_lane = config(zones=LANES[:1])
    result = run_with_store(path_at("1", DOWN, x=150.0), 6, tmp_path, store, cfg=one_lane)
    assert store.vehicles(result.summary["run_id"])[0]["lane"] is None
    no_zones = run_with_store(path_at("2", DOWN, x=150.0), 6, tmp_path, store, cfg=config())
    assert store.vehicles(no_zones.summary["run_id"])[0]["lane"] is None


def test_vehicle_never_fully_in_frame_has_no_snapshot_or_colour(tmp_path, store):
    # x = 10 puts the left edge of every box outside the frame.
    result = run_with_store(path_at("1", DOWN, x=10.0), 6, tmp_path, store)
    (row,) = store.vehicles(result.summary["run_id"])
    assert row["snapshot_path"] is None and row["colour"] is None
    assert not (result.out_dir / "snapshots").exists()


def test_only_counted_vehicles_are_stored(tmp_path, store):
    result = run_with_store(path_at("9", [60, 90], x=150.0), 2, tmp_path, store)
    assert store.vehicles(result.summary["run_id"]) == []
    assert not (result.out_dir / "snapshots").exists()


def test_second_run_into_the_same_database_adds_a_run(tmp_path, store):
    first = run_with_store(path_at("1", DOWN, x=150.0), 6, tmp_path, store)
    second = run_with_store(path_at("1", DOWN, x=150.0), 6, tmp_path, store)
    assert first.summary["run_id"] != second.summary["run_id"]
    assert len(store.runs()) == 2
    assert len(store.vehicles(second.summary["run_id"])) == 1


def test_interrupted_run_keeps_its_rows(tmp_path, store):
    tracker = InterruptingTracker(path_at("1", [110, 130], x=150.0), stop_at=4)
    result = run_with_store({}, 10, tmp_path, store, tracker=tracker)
    run_id = result.summary["run_id"]
    assert [row["track_id"] for row in store.vehicles(run_id)] == ["1"]
    assert store.runs()[0]["frame_count"] == result.frames_processed


def test_snapshot_is_taken_before_the_overlay_is_drawn(tmp_path, store):
    # Two overlapping boxes: the second crop would contain the first box's outline if
    # anything were drawn before all crops were taken.
    script = {
        frame: [Track("1", box(y, 60.0), "car"), Track("2", box(y, 90.0), "car")]
        for frame, y in enumerate(DOWN, start=1)
    }
    result = run_with_store(script, 6, tmp_path, store)
    for name in ("1.jpg", "2.jpg"):
        image = cv2.imread(str(result.out_dir / "snapshots" / name))
        assert image.max() < 30  # the frames are black; any overlay would be bright


def test_run_without_a_store_still_works(tmp_path):
    result = run_script(path_of("1", DOWN), 6, tmp_path)
    assert result.summary["run_id"] is None


def test_colour_is_balanced_against_the_road_inside_the_zones(tmp_path, store):
    # A blue-cast road with a neutral silver vehicle painted where the tracker reports it.
    road, silver = (69, 50, 42), (116, 99, 84)
    frames, script = [], {}
    for frame_index, y in enumerate(DOWN, start=1):
        frame = np.full((SIZE, SIZE, 3), road, dtype=np.uint8)
        x1, y1, x2, y2 = (int(v) for v in box(y, 150.0))
        frame[y1:y2, x1:x2] = silver
        frames.append(frame)
        script[frame_index] = [Track("1", box(y, 150.0), "car")]
    result = run(
        ArraySource(frames, FPS, SIZE, SIZE),
        config(zones=LANES),
        NoDetector(),
        ScriptedTracker(script),
        tmp_path / "out",
        store=store,
    )
    assert store.vehicles(result.summary["run_id"])[0]["colour"] == "grey"
