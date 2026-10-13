import pytest

from vehicle_analytics.state import VehicleRecord
from vehicle_analytics.stats import Stats


def record(track_id, cls_name, direction, speed=None):
    mode = "two_line" if speed is not None else None
    err = 1.0 if speed is not None else None
    return VehicleRecord(track_id, cls_name, direction, 1.0, speed, err, mode)


def test_counts_by_direction_and_total():
    stats = Stats()
    for direction in ["down", "down", "up"]:
        stats.record_count(direction)
    assert stats.counts_by_direction == {"down": 2, "up": 1}
    assert stats.total == 3


def test_flow_uses_elapsed_video_time():
    # Regression for defect 2: the rate depends only on video time elapsed so far.
    stats = Stats()
    for _ in range(5):
        stats.record_count("down")
    assert stats.flow_per_hour(10.0) == pytest.approx({"down": 1800.0})
    assert stats.flow_per_hour(20.0) == pytest.approx({"down": 900.0})


def test_flow_is_zero_before_any_time_has_passed():
    stats = Stats()
    stats.record_count("down")
    assert stats.flow_per_hour(0.0) == {"down": 0.0}
    assert Stats().flow_per_hour(0.0) == {}


def test_vehicles_add_class_counts_and_speeds():
    stats = Stats()
    stats.record_vehicle(record("1", "car", "down", 100.0))
    stats.record_vehicle(record("2", "car", "down", 110.0))
    stats.record_vehicle(record("3", "truck", "down", 90.0))
    stats.record_vehicle(record("4", "car", "up"))
    assert stats.counts_by_class == {"car": 3, "truck": 1}
    assert stats.speed_summary() == {
        "down": {"n": 3, "mean": 100.0, "median": 100.0, "max": 110.0}
    }


def test_summary_shape():
    stats = Stats()
    stats.record_count("down")
    stats.record_vehicle(record("1", "car", "down", 72.0))
    assert stats.summary(60.0) == {
        "total": 1,
        "counts_by_direction": {"down": 1},
        "counts_by_class": {"car": 1},
        "flow_per_hour": {"down": 60.0},
        "speed_kmph": {"down": {"n": 1, "mean": 72.0, "median": 72.0, "max": 72.0}},
    }


def test_empty_summary():
    assert Stats().summary(0.0) == {
        "total": 0,
        "counts_by_direction": {},
        "counts_by_class": {},
        "flow_per_hour": {},
        "speed_kmph": {},
    }
