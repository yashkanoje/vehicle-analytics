import pytest

from vehicle_analytics.geometry import Line
from vehicle_analytics.speed import SpeedEstimate
from vehicle_analytics.state import TrackState, VehicleRecord

A = Line("a", (0.0, 80.0), (200.0, 80.0))
B = Line("b", (0.0, 120.0), (200.0, 120.0))
LINES = [A, B]


def test_first_observation_cannot_cross_anything():
    state = TrackState("1")
    assert state.observe(1, (100.0, 60.0), "car", LINES) == []
    assert state.crossings == {}
    assert state.last_frame == 1


def test_crossings_are_returned_and_stored():
    state = TrackState("1")
    state.observe(1, (100.0, 70.0), "car", LINES)
    events = state.observe(2, (100.0, 90.0), "car", LINES)
    assert [e.line_name for e in events] == ["a"]
    assert state.crossings["a"].frame_time == pytest.approx(1.5)
    assert state.observe(3, (100.0, 130.0), "car", LINES)[0].line_name == "b"
    assert set(state.crossings) == {"a", "b"}


def test_one_step_can_cross_both_lines():
    state = TrackState("1")
    state.observe(1, (100.0, 50.0), "car", LINES)
    events = state.observe(2, (100.0, 150.0), "car", LINES)
    assert [e.line_name for e in events] == ["a", "b"]


def test_line_is_recorded_at_most_once():
    state = TrackState("1")
    state.observe(1, (100.0, 110.0), "car", LINES)
    assert len(state.observe(2, (100.0, 130.0), "car", LINES)) == 1
    assert state.observe(3, (100.0, 110.0), "car", LINES) == []
    assert state.observe(4, (100.0, 130.0), "car", LINES) == []
    assert state.crossings["b"].frame_time == pytest.approx(1.5)


def test_gap_uses_the_last_observed_frame():
    state = TrackState("1")
    state.observe(10, (100.0, 70.0), "car", LINES)
    events = state.observe(14, (100.0, 90.0), "car", LINES)
    assert events[0].frame_time == pytest.approx(12.0)
    assert state.last_frame == 14


def test_class_is_the_majority_vote():
    state = TrackState("1")
    for frame, cls_name in enumerate(["car", "truck", "car", "truck", "car"], start=1):
        state.observe(frame, (100.0, 10.0), cls_name, LINES)
    assert state.cls_name == "car"


def test_no_record_until_counted():
    state = TrackState("1")
    state.observe(1, (100.0, 60.0), "car", LINES)
    assert not state.counted
    assert state.to_record() is None


def test_record_without_speed():
    state = TrackState("1")
    state.observe(1, (100.0, 60.0), "bus", LINES)
    state.mark_counted("down", 0.55)
    assert state.counted
    assert state.to_record() == VehicleRecord("1", "bus", "down", 0.55, None, None, None)


def test_record_with_speed():
    state = TrackState("1")
    state.observe(1, (100.0, 60.0), "car", LINES)
    state.mark_counted("up", 2.0)
    state.speed = SpeedEstimate(kmph=101.5, err_kmph=6.2, mode="two_line")
    assert state.to_record() == VehicleRecord("1", "car", "up", 2.0, 101.5, 6.2, "two_line")


def test_observe_passes_the_ground_mapping_to_crossing_detection():
    def to_ground(p):
        return (p[0], 1000.0 / p[1])

    state = TrackState("1")
    state.observe(0, (100.0, 50.0), "car", [A], to_ground)
    (event,) = state.observe(10, (100.0, 200.0), "car", [A], to_ground)
    assert event.frame_time == pytest.approx(5.0)  # 20 m -> 12.5 m -> 5 m
