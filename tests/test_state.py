import numpy as np
import pytest

from vehicle_analytics.geometry import Line, Zone
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


BAND = Zone("band", "lane", ((0.0, 100.0), (200.0, 100.0), (200.0, 150.0), (0.0, 150.0)), "down")


def metres(p):
    return (p[0] / 10.0, p[1] / 10.0)


def test_ground_history_is_kept_when_a_mapping_is_given():
    state = TrackState("1")
    state.observe(1, (100.0, 50.0), "car", LINES, metres)
    state.observe(2, (100.0, 60.0), "car", LINES, metres)
    assert state.ground_history == [(1, (10.0, 5.0)), (2, (10.0, 6.0))]
    assert state.first_frame == 1


def test_zone_membership_records_the_entry_frame_and_forgets_on_exit():
    state = TrackState("1")
    state.observe(1, (100.0, 90.0), "car", [], None, [BAND])
    assert state.zone_entered == {}
    state.observe(2, (100.0, 110.0), "car", [], None, [BAND])
    state.observe(3, (100.0, 130.0), "car", [], None, [BAND])
    assert state.zone_entered == {"band": 2}
    state.observe(4, (100.0, 160.0), "car", [], None, [BAND])
    assert state.zone_entered == {}


def test_live_speed_over_the_window():
    state = TrackState("1")
    for frame in range(1, 12):  # 1 m per frame at 10 fps = 10 m/s
        state.observe(frame, (100.0, 10.0 * frame), "car", [], metres)
    assert state.live_speed_kmph(fps=10.0, window_s=0.5) == pytest.approx(36.0)


def test_live_speed_needs_two_ground_points_inside_the_window():
    state = TrackState("1")
    assert state.live_speed_kmph(10.0, 0.5) is None
    state.observe(1, (100.0, 10.0), "car", [], metres)
    assert state.live_speed_kmph(10.0, 0.5) is None
    state.observe(30, (100.0, 300.0), "car", [], metres)  # previous point is too old
    assert state.live_speed_kmph(10.0, 0.5) is None


def test_live_speed_is_none_without_a_ground_mapping():
    state = TrackState("1")
    state.observe(1, (100.0, 10.0), "car", [])
    state.observe(2, (100.0, 20.0), "car", [])
    assert state.live_speed_kmph(10.0, 0.5) is None


def patch(value):
    return np.full((4, 4, 3), value, dtype=np.uint8)


def test_colour_is_the_majority_of_the_five_largest_views():
    state = TrackState("1")
    assert state.colour is None
    for area, colour in [(10, "red"), (20, "red"), (30, "blue"), (40, "blue"), (50, "blue")]:
        state.add_appearance(float(area), colour, patch(area))
    assert state.colour == "blue"
    for area in (60, 70, 80):
        state.add_appearance(float(area), "white", patch(area))
    assert state.colour == "white"  # the three largest are white, two blue remain


def test_snapshot_is_the_largest_view():
    state = TrackState("1")
    state.add_appearance(50.0, "red", patch(50))
    state.add_appearance(20.0, "red", patch(20))
    assert state.snapshot[0, 0, 0] == 50


def test_small_views_are_not_wanted_once_five_larger_ones_are_held():
    state = TrackState("1")
    assert state.wants_appearance(1.0)
    for area in (10, 20, 30, 40, 50):
        state.add_appearance(float(area), "red", patch(area))
    assert not state.wants_appearance(5.0)
    assert state.wants_appearance(15.0)


def test_record_carries_the_new_fields():
    state = TrackState("1")
    state.observe(3, (100.0, 60.0), "car", LINES)
    state.observe(9, (100.0, 70.0), "car", LINES)
    state.add_appearance(10.0, "red", patch(1))
    state.mark_counted("down", 0.55, lane="down_2")
    record = state.to_record(fps=10.0, snapshot_path="runs/x/snapshots/1.jpg")
    assert (record.colour, record.lane) == ("red", "down_2")
    assert (record.first_seen_s, record.last_seen_s) == (0.3, 0.9)
    assert record.snapshot_path == "runs/x/snapshots/1.jpg"


def test_partly_visible_observations_are_left_out_of_the_ground_history():
    # A box cut by the frame edge has its bottom-centre pinned there: not a position.
    state = TrackState("1")
    state.observe(1, (100.0, 10.0), "car", [], metres, visible=False)
    state.observe(2, (100.0, 12.0), "car", [], metres, visible=False)
    assert state.ground_history == []
    assert state.live_speed_kmph(10.0, 0.5) is None
    state.observe(3, (100.0, 30.0), "car", [], metres)
    state.observe(4, (100.0, 40.0), "car", [], metres)
    assert state.live_speed_kmph(10.0, 0.5) == pytest.approx(36.0)
    assert len(state.history) == 4


def test_uncounted_vehicle_gives_a_record_only_when_asked():
    state = TrackState("1")
    state.observe(4, (100.0, 60.0), "car", LINES)
    assert state.to_record(fps=10.0) is None
    record = state.to_record(fps=10.0, include_uncounted=True)
    assert (record.direction, record.counted_at_s) == (None, None)
    assert (record.cls_name, record.first_seen_s) == ("car", 0.4)
