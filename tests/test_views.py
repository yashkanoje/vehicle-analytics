import pytest

from vehicle_analytics import views


def vehicle(
    track, cls="car", direction="down", speed=90.0, lane="down_1", colour="white", counted=True
):
    return {
        "id": int(track),
        "track_id": track,
        "class": cls,
        "colour": colour,
        "direction": direction if counted else None,
        "lane": lane,
        "counted_at_s": 1.0 if counted else None,
        "speed_kmph": speed,
        "speed_err_kmph": None if speed is None else 7.0,
    }


RUN = {"id": 1, "fps": 25.0, "frame_count": 1500}  # 60 s


def test_overview_counts_only_counted_vehicles_and_gives_flow_per_hour():
    rows = [
        vehicle("1"),
        vehicle("2", cls="truck", speed=None),
        vehicle("3", direction="up", speed=40.0),
        vehicle("4", counted=False, speed=None),
    ]
    summary = views.run_overview(RUN, rows, [{"rule": "overspeed"}, {"rule": "overspeed"}])
    assert (summary.duration_s, summary.counted, summary.uncounted) == (60.0, 3, 1)
    assert summary.by_direction == {"down": 2, "up": 1}
    assert summary.by_class == {"car": 2, "truck": 1}
    assert summary.flow_per_hour == {"down": 120.0, "up": 60.0}
    assert summary.speeds_kmph == [90.0, 40.0]
    assert summary.violations_by_rule == {"overspeed": 2}


@pytest.mark.parametrize("frames", [None, 0])
def test_a_run_without_a_recorded_length_has_no_flow_rate(frames):
    summary = views.run_overview({**RUN, "frame_count": frames}, [vehicle("1")], [])
    assert summary.duration_s is None and summary.flow_per_hour == {}
    assert summary.counted == 1


def test_speed_histogram_keeps_empty_bands_in_order():
    assert views.speed_histogram([41.0, 47.5, 72.0]) == {40.0: 2, 50.0: 0, 60.0: 0, 70.0: 1}
    assert views.speed_histogram([]) == {}


def test_options_name_missing_values():
    rows = [vehicle("1", lane=None), vehicle("2", lane="down_2")]
    assert views.options(rows, "lane") == [views.NOT_RECORDED, "down_2"]


def test_untouched_filters_keep_every_vehicle_including_those_without_speed():
    rows = [vehicle("1"), vehicle("2", speed=None, lane=None, colour=None)]
    assert views.filter_vehicles(rows) == rows


def test_filters_combine():
    rows = [vehicle("1"), vehicle("2", cls="truck"), vehicle("3", cls="truck", lane=None)]
    kept = views.filter_vehicles(rows, classes=["truck"], lanes=[views.NOT_RECORDED])
    assert [row["track_id"] for row in kept] == ["3"]


def test_a_speed_range_keeps_only_measured_speeds_inside_it():
    rows = [vehicle("1", speed=50.0), vehicle("2", speed=95.0), vehicle("3", speed=None)]
    kept = views.filter_vehicles(rows, speed_range=(90, 100))
    assert [row["track_id"] for row in kept] == ["2"]


def test_vehicle_table_rounds_speed_and_keeps_missing_values_empty():
    (row,) = views.vehicle_table([vehicle("7", speed=94.137)])
    assert (row["record"], row["track"], row["speed km/h"], row["± km/h"]) == (7, "7", 94.1, 7.0)
    (empty,) = views.vehicle_table([vehicle("8", speed=None, lane=None)])
    assert (empty["speed km/h"], empty["± km/h"], empty["lane"]) == (None, None, None)
