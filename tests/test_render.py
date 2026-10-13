import numpy as np

from vehicle_analytics.geometry import Line, Zone
from vehicle_analytics.render import (
    draw_lines,
    draw_panel,
    draw_track,
    draw_zones,
    panel_rows,
    track_label,
)
from vehicle_analytics.speed import SpeedEstimate
from vehicle_analytics.stats import Stats


def test_label_without_speed():
    assert track_label("7", "car", None) == "#7 car"


def test_label_with_speed_and_error_bar():
    speed = SpeedEstimate(kmph=102.86, err_kmph=5.34, mode="two_line")
    assert track_label("7", "car", speed) == "#7 car 102.9 +/- 5.3 km/h"


def test_panel_rows_list_total_then_each_direction():
    stats = Stats()
    for direction in ["up", "down", "down"]:
        stats.record_count(direction)
    assert panel_rows(stats, 60.0) == [
        "Total: 3",
        "down: 2  (120 veh/h)",
        "up: 1  (60 veh/h)",
    ]


def test_drawing_changes_pixels_and_keeps_the_shape():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    draw_lines(frame, [Line("a", (0.0, 80.0), (320.0, 80.0))])
    draw_track(frame, (100.0, 100.0, 160.0, 150.0), "#1 car", counted=True)
    draw_panel(frame, ["Total: 1"])
    assert frame.shape == (200, 320, 3)
    assert frame.any()


def test_label_shows_live_speed_until_the_zone_speed_is_known():
    assert track_label("7", "car", None, live_kmph=93.6) == "#7 car ~94 km/h"
    speed = SpeedEstimate(kmph=102.86, err_kmph=5.34, mode="homography")
    assert track_label("7", "car", speed, live_kmph=93.6) == "#7 car 102.9 +/- 5.3 km/h"


def test_zones_are_drawn():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    square = ((50.0, 50.0), (150.0, 50.0), (150.0, 150.0), (50.0, 150.0))
    draw_zones(frame, [Zone("z", "lane", square, "down")])
    assert frame.any()


def test_a_violating_vehicle_is_drawn_in_red():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    draw_track(frame, (100.0, 100.0, 160.0, 150.0), "#1 car", counted=True, violating=True)
    blue, green, red = (int(v) for v in frame[100, 130])
    assert red > 200 and green < 60 and blue < 60
