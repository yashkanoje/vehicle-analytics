import math

import numpy as np
import pytest

from vehicle_analytics.crossing import CrossingEvent
from vehicle_analytics.geometry import Homography
from vehicle_analytics.speed import zone_speed


def event(line_name, frame_time, point):
    return CrossingEvent("7", line_name, frame_time, point, -1, 1)


A = event("a", 0.0, (100.0, 80.0))
B = event("b", 9.0, (100.0, 280.0))


def test_two_line_speed():
    speed = zone_speed(A, B, 30.0, meters_between_lines=10.0)
    assert speed.kmph == pytest.approx(120.0)
    assert speed.mode == "two_line"


def test_fractional_crossing_times_are_used():
    # Regression for defect 3: whole frames (10 and 19) would give 100.0 km/h.
    a = event("a", 10.5, (100.0, 80.0))
    b = event("b", 19.25, (100.0, 280.0))
    speed = zone_speed(a, b, 25.0, meters_between_lines=10.0)
    assert speed.kmph == pytest.approx(102.857, abs=1e-3)


def test_order_of_crossings_does_not_matter():
    forward = zone_speed(A, B, 30.0, meters_between_lines=10.0)
    backward = zone_speed(B, A, 30.0, meters_between_lines=10.0)
    assert backward.kmph == pytest.approx(forward.kmph)


def test_homography_distance_replaces_the_fixed_distance():
    # 0.1 m per pixel in both directions.
    h = Homography.from_points(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    )
    a = event("a", 0.0, (50.0, 20.0))
    b = event("b", 9.0, (50.0, 80.0))
    speed = zone_speed(a, b, 30.0, meters_between_lines=999.0, homography=h)
    assert speed.kmph == pytest.approx(72.0)
    assert speed.mode == "homography"


def test_uncertainty_combines_localisation_and_scale_in_quadrature():
    speed = zone_speed(
        A, B, 30.0, meters_between_lines=10.0, jitter_px=2.0, distance_rel_uncertainty=0.05
    )
    localisation = math.sqrt(2) * 2.0 / 200.0
    expected_rel = math.hypot(localisation, 0.05)
    assert speed.err_kmph == pytest.approx(120.0 * expected_rel)


def test_no_jitter_and_no_scale_error_gives_zero_uncertainty():
    speed = zone_speed(
        A, B, 30.0, meters_between_lines=10.0, jitter_px=0.0, distance_rel_uncertainty=0.0
    )
    assert speed.err_kmph == pytest.approx(0.0)


def test_same_crossing_time_gives_no_speed():
    b = event("b", 0.0, (100.0, 280.0))
    assert zone_speed(A, b, 30.0, meters_between_lines=10.0) is None


def test_same_crossing_point_gives_no_speed():
    b = event("b", 9.0, (100.0, 80.0))
    assert zone_speed(A, b, 30.0, meters_between_lines=10.0) is None


def test_a_distance_source_is_required():
    with pytest.raises(ValueError, match="meters_between_lines or a homography"):
        zone_speed(A, B, 30.0)


def test_homography_error_bar_uses_the_ground_scale_at_each_crossing():
    # Perspective along the direction of travel: ground y = 1000 / image y, so a
    # pixel covers far more road at the far line than at the near one.
    h = Homography(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1000.0], [0.0, 1.0, 0.0]]))
    a = event("a", 0.0, (0.0, 50.0))  # 20 m out
    b = event("b", 9.0, (0.0, 200.0))  # 5 m out
    speed = zone_speed(a, b, 30.0, homography=h, jitter_px=2.0, distance_rel_uncertainty=0.0)
    assert speed.kmph == pytest.approx(15.0 / 0.3 * 3.6)
    at_a = 20.0 - 1000.0 / 52.0
    at_b = 5.0 - 1000.0 / 202.0
    assert speed.err_kmph == pytest.approx(speed.kmph * math.hypot(at_a, at_b) / 15.0, rel=1e-6)
