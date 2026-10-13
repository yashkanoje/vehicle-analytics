import pytest

from vehicle_analytics.crossing import detect_crossing
from vehicle_analytics.geometry import Line

LINE = Line("a", (0.0, 100.0), (200.0, 100.0))


def cross(p_prev, p_cur, k_prev=10, k_cur=11, line=LINE):
    return detect_crossing(line, "7", p_prev, p_cur, k_prev, k_cur)


def test_slow_crossing_is_stamped_between_the_two_frames():
    event = cross((50.0, 98.0), (50.0, 102.0))
    assert event.track_id == "7"
    assert event.line_name == "a"
    assert event.frame_time == pytest.approx(10.5)
    assert event.point_px == pytest.approx((50.0, 100.0))
    assert (event.side_from, event.side_to) == (-1, 1)


def test_fast_jump_across_the_line_is_detected():
    # Regression for defect 1: neither point is within 5 px of the line.
    event = cross((50.0, 60.0), (50.0, 140.0))
    assert event is not None
    assert event.frame_time == pytest.approx(10.5)


def test_interpolation_follows_the_distance_ratio():
    event = cross((50.0, 99.0), (50.0, 103.0))
    assert event.frame_time == pytest.approx(10.25)


def test_gap_between_observations_is_interpolated():
    event = cross((50.0, 90.0), (50.0, 110.0), k_prev=10, k_cur=14)
    assert event.frame_time == pytest.approx(12.0)


def test_upward_crossing_reports_the_opposite_sides():
    event = cross((50.0, 104.0), (50.0, 96.0))
    assert (event.side_from, event.side_to) == (1, -1)
    assert event.frame_time == pytest.approx(10.5)


def test_same_side_is_not_a_crossing():
    assert cross((50.0, 80.0), (50.0, 95.0)) is None
    assert cross((50.0, 120.0), (50.0, 105.0)) is None


def test_point_exactly_on_the_line_yields_one_crossing_going_down():
    arriving = cross((50.0, 98.0), (50.0, 100.0))
    assert arriving.frame_time == pytest.approx(11.0)
    assert (arriving.side_from, arriving.side_to) == (-1, 1)
    assert cross((50.0, 100.0), (50.0, 104.0), k_prev=11, k_cur=12) is None


def test_point_exactly_on_the_line_yields_one_crossing_going_up():
    assert cross((50.0, 104.0), (50.0, 100.0)) is None
    leaving = cross((50.0, 100.0), (50.0, 96.0), k_prev=11, k_cur=12)
    assert leaving.frame_time == pytest.approx(11.0)
    assert (leaving.side_from, leaving.side_to) == (1, -1)


def test_crossing_beyond_the_segment_ends_is_ignored():
    assert cross((300.0, 90.0), (300.0, 110.0)) is None


def test_diagonal_line():
    diagonal = Line("d", (0.0, 0.0), (100.0, 100.0))
    event = cross((60.0, 40.0), (40.0, 60.0), line=diagonal)
    assert event.point_px == pytest.approx((50.0, 50.0))
    assert event.frame_time == pytest.approx(10.5)
    assert (event.side_from, event.side_to) == (-1, 1)


def perspective(p):
    """Ground mapping with strong perspective: ground y = 1000 / image y."""
    return (p[0], 1000.0 / p[1])


def test_ground_interpolation_removes_perspective_bias_over_a_gap():
    # 20 m -> 5 m in ten frames at constant ground speed; the line is at 10 m.
    event = detect_crossing(LINE, "7", (50.0, 50.0), (50.0, 200.0), 0, 10, to_ground=perspective)
    assert event.point_px == pytest.approx((50.0, 100.0))
    assert event.frame_time == pytest.approx(20.0 / 3.0)


def test_without_a_ground_mapping_interpolation_is_linear_in_the_image():
    event = detect_crossing(LINE, "7", (50.0, 50.0), (50.0, 200.0), 0, 10)
    assert event.frame_time == pytest.approx(10.0 / 3.0)
