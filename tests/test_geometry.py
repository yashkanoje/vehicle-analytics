import pytest

from vehicle_analytics.geometry import Homography, Line, Zone, reference_point

HORIZONTAL = Line("a", (0.0, 100.0), (200.0, 100.0))


def test_reference_point_is_bottom_centre():
    assert reference_point((10.0, 20.0, 50.0, 80.0)) == (30.0, 80.0)


def test_signed_distance_is_positive_below_a_left_to_right_line():
    assert HORIZONTAL.signed_distance((50.0, 130.0)) == pytest.approx(30.0)
    assert HORIZONTAL.signed_distance((50.0, 90.0)) == pytest.approx(-10.0)
    assert HORIZONTAL.signed_distance((50.0, 100.0)) == pytest.approx(0.0)


def test_signed_distance_for_a_diagonal_line():
    diagonal = Line("d", (0.0, 0.0), (100.0, 100.0))
    assert diagonal.signed_distance((0.0, 100.0)) == pytest.approx(100.0 / 2**0.5)
    assert diagonal.signed_distance((100.0, 0.0)) == pytest.approx(-100.0 / 2**0.5)


def test_within_segment():
    assert HORIZONTAL.within_segment((0.0, 100.0))
    assert HORIZONTAL.within_segment((200.0, 100.0))
    assert HORIZONTAL.within_segment((120.0, 100.0))
    assert not HORIZONTAL.within_segment((201.0, 100.0))
    assert not HORIZONTAL.within_segment((-1.0, 100.0))


def test_zero_length_line_is_rejected():
    with pytest.raises(ValueError, match="zero length"):
        Line("a", (5.0, 5.0), (5.0, 5.0))


IMAGE = [(100.0, 400.0), (500.0, 400.0), (400.0, 100.0), (200.0, 100.0)]
GROUND = [(0.0, 0.0), (4.0, 0.0), (4.0, 20.0), (0.0, 20.0)]


def test_homography_maps_each_corner_to_its_ground_point():
    h = Homography.from_points(IMAGE, GROUND)
    for image_point, ground_point in zip(IMAGE, GROUND, strict=True):
        assert h.to_ground(image_point) == pytest.approx(ground_point, abs=1e-6)


def test_homography_maps_the_near_edge_midpoint_to_the_ground_midpoint():
    h = Homography.from_points(IMAGE, GROUND)
    assert h.to_ground((300.0, 400.0)) == pytest.approx((2.0, 0.0), abs=1e-6)


def test_homography_needs_exactly_four_pairs():
    with pytest.raises(ValueError, match="exactly four"):
        Homography.from_points(IMAGE[:3], GROUND[:3])


def test_homography_rejects_collinear_image_points():
    collinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (5.0, 9.0)]
    with pytest.raises(ValueError, match="collinear"):
        Homography.from_points(collinear, GROUND)


def test_homography_rejects_repeated_ground_points():
    repeated = [(0.0, 0.0), (4.0, 0.0), (4.0, 0.0), (0.0, 20.0)]
    with pytest.raises(ValueError, match="collinear"):
        Homography.from_points(IMAGE, repeated)


def test_homography_rejects_corners_given_out_of_order():
    crossed = [IMAGE[0], IMAGE[1], IMAGE[3], IMAGE[2]]
    with pytest.raises(ValueError, match="in order around"):
        Homography.from_points(crossed, GROUND)
    with pytest.raises(ValueError, match="in order around"):
        Homography.from_points(IMAGE, [GROUND[0], GROUND[2], GROUND[1], GROUND[3]])


def test_homography_rejects_non_finite_points():
    bad = [(0.0, 0.0), (4.0, 0.0), (4.0, float("nan")), (0.0, 20.0)]
    with pytest.raises(ValueError, match="finite"):
        Homography.from_points(IMAGE, bad)


SQUARE = Zone("s", "lane", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
L_SHAPE = Zone(
    "l", "restricted", ((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0))
)


def test_zone_contains_points_inside_only():
    assert SQUARE.contains((5.0, 5.0))
    assert not SQUARE.contains((11.0, 5.0))
    assert not SQUARE.contains((5.0, -1.0))
    assert not SQUARE.contains((-0.1, 5.0))


def test_zone_handles_a_concave_outline():
    assert L_SHAPE.contains((2.0, 8.0))
    assert L_SHAPE.contains((8.0, 2.0))
    assert not L_SHAPE.contains((8.0, 8.0))
