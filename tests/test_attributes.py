import numpy as np
import pytest

from vehicle_analytics.attributes import (
    RoadReference,
    box_area,
    box_inside_frame,
    colour_name,
    crop_box,
    lane_at,
)
from vehicle_analytics.geometry import Zone


def solid(bgr, size=(40, 60)):
    return np.full((*size, 3), bgr, dtype=np.uint8)


@pytest.mark.parametrize(
    ("bgr", "name"),
    [
        ((255, 255, 255), "white"),
        ((10, 10, 10), "black"),
        ((128, 128, 128), "grey"),
        ((0, 0, 255), "red"),
        ((0, 255, 255), "yellow"),
        ((0, 255, 0), "green"),
        ((255, 0, 0), "blue"),
        ((0, 128, 255), "other"),  # orange
    ],
)
def test_solid_patches_are_named(bgr, name):
    assert colour_name(solid(bgr)) == name


def test_colour_comes_from_the_centre_of_the_crop():
    crop = solid((30, 30, 30))  # dark road around the vehicle
    crop[10:30, 15:45] = (0, 0, 255)
    assert colour_name(crop) == "red"


def test_red_is_recognised_on_both_sides_of_the_hue_wrap():
    crop = np.zeros((20, 20, 3), dtype=np.uint8)
    crop[:, :10] = (0, 20, 255)  # hue just above 0
    crop[:, 10:] = (20, 0, 255)  # hue just below 180
    assert colour_name(crop) == "red"


def test_empty_crop_is_other():
    assert colour_name(np.zeros((0, 5, 3), dtype=np.uint8)) == "other"


LANE = Zone("down_1", "lane", ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)), "down")
SHOULDER = Zone("hs", "shoulder", ((10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)))


def test_lane_at_ignores_zones_that_are_not_lanes():
    zones = [SHOULDER, LANE]
    assert lane_at((5.0, 5.0), zones) == "down_1"
    assert lane_at((15.0, 5.0), zones) is None
    assert lane_at((50.0, 5.0), zones) is None
    assert lane_at((5.0, 5.0), []) is None


def test_box_inside_frame_needs_a_margin_on_every_side():
    assert box_inside_frame((10.0, 10.0, 50.0, 40.0), 100, 80)
    assert not box_inside_frame((0.0, 10.0, 50.0, 40.0), 100, 80)
    assert not box_inside_frame((10.0, 10.0, 100.0, 40.0), 100, 80)
    assert not box_inside_frame((10.0, -5.0, 50.0, 40.0), 100, 80)
    assert not box_inside_frame((10.0, 10.0, 50.0, 79.5), 100, 80)


def test_box_area_and_crop():
    assert box_area((10.0, 20.0, 50.0, 40.0)) == 800.0
    frame = np.arange(100 * 80 * 3, dtype=np.uint8).reshape(80, 100, 3)
    crop = crop_box(frame, (10.2, 20.7, 50.0, 40.0))
    assert crop.shape == (20, 40, 3)  # corners are truncated to whole pixels
    crop[:] = 0
    assert frame[25, 20].any()  # the crop is a copy


# Median colours measured on the development clip, whose light has a strong blue cast:
# even the bare road surface is far from grey.
ROAD = (69.0, 50.0, 42.0)


def test_neutral_vehicles_are_not_named_after_the_scene_colour_cast():
    assert colour_name(solid((66, 44, 32)), road_bgr=ROAD) == "black"
    assert colour_name(solid((116, 99, 84)), road_bgr=(80.0, 61.0, 50.0)) == "grey"
    assert colour_name(solid((180, 157, 133)), road_bgr=(80.0, 62.0, 52.0)) == "white"


def test_a_coloured_vehicle_keeps_its_colour_under_a_cast():
    assert colour_name(solid((60, 40, 150)), road_bgr=ROAD) == "red"


def test_without_a_road_reference_the_cast_is_taken_at_face_value():
    # Bare road under the cast: with no reference it reads as a blue vehicle.
    assert colour_name(solid((69, 50, 42))) == "blue"


RIGHT_HALF = ((50.0, 0.0), (100.0, 0.0), (100.0, 100.0), (50.0, 100.0))
LEFT_HALF = ((0.0, 0.0), (50.0, 0.0), (50.0, 100.0), (0.0, 100.0))
ROAD_LANE = Zone("r", "lane", RIGHT_HALF, "down")


def grass_and_road():
    frame = solid((10, 200, 10), (100, 100))
    frame[:, 50:] = (69, 50, 42)
    return frame


def test_road_reference_is_the_median_inside_the_lane_zones():
    reference = RoadReference([ROAD_LANE], 100, 100)
    assert reference.colour(grass_and_road()) == pytest.approx((69.0, 50.0, 42.0))


def test_only_lanes_define_the_road():
    verge = Zone("verge", "restricted", LEFT_HALF)
    reference = RoadReference([verge, ROAD_LANE], 100, 100)
    assert reference.colour(grass_and_road()) == pytest.approx((69.0, 50.0, 42.0))
    assert RoadReference([verge], 100, 100).colour(grass_and_road()) is None
    assert RoadReference([], 100, 100).colour(grass_and_road()) is None


def test_a_lorry_filling_the_lane_for_a_moment_does_not_become_the_reference():
    reference = RoadReference([ROAD_LANE], 100, 100)
    for _ in range(10):
        reference.colour(grass_and_road())
    covered = grass_and_road()
    covered[:, 50:] = (250, 250, 250)
    assert reference.colour(covered) == pytest.approx((69.0, 50.0, 42.0))


def test_a_frame_of_another_size_gives_no_reference():
    reference = RoadReference([ROAD_LANE], 100, 100)
    assert reference.colour(solid((1, 2, 3), (50, 50))) is None


def vehicle(paint_bgr, glass_bgr=(40, 30, 25), size=(60, 60)):
    """Paint with a dark windscreen across the middle, as a vehicle seen head-on appears."""
    crop = solid(paint_bgr, size)
    height, width = size
    crop[height // 3 : 2 * height // 3, width // 4 : 3 * width // 4] = glass_bgr
    return crop


def test_colour_is_read_from_the_paint_not_the_windscreen():
    assert colour_name(vehicle((180, 157, 133)), road_bgr=(80.0, 62.0, 52.0)) == "white"
    assert colour_name(vehicle((60, 40, 150)), road_bgr=ROAD) == "red"
    assert colour_name(vehicle((0, 0, 255))) == "red"


def test_a_vehicle_no_brighter_than_the_road_is_black():
    # Highlights lift a black car above any fixed darkness threshold; the road is the yardstick.
    assert colour_name(solid((88, 66, 55)), road_bgr=(80.0, 62.0, 52.0)) == "black"


def test_silver_is_not_bright_enough_to_be_white():
    assert colour_name(solid((176, 136, 113)), road_bgr=(80.0, 62.0, 52.0)) == "grey"


def test_a_pale_tint_is_not_given_a_hue_name():
    # Silver with a slight green tint, as compression and reflections leave on neutral paint.
    assert colour_name(solid((130, 170, 135)), road_bgr=(70.0, 70.0, 70.0)) == "grey"
    assert colour_name(solid((40, 160, 60)), road_bgr=(70.0, 70.0, 70.0)) == "green"


def test_a_coloured_vehicle_in_shade_keeps_its_colour():
    # A blue car in shade is no brighter than the road, and still blue.
    assert colour_name(solid((95, 50, 38)), road_bgr=(80.0, 62.0, 52.0)) == "blue"
