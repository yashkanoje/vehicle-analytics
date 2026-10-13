import numpy as np
import pytest

from vehicle_analytics.attributes import box_area, box_inside_frame, colour_name, crop_box, lane_at
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
