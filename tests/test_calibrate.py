import pytest
import yaml

from vehicle_analytics.calibrate import CalibrationError, calibration_block, parse_points
from vehicle_analytics.config import parse_config

POINTS = [(128.0, 324.0), (512.0, 324.0), (384.0, 108.0), (256.0, 108.0)]


def test_parse_points():
    assert parse_points("128,324 512,324 384,108 256,108") == POINTS


@pytest.mark.parametrize("text", ["", "1,2 3,4 5,6", "1,2 3,4 5,6 seven,8", "1 2 3 4"])
def test_parse_points_rejects_malformed_input(text):
    with pytest.raises(CalibrationError, match="four 'x,y' pairs"):
        parse_points(text)


def test_block_holds_fractions_and_a_ground_rectangle():
    block = calibration_block(POINTS, 640, 360, width_m=3.5, length_m=12.0)
    assert block == {
        "calibration": {
            "image_points": [[0.2, 0.9], [0.8, 0.9], [0.6, 0.3], [0.4, 0.3]],
            "ground_points_m": [[0.0, 0.0], [3.5, 0.0], [3.5, 12.0], [0.0, 12.0]],
        }
    }


def test_block_is_accepted_by_the_config_parser():
    block = calibration_block(POINTS, 640, 360, width_m=3.5, length_m=12.0)
    data = {"lines": {"a": [[0, 0.4], [1, 0.4]], "b": [[0, 0.6], [1, 0.6]]}, **block}
    reloaded = yaml.safe_load(yaml.safe_dump(data))
    assert parse_config(reloaded).calibration is not None


def test_block_needs_four_points_and_positive_dimensions():
    with pytest.raises(CalibrationError, match="exactly four"):
        calibration_block(POINTS[:3], 640, 360, 3.5, 12.0)
    with pytest.raises(CalibrationError, match="greater than 0"):
        calibration_block(POINTS, 640, 360, 0.0, 12.0)
