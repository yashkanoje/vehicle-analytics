import copy
from pathlib import Path

import pytest

from vehicle_analytics.config import ConfigError, load_config, parse_config

MINIMAL = {
    "lines": {"a": [[0.0, 0.4], [1.0, 0.4]], "b": [[0.0, 0.6], [1.0, 0.6]]},
    "speed": {"meters_between_lines": 10.0},
}
LINE_B = MINIMAL["lines"]["b"]
CALIBRATION = {
    "image_points": [[0.2, 0.9], [0.8, 0.9], [0.6, 0.3], [0.4, 0.3]],
    "ground_points_m": [[0.0, 0.0], [3.5, 0.0], [3.5, 18.0], [0.0, 18.0]],
}


def with_changes(**changes):
    data = copy.deepcopy(MINIMAL)
    data.update(changes)
    return data


def test_minimal_config_gets_defaults():
    config = parse_config(MINIMAL)
    assert config.model.weights == "yolo11s.pt"
    assert config.model.conf == 0.4
    assert config.model.classes == ("car", "motorcycle", "bus", "truck")
    assert (config.tracker.max_age, config.tracker.n_init) == (30, 3)
    assert config.tracker.max_cosine_distance == 0.3
    assert config.count_line == "b"
    assert config.speed.meters_between_lines == 10.0
    assert config.speed.distance_rel_uncertainty == 0.05
    assert config.speed.jitter_px == 2.0
    assert config.calibration is None
    assert config.fps_override is None
    assert config.raw == MINIMAL


def test_integers_are_accepted_as_numbers():
    data = with_changes(
        lines={"a": [[0, 0], [1, 0]], "b": [[0, 1], [1, 1]]},
        model={"conf": 1},
        speed={"meters_between_lines": 10},
    )
    config = parse_config(data)
    assert config.model.conf == 1.0
    assert config.lines["b"] == ((0.0, 1.0), (1.0, 1.0))
    assert config.speed.meters_between_lines == 10.0


@pytest.mark.parametrize("data", [None, [], "text", 3])
def test_empty_or_non_mapping_file_is_rejected(data):
    with pytest.raises(ConfigError, match="mapping at the top level"):
        parse_config(data)


def test_missing_lines_names_the_key():
    with pytest.raises(ConfigError, match="'lines'"):
        parse_config({"speed": {"meters_between_lines": 10.0}})
    with pytest.raises(ConfigError, match="'lines.b'"):
        parse_config(with_changes(lines={"a": [[0.0, 0.4], [1.0, 0.4]]}))


def test_wrong_type_names_the_key():
    with pytest.raises(ConfigError, match="'model.conf' must be a number"):
        parse_config(with_changes(model={"conf": "high"}))
    with pytest.raises(ConfigError, match="'model.conf' must be a number"):
        parse_config(with_changes(model={"conf": True}))
    with pytest.raises(ConfigError, match="'tracker' must be a mapping"):
        parse_config(with_changes(tracker=[1, 2]))
    with pytest.raises(ConfigError, match="'model.classes'"):
        parse_config(with_changes(model={"classes": []}))


def test_out_of_range_values_are_rejected():
    with pytest.raises(ConfigError, match="'model.conf'"):
        parse_config(with_changes(model={"conf": 0}))
    with pytest.raises(ConfigError, match="'tracker.max_age'"):
        parse_config(with_changes(tracker={"max_age": 0}))
    with pytest.raises(ConfigError, match="'speed.meters_between_lines'"):
        parse_config(with_changes(speed={"meters_between_lines": -1}))
    with pytest.raises(ConfigError, match="'fps_override'"):
        parse_config(with_changes(fps_override=0))


def test_line_points_must_be_fractions():
    with pytest.raises(ConfigError, match=r"'lines.a\[1\]' must lie between 0 and 1"):
        parse_config(with_changes(lines={"a": [[0.0, 0.4], [640, 0.4]], "b": LINE_B}))
    with pytest.raises(ConfigError, match=r"'lines.a\[0\]' must be a pair of numbers"):
        parse_config(with_changes(lines={"a": [[0.0], [1.0, 0.4]], "b": LINE_B}))
    with pytest.raises(ConfigError, match="'lines.a' must be two points"):
        parse_config(with_changes(lines={"a": [[0.0, 0.4]], "b": LINE_B}))


def test_zero_length_line_is_rejected():
    with pytest.raises(ConfigError, match="'lines.a' has zero length"):
        parse_config(with_changes(lines={"a": [[0.5, 0.5], [0.5, 0.5]], "b": LINE_B}))


def test_count_line_must_name_a_line():
    with pytest.raises(ConfigError, match="'count_line'"):
        parse_config(with_changes(count_line="c"))
    with pytest.raises(ConfigError, match="'count_line'"):
        parse_config(with_changes(count_line=["a"]))
    assert parse_config(with_changes(count_line="a")).count_line == "a"


def test_direction_labels():
    config = parse_config(with_changes(directions={"positive": "south", "negative": "north"}))
    assert config.direction_label(1) == "south"
    assert config.direction_label(-1) == "north"
    with pytest.raises(ConfigError, match="'directions'"):
        parse_config(with_changes(directions={"positive": "x", "negative": "x"}))


def test_a_distance_source_is_required():
    with pytest.raises(ConfigError, match="meters_between_lines.*calibration"):
        parse_config({"lines": MINIMAL["lines"]})


def test_calibration_alone_is_enough():
    config = parse_config({"lines": MINIMAL["lines"], "calibration": CALIBRATION})
    assert config.speed.meters_between_lines is None
    assert len(config.calibration.image_points) == 4
    assert config.calibration.ground_points_m[2] == (3.5, 18.0)


def test_calibration_needs_four_points():
    bad = {**CALIBRATION, "image_points": CALIBRATION["image_points"][:3]}
    with pytest.raises(ConfigError, match="'calibration.image_points' must hold exactly four"):
        parse_config(with_changes(calibration=bad))


def test_degenerate_calibration_is_rejected():
    bad = {**CALIBRATION, "image_points": [[0.1, 0.5], [0.3, 0.5], [0.6, 0.5], [0.4, 0.3]]}
    with pytest.raises(ConfigError, match="'calibration'.*degenerate"):
        parse_config(with_changes(calibration=bad))


def test_lines_are_scaled_to_pixels():
    lines = parse_config(MINIMAL).lines_px(640, 360)
    assert lines["a"].p1 == pytest.approx((0.0, 144.0))
    assert lines["a"].p2 == pytest.approx((640.0, 144.0))
    assert lines["b"].name == "b"


def test_homography_is_built_in_pixels():
    config = parse_config(with_changes(calibration=CALIBRATION))
    homography = config.homography_px(1000, 500)
    assert homography.to_ground((200.0, 450.0)) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert homography.to_ground((600.0, 150.0)) == pytest.approx((3.5, 18.0), abs=1e-6)
    assert parse_config(MINIMAL).homography_px(1000, 500) is None


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "lines:\n  a: [[0, 0.4], [1, 0.4]]\n  b: [[0, 0.6], [1, 0.6]]\n"
        "speed:\n  meters_between_lines: 12\n",
        encoding="utf-8",
    )
    assert load_config(path).speed.meters_between_lines == 12.0


def test_load_config_reports_a_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.yaml")


def test_load_config_reports_invalid_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("lines: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_bundled_config_is_valid():
    path = Path(__file__).parent.parent / "configs" / "highway.yaml"
    assert load_config(path).count_line == "b"


def test_non_finite_numbers_are_rejected():
    with pytest.raises(ConfigError, match="'tracker.max_age' must be a finite number"):
        parse_config(with_changes(tracker={"max_age": float("inf")}))
    ground = [[0.0, 0.0], [3.5, 0.0], [3.5, float("nan")], [0.0, 18.0]]
    bad = {**CALIBRATION, "ground_points_m": ground}
    with pytest.raises(ConfigError, match=r"'calibration.ground_points_m\[2\]' must be finite"):
        parse_config(with_changes(calibration=bad))


def test_misordered_calibration_is_rejected():
    points = CALIBRATION["image_points"]
    bad = {**CALIBRATION, "image_points": [points[0], points[1], points[3], points[2]]}
    with pytest.raises(ConfigError, match="'calibration'.*in order around"):
        parse_config(with_changes(calibration=bad))


def test_load_config_reports_a_file_that_is_not_utf8(tmp_path):
    path = tmp_path / "latin.yaml"
    path.write_bytes(b"# tolerance \xb1 5 %\nlines: {}\n")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(path)
