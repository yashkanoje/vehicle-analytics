import cv2
import numpy as np
import pytest
import yaml

from vehicle_analytics.cli import main

CONFIG_TEXT = (
    "lines:\n  a: [[0, 0.4], [1, 0.4]]\n  b: [[0, 0.6], [1, 0.6]]\n"
    "speed:\n  meters_between_lines: 10\n"
)


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    for _ in range(3):
        writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
    writer.release()
    return path


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "run" in capsys.readouterr().out


def test_unreadable_video_returns_2(tmp_path, config_path, capsys):
    code = main(["run", str(tmp_path / "missing.mp4"), "--config", str(config_path)])
    assert code == 2
    assert "cannot open video" in capsys.readouterr().err


def test_missing_config_returns_2(tmp_path, video_path, capsys):
    code = main(["run", str(video_path), "--config", str(tmp_path / "nope.yaml")])
    assert code == 2
    assert "config file not found" in capsys.readouterr().err


def test_invalid_config_returns_2(tmp_path, video_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("lines: {}\n", encoding="utf-8")
    assert main(["run", str(video_path), "--config", str(bad)]) == 2
    assert "lines.a" in capsys.readouterr().err


def test_calibrate_prints_a_block_for_given_points(video_path, capsys):
    code = main(
        [
            "calibrate",
            str(video_path),
            "--width-m",
            "3.5",
            "--length-m",
            "12",
            "--points",
            "64,216 256,216 192,72 128,72",
        ]
    )
    assert code == 0
    block = yaml.safe_load(capsys.readouterr().out)
    assert block["calibration"]["image_points"][0] == [0.2, 0.9]
    assert block["calibration"]["ground_points_m"][2] == [3.5, 12.0]


def test_calibrate_with_bad_points_returns_2(video_path, capsys):
    code = main(
        ["calibrate", str(video_path), "--width-m", "3.5", "--length-m", "12", "--points", "1,2"]
    )
    assert code == 2
    assert "four 'x,y' pairs" in capsys.readouterr().err
