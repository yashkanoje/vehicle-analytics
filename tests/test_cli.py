import subprocess
import sys

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


def test_out_path_that_is_a_file_returns_2_before_any_model_loads(
    tmp_path, video_path, config_path, capsys
):
    blocker = tmp_path / "taken"
    blocker.write_text("not a folder", encoding="utf-8")
    code = main(["run", str(video_path), "--config", str(config_path), "--out", str(blocker)])
    assert code == 2
    assert "output folder" in capsys.readouterr().err


def test_database_path_that_is_a_folder_returns_2(tmp_path, video_path, config_path, capsys):
    code = main(
        [
            "run",
            str(video_path),
            "--config",
            str(config_path),
            "--out",
            str(tmp_path / "o"),
            "--db",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "cannot open database" in capsys.readouterr().err


def test_challans_for_an_unknown_run_returns_2(tmp_path, capsys):
    code = main(["challans", "--run", "99", "--db", str(tmp_path / "a.db")])
    assert code == 2
    assert "no run 99" in capsys.readouterr().err


def test_challans_with_a_missing_fines_file_returns_2(tmp_path, capsys):
    code = main(
        [
            "challans",
            "--run",
            "1",
            "--db",
            str(tmp_path / "a.db"),
            "--fines",
            str(tmp_path / "f.yaml"),
        ]
    )
    assert code == 2
    assert "fines file not found" in capsys.readouterr().err


def test_challans_command_drafts_pdfs_for_a_stored_run(tmp_path, capsys):
    from vehicle_analytics.rules import Violation
    from vehicle_analytics.state import VehicleRecord
    from vehicle_analytics.store import Store

    db = tmp_path / "a.db"
    with Store(db) as store:
        run_id = store.start_run("clip.mp4", 25.0, {})
        car = store.add_vehicle(
            run_id, VehicleRecord("7", "car", "down", 1.5, 121.4, 9.0, "two_line")
        )
        store.add_violation(
            car, Violation("overspeed", 12.5, 121.4, 100.0, 18.0, "km/h"), None, None
        )
    code = main(["challans", "--run", str(run_id), "--db", str(db), "--out", str(tmp_path / "o")])
    assert code == 0
    assert (tmp_path / "o" / "challans" / str(run_id) / f"VA-{run_id}-0001.pdf").is_file()
    assert "1 challan" in capsys.readouterr().out


def test_the_module_runs_as_a_program():
    # `python -m vehicle_analytics.cli` must reach main(); calling main() in-process cannot show it.
    done = subprocess.run(
        [sys.executable, "-m", "vehicle_analytics.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0
    assert "usage: vehicle-analytics" in done.stdout
