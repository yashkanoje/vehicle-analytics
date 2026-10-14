import subprocess
import sys
from pathlib import Path

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


def stored_run(db):
    from vehicle_analytics.rules import Violation
    from vehicle_analytics.state import VehicleRecord
    from vehicle_analytics.store import Store

    with Store(db) as store:
        run_id = store.start_run("clip.mp4", 25.0, {})
        car = VehicleRecord("7", "car", "down", 1.5, 121.4, 9.0, "two_line")
        speeding = Violation("overspeed", 12.5, 121.4, 100.0, 18.0, "km/h")
        store.add_violation(store.add_vehicle(run_id, car), speeding, None, None)
    return run_id


def test_challans_redraft_goes_to_the_folder_of_the_first_draft(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_id = stored_run(tmp_path / "a.db")
    assert main(["challans", "--run", str(run_id), "--db", "a.db", "--out", "custom"]) == 0
    pdf = tmp_path / "custom" / "challans" / str(run_id) / f"VA-{run_id}-0001.pdf"
    pdf.write_bytes(b"stale")
    assert main(["challans", "--run", str(run_id), "--db", "a.db"]) == 0
    assert pdf.read_bytes().startswith(b"%PDF")
    assert not (tmp_path / "runs").exists()


def test_challans_that_cannot_be_written_return_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_id = stored_run(tmp_path / "a.db")
    (tmp_path / "blocked").mkdir()
    (tmp_path / "blocked" / "challans").write_text("in the way", encoding="utf-8")
    code = main(["challans", "--run", str(run_id), "--db", "a.db", "--out", "blocked"])
    assert code == 2
    assert "cannot write challan" in capsys.readouterr().err


def test_dashboard_starts_streamlit_on_the_database(tmp_path, monkeypatch):
    from vehicle_analytics.store import Store

    Store(tmp_path / "a.db").close()
    calls = []

    def fake_run(command, env, check):
        calls.append((command, env))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("vehicle_analytics.cli.subprocess.run", fake_run)
    code = main(["dashboard", "--db", str(tmp_path / "a.db"), "--root", str(tmp_path)])
    assert code == 0
    ((command, env),) = calls
    assert command[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert command[4].endswith("dashboard.py") and Path(command[4]).is_file()
    flags = " ".join(command[5:])
    assert "--server.address localhost" in flags  # not reachable from the network
    assert "--browser.gatherUsageStats false" in flags  # nothing sent out
    assert "--server.headless true" in flags  # no first-run email prompt
    assert env["VEHICLE_ANALYTICS_DB"] == str((tmp_path / "a.db").resolve())
    assert env["VEHICLE_ANALYTICS_ROOT"] == str(tmp_path.resolve())


def test_dashboard_on_a_missing_database_returns_2_and_creates_nothing(tmp_path, capsys):
    assert main(["dashboard", "--db", str(tmp_path / "typo.db")]) == 2
    assert "no database at" in capsys.readouterr().err
    assert not (tmp_path / "typo.db").exists()


def test_dashboard_with_a_root_that_is_not_a_folder_returns_2(tmp_path, capsys):
    from vehicle_analytics.store import Store

    Store(tmp_path / "a.db").close()
    code = main(["dashboard", "--db", str(tmp_path / "a.db"), "--root", str(tmp_path / "no")])
    assert code == 2
    assert "not a folder" in capsys.readouterr().err


def test_dashboard_without_streamlit_returns_2(tmp_path, monkeypatch, capsys):
    from vehicle_analytics.store import Store

    Store(tmp_path / "a.db").close()
    monkeypatch.setattr("vehicle_analytics.cli.find_spec", lambda name: None)
    assert main(["dashboard", "--db", str(tmp_path / "a.db")]) == 2
    assert "pip install" in capsys.readouterr().err
