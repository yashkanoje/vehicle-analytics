import datetime
import json
import sqlite3

import pytest

from vehicle_analytics.state import VehicleRecord
from vehicle_analytics.store import Store, StoreError


def record(track_id="7", **changes):
    values = {
        "track_id": track_id,
        "cls_name": "car",
        "direction": "down",
        "counted_at_s": 1.5,
        "speed_kmph": 94.2,
        "speed_err_kmph": 7.0,
        "speed_mode": "homography",
        "colour": "red",
        "lane": "down_2",
        "first_seen_s": 0.4,
        "last_seen_s": 2.8,
        "snapshot_path": "runs/x/snapshots/7.jpg",
    }
    return VehicleRecord(**{**values, **changes})


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "analytics.db") as opened:
        yield opened


def test_run_round_trip(store):
    start = datetime.datetime(2025, 12, 8, 17, 30)
    run_id = store.start_run(
        "clip.mp4",
        25.0,
        {"count_line": "b", "when": start},
        site_name="M6",
        site_location="UK",
        recording_start=start,
    )
    store.finish_run(run_id, 1501)
    (row,) = store.runs()
    assert (row["id"], row["video"]) == (run_id, "clip.mp4")
    assert (row["fps"], row["frame_count"]) == (25.0, 1501)
    assert (row["site_name"], row["site_location"]) == ("M6", "UK")
    assert row["recording_start"] == "2025-12-08T17:30:00"
    assert json.loads(row["config_json"])["count_line"] == "b"
    assert datetime.datetime.fromisoformat(row["started_at"])


def test_vehicle_round_trip(store):
    run_id = store.start_run("clip.mp4", 25.0, {})
    vehicle_id = store.add_vehicle(run_id, record())
    (row,) = store.vehicles(run_id)
    assert row["id"] == vehicle_id
    assert (row["track_id"], row["class"]) == ("7", "car")
    assert (row["colour"], row["lane"]) == ("red", "down_2")
    assert (row["direction"], row["counted_at_s"]) == ("down", 1.5)
    assert (row["speed_kmph"], row["speed_err_kmph"]) == (94.2, 7.0)
    assert row["speed_mode"] == "homography"
    assert (row["first_seen_s"], row["last_seen_s"]) == (0.4, 2.8)
    assert row["snapshot_path"] == "runs/x/snapshots/7.jpg"
    assert row["plate"] is None


def test_missing_values_are_stored_as_null(store):
    run_id = store.start_run("clip.mp4", 25.0, {})
    empty = record(
        speed_kmph=None,
        speed_err_kmph=None,
        speed_mode=None,
        colour=None,
        lane=None,
        snapshot_path=None,
    )
    store.add_vehicle(run_id, empty)
    (row,) = store.vehicles(run_id)
    assert row["speed_kmph"] is None and row["colour"] is None and row["lane"] is None


def test_two_runs_keep_their_own_vehicles(store):
    first = store.start_run("clip.mp4", 25.0, {})
    second = store.start_run("clip.mp4", 25.0, {})
    store.add_vehicle(first, record("7"))
    store.add_vehicle(second, record("7"))
    store.add_vehicle(second, record("8"))
    assert [len(store.vehicles(r)) for r in (first, second)] == [1, 2]
    assert len(store.runs()) == 2


def test_a_track_cannot_be_stored_twice_in_one_run(store):
    run_id = store.start_run("clip.mp4", 25.0, {})
    store.add_vehicle(run_id, record("7"))
    with pytest.raises(StoreError, match="track 7"):
        store.add_vehicle(run_id, record("7"))


def test_vehicle_needs_an_existing_run(store):
    with pytest.raises(StoreError, match="run 99"):
        store.add_vehicle(99, record())


def test_rows_survive_reopening(tmp_path):
    path = tmp_path / "analytics.db"
    with Store(path) as first:
        run_id = first.start_run("clip.mp4", 25.0, {})
        first.add_vehicle(run_id, record())
    with Store(path) as second:
        assert len(second.vehicles(run_id)) == 1


def test_missing_folder_is_created(tmp_path):
    path = tmp_path / "new" / "deeper" / "analytics.db"
    with Store(path):
        pass
    assert path.is_file()


def test_path_that_is_a_folder_is_a_store_error(tmp_path):
    with pytest.raises(StoreError, match="cannot open database"):
        Store(tmp_path)


def test_file_that_is_not_a_database_is_a_store_error(tmp_path):
    path = tmp_path / "notes.db"
    path.write_text("this is not a database, it is a note " * 50, encoding="utf-8")
    with pytest.raises(StoreError, match="cannot open database"):
        Store(path)


def test_only_the_phase_b_tables_exist(tmp_path):
    path = tmp_path / "analytics.db"
    with Store(path):
        pass
    names = sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    assert {name for (name,) in names} == {"runs", "vehicles"}


def test_write_failures_are_store_errors(tmp_path):
    store = Store(tmp_path / "analytics.db")
    run_id = store.start_run("clip.mp4", 25.0, {})
    store.close()
    with pytest.raises(StoreError, match="cannot write to database"):
        store.start_run("clip.mp4", 25.0, {})
    with pytest.raises(StoreError, match="cannot write to database"):
        store.add_vehicle(run_id, record())
    with pytest.raises(StoreError, match="cannot write to database"):
        store.finish_run(run_id, 5)
