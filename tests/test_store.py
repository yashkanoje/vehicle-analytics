import datetime
import json
import sqlite3

import pytest

from vehicle_analytics.rules import Violation
from vehicle_analytics.state import VehicleRecord
from vehicle_analytics.store import Store, StoreError, resolve_path


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


def test_tables(tmp_path):
    path = tmp_path / "analytics.db"
    with Store(path):
        pass
    names = sqlite3.connect(path).execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    assert {name for (name,) in names} == {"runs", "vehicles", "violations", "challans"}


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


SPEEDING = Violation("overspeed", 12.5, 120.0, 100.0, 10.0, "km/h")
ON_SHOULDER = Violation("restricted_zone", 30.0, 2.0, 2.0, unit="s", zone="hs")


def seeded(store):
    """A run with one speeding car and one uncounted vehicle on the shoulder."""
    run_id = store.start_run("clip.mp4", 25.0, {}, site_name="M6")
    car = store.add_vehicle(run_id, record("7"))
    lorry = store.add_vehicle(
        run_id, record("9", cls_name="truck", direction=None, counted_at_s=None, lane=None)
    )
    first = store.add_violation(car, SPEEDING, "ev/7_frame.jpg", "ev/7_crop.jpg")
    second = store.add_violation(lorry, ON_SHOULDER, None, None)
    return run_id, first, second


def test_uncounted_vehicle_can_be_stored(store):
    run_id, _, _ = seeded(store)
    rows = {row["track_id"]: row for row in store.vehicles(run_id)}
    assert rows["9"]["direction"] is None and rows["9"]["counted_at_s"] is None


def test_violation_round_trip(store):
    run_id, first, second = seeded(store)
    rows = store.violations(run_id)
    assert [row["id"] for row in rows] == [first, second]
    speeding = rows[0]
    assert (speeding["rule"], speeding["track_id"], speeding["class"]) == ("overspeed", "7", "car")
    assert (speeding["measured"], speeding["limit_value"]) == (120.0, 100.0)
    assert (speeding["margin"], speeding["unit"], speeding["at_s"]) == (10.0, "km/h", 12.5)
    assert (speeding["evidence_path"], speeding["crop_path"]) == ("ev/7_frame.jpg", "ev/7_crop.jpg")
    shoulder = rows[1]
    assert (shoulder["zone"], shoulder["margin"], shoulder["evidence_path"]) == ("hs", None, None)


def test_violation_needs_an_existing_vehicle(store):
    with pytest.raises(StoreError, match="vehicle 99"):
        store.add_violation(99, SPEEDING, None, None)


def test_has_run(store):
    run_id, _, _ = seeded(store)
    assert store.has_run(run_id) and not store.has_run(run_id + 1)


def test_challan_details_before_and_after_a_challan(store):
    run_id, first, second = seeded(store)
    before = store.challan_details(run_id)
    assert [row["violation_id"] for row in before] == [first, second]
    assert before[0]["challan_no"] is None and before[0]["site_name"] == "M6"
    assert (before[0]["track_id"], before[0]["class"], before[0]["colour"]) == ("7", "car", "red")
    assert (before[1]["direction"], before[1]["video"]) == (None, "clip.mp4")

    store.save_challan(
        first, "VA-1-0001", "2026-10-13T12:00:00", "s. 183", "Speeding", "Rs. 1,000", "c/1.pdf"
    )
    after = store.challan_details(run_id)
    assert (after[0]["challan_no"], after[0]["status"]) == ("VA-1-0001", "pending_review")
    assert after[0]["created_at"] == "2026-10-13T12:00:00"
    assert after[1]["challan_no"] is None
    assert (after[0]["pdf_path"], after[0]["speed_mode"]) == ("c/1.pdf", "homography")


def test_saving_a_challan_again_updates_the_fine_and_keeps_its_identity(store):
    run_id, first, _ = seeded(store)
    store.save_challan(first, "VA-1-0001", "2026-10-13T12:00:00", None, None, None, "c/a.pdf")
    store.save_challan(
        first, "VA-1-0099", "2030-01-01T00:00:00", "s. 183", "Speeding", "Rs. 1,000", "c/b.pdf"
    )
    (row,) = [r for r in store.challan_details(run_id) if r["violation_id"] == first]
    assert (row["challan_no"], row["created_at"]) == ("VA-1-0001", "2026-10-13T12:00:00")
    count = store._db.execute("SELECT COUNT(*) FROM challans").fetchone()[0]
    fine = store._db.execute("SELECT section, penalty, pdf_path FROM challans").fetchone()
    assert count == 1 and tuple(fine) == ("s. 183", "Rs. 1,000", "c/b.pdf")


def test_challan_numbers_are_unique(store):
    _, first, second = seeded(store)
    store.save_challan(first, "VA-1-0001", "2026-10-13T12:00:00", None, None, None, "a.pdf")
    with pytest.raises(StoreError, match="VA-1-0001"):
        store.save_challan(second, "VA-1-0001", "2026-10-13T12:00:00", None, None, None, "b.pdf")


def test_database_from_an_older_version_is_refused(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, video TEXT)")
    old.commit()
    old.close()
    with pytest.raises(StoreError, match="older version"):
        Store(path)


def test_a_current_database_reopens(tmp_path):
    path = tmp_path / "analytics.db"
    with Store(path) as first:
        run_id, _, _ = seeded(first)
    with Store(path) as second:
        assert len(second.violations(run_id)) == 2


def test_paths_are_resolved_against_the_folder_the_run_was_started_from(tmp_path):
    assert resolve_path("runs/a/7.jpg", tmp_path) == tmp_path / "runs" / "a" / "7.jpg"
    absolute = tmp_path / "x.jpg"
    assert resolve_path(str(absolute), "somewhere_else") == absolute
    assert resolve_path(None, tmp_path) is None
    assert resolve_path("", tmp_path) is None


def challan_row(store, run_id, violation_id):
    (row,) = [r for r in store.challan_details(run_id) if r["violation_id"] == violation_id]
    return row


def test_a_challan_can_be_approved_and_then_rejected(store):
    run_id, first, _ = seeded(store)
    store.save_challan(first, "VA-1-0001", "2026-10-13T12:00:00", None, None, None, "c/1.pdf")
    assert challan_row(store, run_id, first)["reviewed_at"] is None
    when = store.set_challan_status(first, "approved")
    row = challan_row(store, run_id, first)
    assert (row["status"], row["reviewed_at"]) == ("approved", when)
    store.set_challan_status(first, "rejected")
    assert challan_row(store, run_id, first)["status"] == "rejected"


def test_redrafting_a_challan_keeps_its_review(store):
    run_id, first, _ = seeded(store)
    store.save_challan(first, "VA-1-0001", "2026-10-13T12:00:00", None, None, None, "c/1.pdf")
    when = store.set_challan_status(first, "approved")
    store.save_challan(first, "VA-1-0001", "2026-10-13T12:00:00", "s. 183", None, None, "c/1.pdf")
    row = challan_row(store, run_id, first)
    assert (row["status"], row["reviewed_at"], row["section"]) == ("approved", when, "s. 183")


def test_a_review_needs_a_challan_and_a_decision(store):
    run_id, first, second = seeded(store)
    with pytest.raises(StoreError, match="no challan"):
        store.set_challan_status(second, "approved")
    store.save_challan(first, "VA-1-0001", "2026-10-13T12:00:00", None, None, None, "c/1.pdf")
    with pytest.raises(ValueError, match="pending_review"):
        store.set_challan_status(first, "pending_review")


def test_challan_details_carry_the_stored_fine(store):
    run_id, first, _ = seeded(store)
    store.save_challan(
        first, "VA-1-0001", "2026-10-13T12:00:00", "s. 183", "Speeding", "Rs. 1,000", "c/1.pdf"
    )
    row = challan_row(store, run_id, first)
    assert (row["section"], row["description"], row["penalty"]) == (
        "s. 183",
        "Speeding",
        "Rs. 1,000",
    )
