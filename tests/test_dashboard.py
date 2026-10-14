from pathlib import Path

import cv2
import numpy as np
import pytest
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

import vehicle_analytics
from vehicle_analytics.challan import generate_for_run
from vehicle_analytics.rules import Violation
from vehicle_analytics.state import VehicleRecord
from vehicle_analytics.store import Store
from vehicle_analytics.views import DB_ENV, ROOT_ENV

APP = Path(vehicle_analytics.__file__).with_name("dashboard.py")
OUT = Path("runs") / "clip"
SNAPSHOT = OUT / "snapshots" / "1" / "7.jpg"
FRAME = OUT / "evidence" / "1" / "7_overspeed_frame.jpg"
CROP = OUT / "evidence" / "1" / "7_overspeed_crop.jpg"


@pytest.fixture
def site(tmp_path, monkeypatch):
    """A database whose stored paths are relative to `site`, as a real run leaves them.

    The tests then run from another folder, with `site` passed as the root.
    """
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.chdir(site)
    for path in (SNAPSHOT, FRAME, CROP):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((20, 30, 3), 120, dtype=np.uint8))
    with Store(Path("data") / "analytics.db") as store:
        run_id = store.start_run("clip.mp4", 25.0, {}, site_name="Test site")
        store.finish_run(run_id, 1500)
        car = VehicleRecord(
            "7", "car", "down", 1.5, 121.4, 9.0, "homography",
            colour="red", lane="down_2", snapshot_path=SNAPSHOT.as_posix(),
        )
        lorry = VehicleRecord("8", "truck", "up", 3.0, None, None, None)
        car_id = store.add_vehicle(run_id, car)
        store.add_vehicle(run_id, lorry)
        speeding = Violation("overspeed", 12.5, 121.4, 100.0, 18.0, "km/h")
        store.add_violation(car_id, speeding, FRAME.as_posix(), CROP.as_posix())
        generate_for_run(store, run_id, {}, OUT)
        # A second run with other lanes and speeds, for switching runs in the sidebar.
        other = store.start_run("other.mp4", 25.0, {})
        store.finish_run(other, 750)
        for track, lane, speed in (("1", "up_1", 40.0), ("2", "up_3", 55.0)):
            store.add_vehicle(
                other, VehicleRecord(track, "car", "up", 1.0, speed, 4.0, "homography", lane=lane)
            )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv(DB_ENV, str(site / "data" / "analytics.db"))
    monkeypatch.setenv(ROOT_ENV, str(site))
    return site


def app(run_id=1):
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    if not at.exception and at.sidebar.selectbox:
        at.sidebar.selectbox(key="run").set_value(run_id).run()  # the default is the latest run
    return at


def go(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception
    return at


def texts(at):
    return " ".join(e.value for kind in (at.markdown, at.text, at.caption) for e in kind)


def test_overview_page(site):
    at = app()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Vehicles counted"] == "2"
    assert metrics["Violations"] == "1"
    assert metrics["Video length"] == "60.0 s"


def test_vehicles_page_shows_the_snapshot_found_through_the_root(site):
    at = go(app(), "Vehicles")
    assert len(at.dataframe[0].value) == 2
    assert not at.warning  # the snapshot was found from another working folder
    assert [image.captions for image in at.image] == [["Snapshot"]]
    assert "Exceeding the speed limit" in texts(at)


def test_vehicle_filters(site):
    at = go(app(), "Vehicles")
    at.multiselect(key="filter_class_1").set_value(["truck"]).run()
    assert not at.exception
    assert list(at.dataframe[0].value["track"]) == ["8"]
    at.multiselect(key="filter_direction_1").set_value(["down"]).run()  # no down-bound truck
    assert not at.exception and "No vehicle matches" in at.info[0].value


def test_switching_runs_does_not_carry_filters_or_the_chosen_vehicle(site):
    at = go(app(), "Vehicles")
    at.multiselect(key="filter_lane_1").set_value(["down_2"]).run()
    at.selectbox(key="vehicle_1").set_value(1).run()
    at.slider(key="filter_speed_1").set_value((121, 121)).run()  # run 1 speeds: 121.4 only
    assert not at.exception
    at.sidebar.selectbox(key="run").set_value(2).run()
    assert not at.exception
    assert sorted(at.dataframe[0].value["track"]) == ["1", "2"]


def test_violations_page_approves_a_challan(site):
    at = go(app(), "Violations")
    assert len(at.download_button) == 1
    assert len(at.image) == 2 and not at.warning
    at.button(key="approve_1").click().run()
    assert not at.exception
    assert "Approved by a reviewer" in texts(at)
    with Store(site / "data" / "analytics.db") as store:
        (row,) = store.challan_details(1)
    assert row["status"] == "approved"
    pdf = site / row["pdf_path"]
    assert "Approved by a reviewer" in " ".join(p.extract_text() for p in PdfReader(pdf).pages)


def test_a_missing_image_is_reported_by_path(site):
    (site / CROP).unlink()
    at = go(app(), "Violations")
    assert "not found" in at.warning[0].value and "7_overspeed_crop.jpg" in at.warning[0].value
    assert len(at.image) == 1


def test_a_missing_database_is_reported_and_not_created(site, tmp_path, monkeypatch):
    missing = tmp_path / "typo.db"
    monkeypatch.setenv(DB_ENV, str(missing))
    at = app()
    assert not at.exception and "No database" in at.error[0].value
    assert not missing.exists()


def test_an_old_database_is_reported(site, tmp_path, monkeypatch):
    import sqlite3

    old = tmp_path / "old.db"
    connection = sqlite3.connect(old)
    connection.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()
    monkeypatch.setenv(DB_ENV, str(old))
    at = app()
    assert not at.exception and "older version" in at.error[0].value


def test_a_database_without_runs(site, tmp_path, monkeypatch):
    empty = tmp_path / "empty.db"
    Store(empty).close()
    monkeypatch.setenv(DB_ENV, str(empty))
    at = app()
    assert not at.exception and "No runs" in at.info[0].value


def pdf_text(path):
    return " ".join(p.extract_text() for p in PdfReader(path).pages)


def test_violations_page_rejects_a_challan(site):
    at = go(app(), "Violations")
    at.button(key="reject_1").click().run()
    assert not at.exception
    assert "Rejected by a reviewer" in texts(at)
    with Store(site / "data" / "analytics.db") as store:
        (row,) = store.challan_details(1)
    assert row["status"] == "rejected"


def test_a_failed_reprint_can_be_retried(site):
    pdf = site / "runs" / "clip" / "challans" / "1" / "VA-1-0001.pdf"
    pdf.unlink()
    pdf.mkdir()  # a folder in the PDF's place makes the write fail on every platform
    at = go(app(), "Violations")
    at.button(key="approve_1").click().run()
    assert not at.exception
    assert "decision saved" in at.warning[0].value
    assert "Approved by a reviewer" in texts(at)
    assert not at.button(key="approve_1").disabled
    pdf.rmdir()
    at.button(key="approve_1").click().run()
    assert not at.exception
    assert "Approved by a reviewer" in pdf_text(pdf)


def test_the_draft_hint_names_the_database(site):
    db = site / "data" / "analytics.db"
    with Store(db) as store:
        lorry = next(v for v in store.vehicles(1) if v["track_id"] == "8")
        store.add_violation(lorry["id"], Violation("restricted_zone", 5.0, zone="bay"), None, None)
    at = go(app(), "Violations")
    info = at.info[0].value
    assert "1 of 2 violations have no challan yet" in info
    assert f"--db {db}" in info

