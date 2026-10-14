import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
from pypdf import PdfReader

from vehicle_analytics.challan import (
    ChallanError,
    Fine,
    challan_fields,
    generate_for_run,
    load_fines,
    render_pdf,
    review,
)
from vehicle_analytics.config import ConfigError
from vehicle_analytics.rules import Violation
from vehicle_analytics.state import VehicleRecord
from vehicle_analytics.store import Store

ROW = {
    "violation_id": 3,
    "rule": "overspeed",
    "zone": None,
    "measured": 121.44,
    "limit_value": 100.0,
    "margin": 17.96,
    "unit": "km/h",
    "at_s": 12.5,
    "vehicle_id": 14,
    "track_id": "7",
    "class": "car",
    "colour": "red",
    "lane": "down_3",
    "direction": "down",
    "video": "clip.mp4",
    "site_name": "M6 overbridge",
    "site_location": "Staffordshire",
    "recording_start": "2025-12-08T17:30:00",
    "speed_mode": "homography",
    "status": None,
    "reviewed_at": None,
}
PENALTY = {"default": "Rs. 1,000", "truck": "Rs. 2,000"}
FINE = Fine("s. 183(1)", "Driving at excessive speed", PENALTY, "x")


def as_dict(fields):
    return dict(fields)


def test_fields_for_a_speeding_car():
    fields = as_dict(challan_fields(ROW, FINE, "VA-1-0001", "2026-10-13T12:00:00"))
    assert fields["Challan no."] == "VA-1-0001"
    assert fields["Status"] == "Pending review"
    assert "Reviewed" not in fields
    assert fields["Violation"] == "Exceeding the speed limit"
    assert fields["Measured"] == "121.4 km/h"
    assert fields["Limit"] == "100.0 km/h"
    assert fields["Tolerance applied"] == "18.0 km/h"
    assert fields["Date and time"] == "2025-12-08 17:30:12"
    assert fields["Site"] == "M6 overbridge, Staffordshire"
    assert fields["Vehicle record"] == "14 (track 7 in clip.mp4)"
    assert fields["Vehicle class"] == "car"
    assert fields["Lane"] == "down_3"
    assert fields["Direction"] == "down"
    assert fields["Apparent colour (unverified)"] == "red"
    assert fields["Registration number"] == "Not read"
    assert fields["Section"] == "s. 183(1)"
    assert fields["Penalty"] == "Rs. 1,000"


def test_penalty_follows_the_vehicle_class():
    fields = as_dict(challan_fields({**ROW, "class": "truck"}, FINE, "VA-1-0001", "2026-10-13"))
    assert fields["Penalty"] == "Rs. 2,000"


def test_missing_fine_gives_no_amount():
    fields = as_dict(challan_fields(ROW, None, "VA-1-0001", "2026-10-13T12:00:00"))
    assert fields["Section"] == "As per applicable schedule"
    assert fields["Penalty"] == "As per applicable schedule"


def test_fields_without_a_recording_time_or_numbers():
    row = {
        **ROW,
        "rule": "wrong_way",
        "zone": "up_1",
        "measured": None,
        "limit_value": None,
        "margin": None,
        "unit": None,
        "recording_start": None,
        "site_name": None,
        "site_location": None,
        "direction": None,
        "lane": None,
        "colour": None,
    }
    fields = as_dict(challan_fields(row, None, "VA-1-0002", "2026-10-13T12:00:00"))
    assert fields["Violation"] == "Driving against the permitted direction"
    assert fields["Date and time"] == "12.5 s from the start of the video"
    assert fields["Zone"] == "up_1"
    assert fields["Site"] == "Not recorded"
    assert fields["Direction"] == "Not recorded"
    assert "Measured" not in fields and "Tolerance applied" not in fields


def test_dwell_rules_show_time_without_a_tolerance():
    row = {
        **ROW,
        "rule": "restricted_zone",
        "zone": "hs",
        "measured": 2.4,
        "limit_value": 2.0,
        "margin": None,
        "unit": "s",
    }
    fields = as_dict(challan_fields(row, None, "VA-1-0003", "2026-10-13T12:00:00"))
    assert (fields["Measured"], fields["Limit"]) == ("2.4 s", "2.0 s")
    assert "Tolerance applied" not in fields


def test_load_fines(tmp_path):
    path = tmp_path / "fines.yaml"
    path.write_text(
        "overspeed:\n  section: s. 183\n  description: Speeding\n"
        "  penalty:\n    default: Rs. 1,000\n    truck: Rs. 2,000\n  source: the Act\n"
        "stopped:\n  section: s. 122\n",
        encoding="utf-8",
    )
    fines = load_fines(path)
    assert fines["overspeed"].penalty_for("truck") == "Rs. 2,000"
    assert fines["overspeed"].penalty_for("bus") == "Rs. 1,000"
    assert (fines["stopped"].section, fines["stopped"].penalty_for("car")) == ("s. 122", None)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- a\n- b\n", "must be a mapping of rule names"),
        ("speeding: {section: x}\n", "unknown rule 'speeding'"),
        ("overspeed: text\n", "'overspeed' must be a mapping"),
        ("overspeed: {penalty: {truck: x}}\n", "'overspeed.penalty' needs a 'default'"),
        ("overspeed: {section: 5}\n", "'overspeed.section' must be text"),
    ],
)
def test_malformed_fines_files(tmp_path, text, message):
    path = tmp_path / "fines.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_fines(path)


def test_missing_fines_file(tmp_path):
    with pytest.raises(ConfigError, match="fines file not found"):
        load_fines(tmp_path / "nope.yaml")


def image(path, colour):
    cv2.imwrite(str(path), np.full((60, 80, 3), colour, dtype=np.uint8))
    return path.as_posix()


def test_pdf_is_written(tmp_path):
    fields = challan_fields(ROW, FINE, "VA-1-0001", "2026-10-13T12:00:00")
    path = tmp_path / "c.pdf"
    render_pdf(path, fields, image(tmp_path / "f.jpg", 90), image(tmp_path / "c.jpg", 200))
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 2000


def test_pdf_without_evidence_images(tmp_path):
    fields = challan_fields(ROW, None, "VA-1-0001", "2026-10-13T12:00:00")
    path = tmp_path / "c.pdf"
    render_pdf(path, fields, None, (tmp_path / "gone.jpg").as_posix())
    assert path.read_bytes().startswith(b"%PDF")


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "analytics.db") as opened:
        yield opened


def seeded(store, tmp_path):
    run_id = store.start_run("clip.mp4", 25.0, {})
    car = VehicleRecord("7", "car", "down", 1.5, 121.4, 9.0, "homography")
    lorry = VehicleRecord("9", "truck", None, None, None, None, None)
    car_id, lorry_id = store.add_vehicle(run_id, car), store.add_vehicle(run_id, lorry)
    evidence = image(tmp_path / "f.jpg", 90)
    speeding = Violation("overspeed", 12.5, 121.4, 100.0, 18.0, "km/h")
    on_shoulder = Violation("restricted_zone", 5.0, 2.0, 2.0, unit="s", zone="hs")
    store.add_violation(car_id, speeding, evidence, None)
    store.add_violation(lorry_id, on_shoulder, None, None)
    return run_id


def test_generate_makes_one_numbered_pdf_per_violation(store, tmp_path):
    run_id = seeded(store, tmp_path)
    paths = generate_for_run(store, run_id, {"overspeed": FINE}, tmp_path / "out")
    assert [p.name for p in paths] == [f"VA-{run_id}-0001.pdf", f"VA-{run_id}-0002.pdf"]
    assert all(p.read_bytes().startswith(b"%PDF") for p in paths)
    assert paths[0].parent == tmp_path / "out" / "challans" / str(run_id)
    rows = store.challan_details(run_id)
    # Numbered in order of violation time: the lorry on the shoulder at 5.0 s comes first.
    assert [(r["rule"], r["challan_no"]) for r in rows] == [
        ("restricted_zone", f"VA-{run_id}-0001"),
        ("overspeed", f"VA-{run_id}-0002"),
    ]
    assert all(r["status"] == "pending_review" for r in rows)


def test_generating_twice_keeps_numbers_and_does_not_duplicate(store, tmp_path):
    run_id = seeded(store, tmp_path)
    generate_for_run(store, run_id, {}, tmp_path / "out")
    first = [(r["challan_no"], r["created_at"]) for r in store.challan_details(run_id)]
    generate_for_run(store, run_id, {"overspeed": FINE}, tmp_path / "out")
    second = [(r["challan_no"], r["created_at"]) for r in store.challan_details(run_id)]
    assert first == second and len(second) == 2
    assert store._db.execute("SELECT COUNT(*) FROM challans").fetchone()[0] == 2
    with_section = "SELECT COUNT(*) FROM challans WHERE section = 's. 183(1)'"
    assert store._db.execute(with_section).fetchone()[0] == 1


def test_a_later_violation_gets_the_next_number(store, tmp_path):
    run_id = seeded(store, tmp_path)
    generate_for_run(store, run_id, {}, tmp_path / "out")
    van = store.add_vehicle(run_id, VehicleRecord("11", "car", "up", 0.5, None, None, None))
    store.add_violation(van, Violation("wrong_way", 1.0, zone="up_1"), None, None)
    paths = generate_for_run(store, run_id, {}, tmp_path / "out")
    numbers = {r["rule"]: r["challan_no"] for r in store.challan_details(run_id)}
    assert numbers["wrong_way"] == f"VA-{run_id}-0003"
    assert len(paths) == 3


def test_a_run_without_violations_makes_nothing(store, tmp_path):
    run_id = store.start_run("clip.mp4", 25.0, {})
    assert generate_for_run(store, run_id, {}, tmp_path / "out") == []
    assert not (tmp_path / "out" / "challans").exists()


def test_overspeed_challan_says_how_the_speed_was_measured():
    calibrated = as_dict(challan_fields(ROW, None, "VA-1-0001", "2026-10-13T12:00:00"))
    assert calibrated["Speed measured by"] == "Ground-plane calibration"
    row = {**ROW, "speed_mode": "two_line"}
    assumed = as_dict(challan_fields(row, None, "VA-1-0001", "2026-10-13T12:00:00"))
    assert assumed["Speed measured by"] == "Assumed distance between two lines (not calibrated)"
    other = {**ROW, "rule": "wrong_way", "measured": None, "limit_value": None, "margin": None}
    assert "Speed measured by" not in as_dict(
        challan_fields(other, None, "VA-1-0002", "2026-10-13")
    )


def pdf_text(path):
    return " ".join(page.extract_text() for page in PdfReader(str(path)).pages)


def test_markup_characters_in_values_are_printed_not_interpreted(tmp_path):
    row = {**ROW, "site_name": "M&S <north> car park", "lane": "Lane <b", "video": "R&D <test>.mp4"}
    path = tmp_path / "c.pdf"
    render_pdf(path, challan_fields(row, None, "VA-1-0001", "2026-10-13T12:00:00"), None, None)
    text = pdf_text(path)
    assert "M&S <north> car park" in text
    assert "Lane <b" in text
    assert "R&D <test>.mp4" in text


def test_missing_evidence_images_are_reported(store, tmp_path, caplog):
    run_id = seeded(store, tmp_path)
    (tmp_path / "f.jpg").unlink()
    with caplog.at_level(logging.WARNING, logger="vehicle_analytics.challan"):
        generate_for_run(store, run_id, {}, tmp_path / "out")
    assert "evidence image not found" in caplog.text and "f.jpg" in caplog.text


def test_a_folder_that_cannot_be_made_is_a_challan_error(store, tmp_path):
    run_id = seeded(store, tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "challans").write_text("a file is in the way", encoding="utf-8")
    with pytest.raises(ChallanError, match="cannot write challan"):
        generate_for_run(store, run_id, {}, tmp_path / "out")


def test_a_challan_number_of_another_shape_is_ignored_when_numbering(store, tmp_path):
    run_id = seeded(store, tmp_path)
    first = store.challan_details(run_id)[0]["violation_id"]
    store.save_challan(first, "LEGACY", "2026-10-13T12:00:00", None, None, None, "x.pdf")
    generate_for_run(store, run_id, {}, tmp_path / "out")
    numbers = sorted(row["challan_no"] for row in store.challan_details(run_id))
    assert numbers == ["LEGACY", f"VA-{run_id}-0001"]


def test_fields_show_the_review():
    row = {**ROW, "status": "approved", "reviewed_at": "2026-10-15T09:30:00"}
    fields = as_dict(challan_fields(row, FINE, "VA-1-0001", "2026-10-13T12:00:00"))
    assert fields["Status"] == "Approved by a reviewer"
    assert fields["Reviewed"] == "2026-10-15 09:30:00"


def test_redraft_after_a_review_prints_the_decision_and_keeps_the_number(store, tmp_path):
    run_id = seeded(store, tmp_path)
    first = generate_for_run(store, run_id, {"overspeed": FINE}, tmp_path / "out")[0]
    violation_id = store.challan_details(run_id)[0]["violation_id"]
    store.set_challan_status(violation_id, "rejected")
    again = generate_for_run(store, run_id, {"overspeed": FINE}, tmp_path / "out")[0]
    assert again == first
    text = pdf_text(first)
    assert "Rejected by a reviewer" in text and "Pending review" not in text


def test_review_reprints_the_pdf_from_another_working_folder(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id = seeded(store, tmp_path)
    generate_for_run(store, run_id, {"overspeed": FINE}, Path("out"))  # relative paths stored
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    # Rows are in time order: the shoulder violation (5.0 s) is 0001, the speeding one 0002.
    speeding = next(r for r in store.challan_details(run_id) if r["rule"] == "overspeed")
    review(store, run_id, speeding["violation_id"], "approved", root=tmp_path)
    text = pdf_text(tmp_path / "out" / "challans" / str(run_id) / f"VA-{run_id}-0002.pdf")
    assert "Approved by a reviewer" in text and "s. 183(1)" in text
    assert not (elsewhere / "out").exists()


def test_a_reprint_that_fails_keeps_the_decision_and_says_so(store, tmp_path):
    run_id = seeded(store, tmp_path)
    pdf = generate_for_run(store, run_id, {"overspeed": FINE}, tmp_path / "out")[0]
    pdf.unlink()
    pdf.mkdir()  # a folder where the PDF should be: it cannot be written
    violation_id = store.challan_details(run_id)[0]["violation_id"]
    with pytest.raises(ChallanError, match="decision saved"):
        review(store, run_id, violation_id, "approved", root=tmp_path)
    assert store.challan_details(run_id)[0]["status"] == "approved"
