"""SQLite storage for runs and vehicles. All SQL in the package lives here and in schema.sql."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from importlib import resources
from pathlib import Path

from vehicle_analytics.rules import Violation
from vehicle_analytics.state import VehicleRecord

SCHEMA_VERSION = 2


class StoreError(RuntimeError):
    """The database cannot be opened, or a row cannot be written."""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        schema_file = resources.files("vehicle_analytics").joinpath("schema.sql")
        schema = schema_file.read_text(encoding="utf-8")
        self._db: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path))
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys = ON")
            version = self._db.execute("PRAGMA user_version").fetchone()[0]
            tables = self._db.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table'"
            ).fetchone()[0]
            outdated = tables > 0 and version != SCHEMA_VERSION
            if not outdated:
                self._db.executescript(schema)
                self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except (sqlite3.Error, OSError) as exc:
            if self._db is not None:
                self._db.close()
            raise StoreError(f"cannot open database {self.path}: {exc}") from exc
        if outdated:
            self._db.close()
            raise StoreError(
                f"database {self.path} was made by an older version of vehicle-analytics "
                f"(schema {version}, this version uses {SCHEMA_VERSION}); "
                "choose a new file with --db"
            )

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    def _write(self, sql: str, values: tuple) -> int:
        """Run one statement in its own transaction.

        Any failure is a StoreError, except a broken table rule, which the caller names.
        """
        try:
            with self._db:
                return int(self._db.execute(sql, values).lastrowid or 0)
        except sqlite3.IntegrityError:
            raise
        except sqlite3.Error as exc:
            raise StoreError(f"cannot write to database {self.path}: {exc}") from exc

    def start_run(
        self,
        video: str,
        fps: float,
        config: dict,
        *,
        site_name: str | None = None,
        site_location: str | None = None,
        recording_start: datetime | None = None,
    ) -> int:
        return self._write(
            "INSERT INTO runs (video, site_name, site_location, recording_start, fps,"
            " started_at, config_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                video,
                site_name,
                site_location,
                recording_start.isoformat() if recording_start else None,
                fps,
                datetime.now().isoformat(timespec="seconds"),
                json.dumps(config, default=str),
            ),
        )

    def finish_run(self, run_id: int, frame_count: int) -> None:
        self._write("UPDATE runs SET frame_count = ? WHERE id = ?", (frame_count, run_id))

    def add_vehicle(self, run_id: int, record: VehicleRecord) -> int:
        try:
            return self._write(
                "INSERT INTO vehicles (run_id, track_id, class, colour, direction, lane,"
                " first_seen_s, last_seen_s, counted_at_s, speed_kmph, speed_err_kmph,"
                " speed_mode, snapshot_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    record.track_id,
                    record.cls_name,
                    record.colour,
                    record.direction,
                    record.lane,
                    record.first_seen_s,
                    record.last_seen_s,
                    record.counted_at_s,
                    record.speed_kmph,
                    record.speed_err_kmph,
                    record.speed_mode,
                    record.snapshot_path,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StoreError(
                f"cannot store track {record.track_id} in run {run_id}: {exc}"
            ) from exc

    def has_run(self, run_id: int) -> bool:
        row = self._db.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row is not None

    def add_violation(
        self,
        vehicle_id: int,
        violation: Violation,
        evidence_path: str | None,
        crop_path: str | None,
    ) -> int:
        try:
            return self._write(
                "INSERT INTO violations (vehicle_id, rule, zone, measured, limit_value, margin,"
                " unit, at_s, evidence_path, crop_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    vehicle_id,
                    violation.rule,
                    violation.zone,
                    violation.measured,
                    violation.limit_value,
                    violation.margin,
                    violation.unit,
                    violation.at_s,
                    evidence_path,
                    crop_path,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StoreError(
                f"cannot store a violation for vehicle {vehicle_id}: {exc}"
            ) from exc

    def save_challan(
        self,
        violation_id: int,
        challan_no: str,
        created_at: str,
        section: str | None,
        description: str | None,
        penalty: str | None,
        pdf_path: str,
    ) -> None:
        """Create the challan for a violation, or refresh the fine and PDF of the one it has.

        The number, creation time and review status of an existing challan are never changed.
        """
        try:
            self._write(
                "INSERT INTO challans (violation_id, challan_no, created_at, section,"
                " description, penalty, pdf_path) VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (violation_id) DO UPDATE SET section = excluded.section,"
                " description = excluded.description, penalty = excluded.penalty,"
                " pdf_path = excluded.pdf_path",
                (violation_id, challan_no, created_at, section, description, penalty, pdf_path),
            )
        except sqlite3.IntegrityError as exc:
            raise StoreError(f"cannot store challan {challan_no}: {exc}") from exc

    def runs(self) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM runs ORDER BY id").fetchall()

    def vehicles(self, run_id: int) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM vehicles WHERE run_id = ? ORDER BY counted_at_s", (run_id,)
        ).fetchall()

    def violations(self, run_id: int) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT v.*, veh.track_id, veh.class FROM violations v"
            " JOIN vehicles veh ON veh.id = v.vehicle_id"
            " WHERE veh.run_id = ? ORDER BY v.at_s, v.id",
            (run_id,),
        ).fetchall()

    def challan_details(self, run_id: int) -> list[sqlite3.Row]:
        """Everything a challan shows, one row per violation of the run."""
        return self._db.execute(
            "SELECT v.id AS violation_id, v.rule, v.zone, v.measured, v.limit_value, v.margin,"
            " v.unit, v.at_s, v.evidence_path, v.crop_path,"
            " veh.id AS vehicle_id, veh.track_id, veh.class, veh.colour, veh.lane,"
            " veh.direction, r.video, r.site_name, r.site_location, r.recording_start,"
            " c.id AS challan_id, c.challan_no, c.created_at, c.status"
            " FROM violations v"
            " JOIN vehicles veh ON veh.id = v.vehicle_id"
            " JOIN runs r ON r.id = veh.run_id"
            " LEFT JOIN challans c ON c.violation_id = v.id"
            " WHERE r.id = ? ORDER BY v.at_s, v.id",
            (run_id,),
        ).fetchall()
