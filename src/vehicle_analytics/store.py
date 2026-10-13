"""SQLite storage for runs and vehicles. All SQL in the package lives here and in schema.sql."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from importlib import resources
from pathlib import Path

from vehicle_analytics.state import VehicleRecord


class StoreError(RuntimeError):
    """The database cannot be opened, or a row cannot be written."""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        schema_file = resources.files("vehicle_analytics").joinpath("schema.sql")
        schema = schema_file.read_text(encoding="utf-8")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.path))
            self._db.row_factory = sqlite3.Row
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(schema)
        except (sqlite3.Error, OSError) as exc:
            raise StoreError(f"cannot open database {self.path}: {exc}") from exc

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

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
        with self._db:
            cursor = self._db.execute(
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
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, frame_count: int) -> None:
        with self._db:
            self._db.execute("UPDATE runs SET frame_count = ? WHERE id = ?", (frame_count, run_id))

    def add_vehicle(self, run_id: int, record: VehicleRecord) -> int:
        try:
            with self._db:
                cursor = self._db.execute(
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
        return int(cursor.lastrowid)

    def runs(self) -> list[sqlite3.Row]:
        return self._db.execute("SELECT * FROM runs ORDER BY id").fetchall()

    def vehicles(self, run_id: int) -> list[sqlite3.Row]:
        return self._db.execute(
            "SELECT * FROM vehicles WHERE run_id = ? ORDER BY counted_at_s", (run_id,)
        ).fetchall()
