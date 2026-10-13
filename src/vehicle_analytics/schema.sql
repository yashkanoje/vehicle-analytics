-- One row per processed video.
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY,
    video           TEXT    NOT NULL,
    site_name       TEXT,
    site_location   TEXT,
    recording_start TEXT,               -- ISO 8601, when the recording began, if known
    fps             REAL    NOT NULL,
    frame_count     INTEGER,            -- NULL until the run ends
    started_at      TEXT    NOT NULL,   -- ISO 8601, when processing began
    config_json     TEXT    NOT NULL
);

-- One row per counted vehicle. Times are seconds from the start of the video.
CREATE TABLE IF NOT EXISTS vehicles (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    track_id       TEXT    NOT NULL,
    class          TEXT    NOT NULL,
    colour         TEXT,
    direction      TEXT    NOT NULL,
    lane           TEXT,
    first_seen_s   REAL,
    last_seen_s    REAL,
    counted_at_s   REAL    NOT NULL,
    speed_kmph     REAL,
    speed_err_kmph REAL,
    speed_mode     TEXT,
    snapshot_path  TEXT,
    plate          TEXT,                -- not read by this software; always NULL
    UNIQUE (run_id, track_id)
);

CREATE INDEX IF NOT EXISTS vehicles_by_run ON vehicles (run_id);
