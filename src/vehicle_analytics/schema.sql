-- Schema version 2. The version is kept in PRAGMA user_version and checked by store.py.

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

-- One row per counted vehicle, and per uncounted vehicle that broke a rule.
-- Times are seconds from the start of the video.
CREATE TABLE IF NOT EXISTS vehicles (
    id             INTEGER PRIMARY KEY,
    run_id         INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    track_id       TEXT    NOT NULL,
    class          TEXT    NOT NULL,
    colour         TEXT,
    direction      TEXT,                -- NULL when the vehicle never crossed the count line
    lane           TEXT,
    first_seen_s   REAL,
    last_seen_s    REAL,
    counted_at_s   REAL,                -- NULL when the vehicle never crossed the count line
    speed_kmph     REAL,
    speed_err_kmph REAL,
    speed_mode     TEXT,
    snapshot_path  TEXT,
    plate          TEXT,                -- not read by this software; always NULL
    UNIQUE (run_id, track_id)
);

CREATE INDEX IF NOT EXISTS vehicles_by_run ON vehicles (run_id);

-- One row per rule broken by a vehicle. measured, limit_value and margin are kept so the
-- decision can be re-derived from the row.
CREATE TABLE IF NOT EXISTS violations (
    id            INTEGER PRIMARY KEY,
    vehicle_id    INTEGER NOT NULL REFERENCES vehicles (id) ON DELETE CASCADE,
    rule          TEXT    NOT NULL,
    zone          TEXT,
    measured      REAL,
    limit_value   REAL,
    margin        REAL,
    unit          TEXT,
    at_s          REAL    NOT NULL,
    evidence_path TEXT,
    crop_path     TEXT
);

CREATE INDEX IF NOT EXISTS violations_by_vehicle ON violations (vehicle_id);

-- A draft challan for one violation, awaiting human review.
CREATE TABLE IF NOT EXISTS challans (
    id           INTEGER PRIMARY KEY,
    violation_id INTEGER NOT NULL UNIQUE REFERENCES violations (id) ON DELETE CASCADE,
    challan_no   TEXT    NOT NULL UNIQUE,
    created_at   TEXT    NOT NULL,
    section      TEXT,
    description  TEXT,
    penalty      TEXT,                  -- text: the schedule gives ranges and class-specific sums
    pdf_path     TEXT,
    status       TEXT    NOT NULL DEFAULT 'pending_review'
                 CHECK (status IN ('pending_review', 'approved', 'rejected')),
    reviewed_at  TEXT
);
